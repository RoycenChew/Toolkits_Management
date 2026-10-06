"""Unsupervised entity resolution: learned blocking, Fellegi-Sunter scoring, clustering.

Three separate pieces of prior art, reimplemented from their algorithms and
composed into one pipeline:

* Affine-gap string distance (Gotoh's variant of Needleman-Wunsch), used by
  dedupe via its `affinegap` extension. Penalises opening a gap far more than
  extending one, which is exactly right for names and addresses where the
  difference is usually an omitted middle name, not scattered typos.
* Greedy blocking-predicate cover, the idea behind dedupe's `training.py`:
  choosing a small set of cheap keys that covers known duplicates while keeping
  the candidate-pair count bounded. This is weighted set cover.
* Fellegi-Sunter probabilistic record linkage with EM parameter estimation, the
  model behind Splink. Learns P(agreement | match) and P(agreement | non-match)
  from the data itself, with no labels, and emits an interpretable match weight.

Nothing here requires a SQL engine, a model, or a network call.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from .models import (
    CandidatePair,
    EntityCluster,
    FieldComparison,
    Predicate,
    Record,
    ResolutionConfig,
    ResolutionRequest,
    ResolutionResult,
    ScoredPair,
    TrainedModel,
)

NO_MATCH_LEVEL = "none"

_DEFAULT_M_TOP = 0.85
"""Untrained P(strongest level | match). Records that match usually agree on a
field; 0.85 says "usually" without claiming to know how often."""
_DEFAULT_U_TOP = 0.85
"""Untrained P(no level reached | non-match), the mirror of the above."""
"""The implicit lowest comparison level: the field agreed at no threshold."""

_EPS = 1e-9


# --------------------------------------------------------------------------
# String similarity
# --------------------------------------------------------------------------


def affine_gap_distance(
    a: str,
    b: str,
    match_cost: float = -5.0,
    mismatch_cost: float = 5.0,
    gap_open: float = 4.0,
    gap_extend: float = 1.0,
) -> float:
    """Minimum edit cost under an affine gap penalty (Gotoh, 1982).

    Opening a gap costs `gap_open`; each further character in that same gap costs
    only `gap_extend`. A matched character earns a negative cost, so identical
    strings score strongly negative and the measure stays discriminative for long
    inputs. Runs in O(len(a) * len(b)) time and O(len(b)) space.
    """
    if a == b:
        return match_cost * len(a)
    if not a or not b:
        return gap_open + gap_extend * (len(a) + len(b) - 1)

    n = len(b)
    large = float("inf")
    # D[j]: best cost aligning a[:i] with b[:j], ending in a match/mismatch.
    # V[j]: best cost overall. H tracks gaps in `a`, tracked per-row as a scalar.
    d = [large] * (n + 1)
    v = [0.0] + [gap_open + gap_extend * (j - 1) for j in range(1, n + 1)]

    for i in range(1, len(a) + 1):
        prev_v = v[0]
        v[0] = gap_open + gap_extend * (i - 1)
        h = large  # cost of a gap in `a` ending at this cell
        for j in range(1, n + 1):
            # Vertical gap (skip a character of b).
            d[j] = min(d[j] + gap_extend, v[j] + gap_open)
            # Horizontal gap (skip a character of a).
            h = min(h + gap_extend, v[j - 1] + gap_open)
            # Substitution / match.
            sub = prev_v + (match_cost if a[i - 1] == b[j - 1] else mismatch_cost)
            prev_v = v[j]
            v[j] = min(d[j], h, sub)
    return v[n]


def affine_gap_similarity(a: str, b: str) -> float:
    """Normalise affine-gap distance into a [0, 1] similarity.

    The best achievable cost for two strings is a full match of the shorter one;
    the worst reasonable cost is aligning nothing at all. We interpolate between
    those two references, which keeps the scale stable across string lengths.
    """
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    best = affine_gap_distance(a, a) if len(a) <= len(b) else affine_gap_distance(b, b)
    worst = 4.0 + 1.0 * (len(a) + len(b) - 1)
    actual = affine_gap_distance(a, b)
    if worst - best <= 0:
        return 0.0
    sim = 1.0 - (actual - best) / (worst - best)
    return max(0.0, min(1.0, sim))


# --------------------------------------------------------------------------
# Blocking predicate library
# --------------------------------------------------------------------------


def _first_n(n: int):
    return lambda s: [s[:n]] if len(s) >= n else [s]


def _last_n(n: int):
    return lambda s: [s[-n:]] if len(s) >= n else [s]


def _ngrams(n: int):
    def fn(s: str) -> list[str]:
        if len(s) < n:
            return [s]
        return sorted({s[i : i + n] for i in range(len(s) - n + 1)})

    return fn


def _tokens(s: str) -> list[str]:
    return sorted({t for t in "".join(c if c.isalnum() else " " for c in s).split() if t})


def _sorted_acronym(s: str) -> list[str]:
    initials = "".join(sorted(t[0] for t in _tokens(s)))
    return [initials] if initials else []


def _digits(s: str) -> list[str]:
    d = "".join(c for c in s if c.isdigit())
    return [d] if d else []


def _alpha_skeleton(s: str) -> list[str]:
    """Letters only, deduplicated in order. Survives punctuation and spacing
    differences ('O'Brien Ltd.' and 'obrien ltd' collide)."""
    letters = "".join(c for c in s if c.isalpha())
    return [letters] if letters else []


def default_predicates(fields: Iterable[str]) -> list[Predicate]:
    """A general-purpose predicate library, instantiated for each field.

    These are deliberately cheap and diverse: prefix, suffix, token, n-gram,
    numeric and letter-skeleton keys fail in different ways, which is what makes
    the greedy cover able to find a complementary subset.
    """
    library = [
        ("first4", _first_n(4)),
        ("first6", _first_n(6)),
        ("last4", _last_n(4)),
        ("token", _tokens),
        ("gram3", _ngrams(3)),
        ("gram4", _ngrams(4)),
        ("acronym", _sorted_acronym),
        ("digits", _digits),
        ("alpha", _alpha_skeleton),
    ]
    return [Predicate(name, f, fn) for f in fields for name, fn in library]


# --------------------------------------------------------------------------
# Component
# --------------------------------------------------------------------------


class EntityResolutionComponent:
    """Records in, entity clusters out.

    The pipeline is: learn (or accept) blocking predicates -> generate candidate
    pairs -> compute a comparison pattern per pair -> learn m/u probabilities by
    EM -> score each pair with an interpretable match weight -> cluster.
    """

    def execute(self, input_data: ResolutionRequest) -> ResolutionResult:
        cfg = input_data.config
        records = dict(input_data.records)
        if not cfg.comparisons:
            raise ValueError("config.comparisons must not be empty")

        fields = [c.field for c in cfg.comparisons]
        predicates = list(cfg.predicates) if cfg.predicates else default_predicates(fields)

        selected = self._learn_predicate_cover(
            records, predicates, input_data.labelled_pairs, cfg
        )
        pairs = self._candidate_pairs(records, selected, cfg)

        patterns = {
            (p.left, p.right): self._pattern(records[p.left], records[p.right], cfg)
            for p in pairs
        }
        model = self._train_em(list(patterns.values()), cfg)
        scored = [
            self._score(left, right, pattern, model)
            for (left, right), pattern in patterns.items()
        ]
        scored.sort(key=lambda s: (-s.match_probability, s.left, s.right))

        clusters = self._cluster(records.keys(), scored, cfg)

        n = len(records)
        full = n * (n - 1) // 2
        return ResolutionResult(
            clusters=clusters,
            scored_pairs=scored,
            model=model,
            selected_predicates=[p.name + ":" + p.field for p in selected],
            pairs_compared=len(pairs),
            pairs_avoided=max(0, full - len(pairs)),
        )

    # --- scoring one record ----------------------------------------------

    def score_record(
        self,
        record: Record,
        candidates: Mapping[str, Record],
        config: ResolutionConfig,
        model: TrainedModel,
        record_id: str = "query",
    ) -> list[ScoredPair]:
        """Score one record against candidates with weights that already exist.

        `execute` resolves a batch, and its EM step is what makes it a batch
        operation: m and u probabilities are estimated from the distribution of
        comparison patterns across many pairs. The question an application asks
        at runtime is the other one - "here is one new record, which of these
        is it?" - and EM over a batch of one has nothing to estimate from. It
        would return the seed parameters while looking like a measurement,
        which is worse than refusing.

        So the model is a parameter. Train it once with `execute` and keep it,
        or build fixed weights with `default_model` when there is nothing to
        train on yet. The weight arithmetic is the same function the batch path
        uses, so a cached model cannot quietly come to mean something else.

        `candidates` may include `record_id` itself - passing the whole corpus
        is the obvious call site - and that pair is skipped, because a record
        matching itself at probability 1.0 would head every result and say
        nothing.
        """
        if not config.comparisons:
            raise ValueError("config.comparisons must not be empty")
        if model is None:
            raise ValueError(
                "score_record needs a TrainedModel; train one with execute() or"
                " build fixed weights with default_model(config)"
            )
        self._check_model_covers(config, model)

        scored = [
            self._score(
                record_id,
                candidate_id,
                self._pattern(record, candidate, config),
                model,
            )
            for candidate_id, candidate in candidates.items()
            if candidate_id != record_id
        ]
        scored.sort(key=lambda s: (-s.match_probability, s.right))
        return scored

    @staticmethod
    def _check_model_covers(config: ResolutionConfig, model: TrainedModel) -> None:
        """Refuse a model that has never seen one of the configured fields.

        `_score` falls back to epsilon for an unknown field, and epsilon over
        epsilon is 1, which contributes exactly zero bits. So adding a
        comparison and forgetting to retrain would score as if the new field
        did not exist, and look like it worked.
        """
        missing = [
            c.field
            for c in config.comparisons
            if c.field not in model.m_probabilities or c.field not in model.u_probabilities
        ]
        if missing:
            raise ValueError(
                "this model has no weights for: "
                + ", ".join(sorted(missing))
                + "; retrain it with execute() or rebuild it with default_model()"
            )

    # --- blocking --------------------------------------------------------

    def _blocks(
        self, records: Mapping[str, Record], predicate: Predicate, cfg: ResolutionConfig
    ) -> dict[str, list[str]]:
        blocks: dict[str, list[str]] = defaultdict(list)
        for rid, record in records.items():
            for key in predicate.keys(record):
                blocks[key].append(rid)
        return {
            k: v for k, v in blocks.items() if 1 < len(v) <= cfg.max_block_size
        }

    def _pairs_of(self, blocks: Mapping[str, Sequence[str]]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for members in blocks.values():
            ordered = sorted(members)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    out.add((ordered[i], ordered[j]))
        return out

    def _learn_predicate_cover(
        self,
        records: Mapping[str, Record],
        predicates: Sequence[Predicate],
        labelled: Sequence[tuple[str, str, bool]],
        cfg: ResolutionConfig,
    ) -> list[Predicate]:
        """Greedy weighted set cover over blocking predicates.

        Each predicate covers some set of known duplicate pairs and costs the
        number of candidate pairs it generates. We repeatedly take the predicate
        with the best newly-covered-per-cost ratio. This is the standard greedy
        approximation for weighted set cover, and it is why a learned blocking
        scheme beats a hand-written one: it finds predicates that are cheap
        *given what the others already cover*.

        With no labels there is nothing to cover, so we fall back to the
        cheapest predicates, which keeps the pair budget sane without guessing
        at recall.
        """
        positives = {
            tuple(sorted((left_id, right_id)))
            for left_id, right_id, is_match in labelled
            if is_match and left_id in records and right_id in records
        }
        costs: dict[str, int] = {}
        covers: dict[str, set[tuple[str, str]]] = {}
        for pred in predicates:
            pairs = self._pairs_of(self._blocks(records, pred, cfg))
            key = pred.name + ":" + pred.field
            costs[key] = len(pairs)
            covers[key] = pairs

        usable = [p for p in predicates if costs[p.name + ":" + p.field] > 0]
        if not usable:
            return list(predicates[: cfg.max_predicates])

        if not positives:
            usable.sort(key=lambda p: costs[p.name + ":" + p.field])
            return usable[: cfg.max_predicates]

        remaining = set(positives)
        chosen: list[Predicate] = []
        pool = list(usable)
        while remaining and pool and len(chosen) < cfg.max_predicates:
            best = None
            best_ratio = 0.0
            for pred in pool:
                key = pred.name + ":" + pred.field
                gain = len(covers[key] & remaining)
                if gain == 0:
                    continue
                ratio = gain / (costs[key] + 1)
                if ratio > best_ratio:
                    best = pred
                    best_ratio = ratio
            if best is None:
                break
            chosen.append(best)
            remaining -= covers[best.name + ":" + best.field]
            pool.remove(best)

        if not chosen:
            usable.sort(key=lambda p: costs[p.name + ":" + p.field])
            return usable[: cfg.max_predicates]
        return chosen

    def _candidate_pairs(
        self,
        records: Mapping[str, Record],
        predicates: Sequence[Predicate],
        cfg: ResolutionConfig,
    ) -> list[CandidatePair]:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for pred in predicates:
            for pair in self._pairs_of(self._blocks(records, pred, cfg)):
                counts[pair] += 1
        return [
            CandidatePair(left=left_id, right=right_id, shared_keys=c)
            for (left_id, right_id), c in sorted(counts.items())
        ]

    # --- comparison ------------------------------------------------------

    def _level_for(self, comparison: FieldComparison, left: Record, right: Record) -> str:
        a = left.get(comparison.field)
        b = right.get(comparison.field)
        if a is None or b is None:
            return NO_MATCH_LEVEL
        compare = comparison.comparator or affine_gap_similarity
        sim = compare(str(a), str(b))
        for level in comparison.levels:
            if sim >= level.threshold:
                return level.label
        return NO_MATCH_LEVEL

    def _pattern(
        self, left: Record, right: Record, cfg: ResolutionConfig
    ) -> dict[str, str]:
        return {c.field: self._level_for(c, left, right) for c in cfg.comparisons}

    def _all_levels(self, cfg: ResolutionConfig) -> dict[str, list[str]]:
        return {
            c.field: [lv.label for lv in c.levels] + [NO_MATCH_LEVEL]
            for c in cfg.comparisons
        }

    # --- Fellegi-Sunter EM ----------------------------------------------

    def _train_em(
        self, patterns: Sequence[Mapping[str, str]], cfg: ResolutionConfig
    ) -> TrainedModel:
        """Estimate m and u probabilities with no labelled data.

        The generative story: each pair is a match with prior probability lambda;
        given match status, each field's comparison level is drawn independently
        from m (if match) or u (if not). EM alternates between assigning each
        observed pattern a posterior match probability and re-estimating the
        parameters as posterior-weighted level frequencies.

        Conditional independence between fields is the model's known weakness
        (first name and gender are correlated, for instance). It is also what
        makes the parameters estimable without labels, and the resulting weights
        interpretable.
        """
        levels = self._all_levels(cfg)
        if not patterns:
            uniform = {
                f: {lv: 1.0 / len(lvs) for lv in lvs} for f, lvs in levels.items()
            }
            return TrainedModel(0.5, uniform, dict(uniform), 0, True)

        # Initialise pessimistically but not degenerately: a match is assumed to
        # agree strongly, a non-match to agree only by chance.
        m_prob = {
            f: self._seed(lvs, top_mass=0.85) for f, lvs in levels.items()
        }
        u_prob = {
            f: self._seed(list(reversed(lvs)), top_mass=0.85) for f, lvs in levels.items()
        }
        lam = 0.1
        converged = False
        iterations = 0

        # The loop variable is read after the loop to report how many iterations
        # EM actually needed, so it is intentionally not used in the body.
        for iterations in range(1, cfg.em_iterations + 1):  # noqa: B007
            responsibilities = []
            for pattern in patterns:
                pm = lam
                pu = 1.0 - lam
                for f, level in pattern.items():
                    pm *= m_prob[f].get(level, _EPS)
                    pu *= u_prob[f].get(level, _EPS)
                total = pm + pu
                responsibilities.append(0.5 if total <= 0 else pm / total)

            new_lam = sum(responsibilities) / len(responsibilities)
            new_lam = min(max(new_lam, 1e-6), 1 - 1e-6)

            new_m: dict[str, dict[str, float]] = {}
            new_u: dict[str, dict[str, float]] = {}
            for f, lvs in levels.items():
                # Laplace smoothing keeps an unobserved level from collapsing a
                # whole pattern's likelihood to zero.
                mc = {lv: _EPS for lv in lvs}
                uc = {lv: _EPS for lv in lvs}
                for pattern, g in zip(patterns, responsibilities):
                    lv = pattern.get(f, NO_MATCH_LEVEL)
                    mc[lv] = mc.get(lv, 0.0) + g
                    uc[lv] = uc.get(lv, 0.0) + (1.0 - g)
                m_total = sum(mc.values())
                u_total = sum(uc.values())
                new_m[f] = {lv: c / m_total for lv, c in mc.items()}
                new_u[f] = {lv: c / u_total for lv, c in uc.items()}

            delta = abs(new_lam - lam) + sum(
                abs(new_m[f][lv] - m_prob[f][lv]) + abs(new_u[f][lv] - u_prob[f][lv])
                for f, lvs in levels.items()
                for lv in lvs
            )
            lam, m_prob, u_prob = new_lam, new_m, new_u
            if delta < cfg.em_tolerance:
                converged = True
                break

        return TrainedModel(lam, m_prob, u_prob, iterations, converged)

    def _seed(self, levels: Sequence[str], top_mass: float) -> dict[str, float]:
        """Put most of the probability mass on the first level, spread the rest."""
        if len(levels) == 1:
            return {levels[0]: 1.0}
        rest = (1.0 - top_mass) / (len(levels) - 1)
        return {lv: (top_mass if i == 0 else rest) for i, lv in enumerate(levels)}

    def _score(
        self,
        left: str,
        right: str,
        pattern: Mapping[str, str],
        model: TrainedModel,
    ) -> ScoredPair:
        """Convert a comparison pattern into a match weight and a probability.

        match_weight = log2(lambda / (1 - lambda)) + sum over fields of
        log2(m / u). Each term is that field's evidence in bits, which is what
        makes the score auditable: you can point at the field that decided it.
        """
        lam = min(max(model.lambda_prior, 1e-9), 1 - 1e-9)
        weight = math.log2(lam / (1.0 - lam))
        for f, level in pattern.items():
            m = max(model.m_probabilities.get(f, {}).get(level, _EPS), _EPS)
            u = max(model.u_probabilities.get(f, {}).get(level, _EPS), _EPS)
            weight += math.log2(m / u)
        # Guard the exponent so an overwhelming weight saturates instead of
        # overflowing.
        clamped = max(-500.0, min(500.0, weight))
        odds = 2.0**clamped
        probability = odds / (1.0 + odds)
        return ScoredPair(
            left=left,
            right=right,
            match_probability=probability,
            match_weight=weight,
            pattern=dict(pattern),
        )

    # --- clustering ------------------------------------------------------

    def _cluster(
        self,
        record_ids: Iterable[str],
        scored: Sequence[ScoredPair],
        cfg: ResolutionConfig,
    ) -> list[EntityCluster]:
        """Turn pairwise scores into disjoint entities.

        Connected components are the naive choice and they chain: A~B and B~C
        merges A with C even when A and C clearly differ. Average linkage
        resists that by requiring the mean score *between* two groups to clear
        the threshold before they merge, which is the behaviour you want for
        customer or patient records.
        """
        ids = sorted(set(record_ids))
        above = [s for s in scored if s.match_probability >= cfg.match_threshold]
        scores = {
            (s.left, s.right): s.match_probability for s in above
        }
        scores.update({(s.right, s.left): s.match_probability for s in above})

        if cfg.cluster_link == "connected":
            groups = self._connected_components(ids, above)
        elif cfg.cluster_link == "average":
            groups = self._average_linkage(ids, above, cfg.match_threshold)
        else:
            raise ValueError("cluster_link must be 'average' or 'connected'")

        clusters: list[EntityCluster] = []
        for index, members in enumerate(groups):
            member_list = sorted(members)
            within = [
                scores[(a, b)]
                for i, a in enumerate(member_list)
                for b in member_list[i + 1 :]
                if (a, b) in scores
            ]
            clusters.append(
                EntityCluster(
                    cluster_id=index,
                    record_ids=member_list,
                    cohesion=(sum(within) / len(within)) if within else 1.0,
                )
            )
        clusters.sort(key=lambda c: (-len(c.record_ids), c.record_ids[0]))
        return [
            EntityCluster(i, c.record_ids, c.cohesion) for i, c in enumerate(clusters)
        ]

    def _connected_components(
        self, ids: Sequence[str], edges: Sequence[ScoredPair]
    ) -> list[set[str]]:
        parent = {i: i for i in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for edge in edges:
            a, b = find(edge.left), find(edge.right)
            if a != b:
                parent[b] = a

        groups: dict[str, set[str]] = defaultdict(set)
        for i in ids:
            groups[find(i)].add(i)
        return list(groups.values())

    def _average_linkage(
        self, ids: Sequence[str], edges: Sequence[ScoredPair], threshold: float
    ) -> list[set[str]]:
        pair_scores: dict[tuple[str, str], float] = {}
        for e in edges:
            key = (e.left, e.right) if e.left < e.right else (e.right, e.left)
            pair_scores[key] = max(pair_scores.get(key, 0.0), e.match_probability)

        clusters: list[set[str]] = [{i} for i in ids]

        def between(a: set[str], b: set[str]) -> float:
            """Mean score over all cross-cluster pairs. Absent pairs count as 0,
            which is what stops a single strong link from dragging groups
            together."""
            total = 0.0
            for x in a:
                for y in b:
                    key = (x, y) if x < y else (y, x)
                    total += pair_scores.get(key, 0.0)
            return total / (len(a) * len(b))

        merged = True
        while merged:
            merged = False
            best = None
            best_score = threshold
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    score = between(clusters[i], clusters[j])
                    if score >= best_score:
                        best = (i, j)
                        best_score = score
            if best is not None:
                i, j = best
                clusters[i] |= clusters[j]
                clusters.pop(j)
                merged = True
        return clusters


def default_model(
    config: ResolutionConfig, match_rate: float = 0.01
) -> TrainedModel:
    """Fellegi-Sunter weights with nothing trained, for the first run.

    A new deployment has no corpus to learn from, and waiting for one means
    shipping nothing. These are the same seed parameters the EM step starts
    from - most of the m mass on the strongest level, most of the u mass on
    `NO_MATCH_LEVEL` - which is to say: a match usually agrees on a field, a
    non-match usually does not. That is weak but not arbitrary, and it is
    directionally right on every field it is given.

    `iterations=0` and `converged=False` are the honest record that these were
    asserted rather than measured. Retrain with `execute` as soon as there is a
    batch to retrain on.
    """
    if not config.comparisons:
        raise ValueError("config.comparisons must not be empty")
    if not 0.0 < match_rate < 1.0:
        raise ValueError("match_rate must be strictly between 0 and 1")

    component = EntityResolutionComponent()
    levels = component._all_levels(config)
    return TrainedModel(
        lambda_prior=match_rate,
        m_probabilities={
            field: component._seed(field_levels, _DEFAULT_M_TOP)
            for field, field_levels in levels.items()
        },
        u_probabilities={
            # Reversed: for a non-match the mass belongs on NO_MATCH_LEVEL,
            # which `_all_levels` puts last.
            field: component._seed(list(reversed(field_levels)), _DEFAULT_U_TOP)
            for field, field_levels in levels.items()
        },
        iterations=0,
        converged=False,
    )


__all__ = [
    "EntityResolutionComponent",
    "affine_gap_distance",
    "default_model",
    "affine_gap_similarity",
    "default_predicates",
    "NO_MATCH_LEVEL",
]
