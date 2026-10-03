"""Structure-aware, provenance-preserving chunking.

This is the component the roadmap identified as genuinely missing from the
ecosystem. Chonkie, LangChain and LlamaIndex all chunk *text*: a string goes in,
strings come out, and any knowledge of which page or which region the text came
from is gone. Nothing chunks *blocks that carry geometry* and propagates that
geometry into every chunk.

That propagation is the whole point. A chunk that knows it came from page 7 at a
specific bounding box can be cited, highlighted and audited. A chunk that is just
a string can only be trusted.

Three decisions here that matter more than the token arithmetic:

* **Sections beat budgets.** A heading is the author telling you where a topic
  starts. Packing across it to fill a 512-token quota discards that for nothing.
* **Heading breadcrumbs are prepended.** A chunk reading "must not exceed 40V"
  is useless in isolation; "Manual > Safety > Voltage" attached to it is not.
* **Overlap is in whole sentences.** Overlapping by raw token count severs
  sentences, and a half-sentence at a chunk boundary embeds to noise.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from ..core.models import Block, BlockType, Chunk, Document, Provenance
from .models import ChunkConfig, ChunkRequest, ChunkResult

# Sentence boundary: terminal punctuation, optional closing quote/bracket, then
# whitespace followed by something that plausibly starts a sentence. The
# abbreviation guard is a blocklist rather than a model; it covers the cases that
# actually appear in documents and accepts the rest as boundaries.
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=[A-Z0-9"\'(\[])')
_ABBREVIATION = re.compile(
    r"\b(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|Inc|Ltd|Corp|No|Fig|Eq|Ch|Sec|Art|approx|cf|al)\.$",
    re.IGNORECASE,
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF      # CJK unified ideographs
        or 0x3400 <= code <= 0x4DBF   # extension A
        or 0x3040 <= code <= 0x30FF   # hiragana + katakana
        or 0xAC00 <= code <= 0xD7AF   # hangul syllables
        or 0xF900 <= code <= 0xFAFF   # compatibility ideographs
    )


def estimate_tokens(text: str) -> int:
    """Rough token count with no tokenizer.

    Two crude signals for space-separated text — characters over four, and words
    times 1.3 — and the larger wins, because each fails in a different
    direction: the character rule underestimates text with many short words, the
    word rule underestimates long technical terms that split into several
    tokens. Overestimating slightly is the safe error for a budget.

    **CJK is counted separately, at roughly one token per character.** Both
    heuristics fail catastrophically otherwise: Chinese and Japanese have no
    spaces, so the word arm collapses to 1, and the chars/4 arm underestimates
    by about 4x. Measured on the stress corpus, a 478-token Chinese passage was
    estimated at 133 — so every chunk silently overran the embedding window and
    the budget was meaningless for any non-Latin document.
    """
    stripped = text.strip()
    if not stripped:
        return 0

    cjk = sum(1 for char in stripped if _is_cjk(char))
    if not cjk:
        return max(1, int(_latin_arms(stripped)))

    # Mixed text: score each script with the rule that suits it, then add.
    rest = "".join(char for char in stripped if not _is_cjk(char))
    rest_tokens = _latin_arms(rest) if rest.strip() else 0.0
    return max(1, int(cjk + rest_tokens))


def _latin_arms(text: str) -> float:
    """The three heuristics for space-separated text; the largest wins.

    The third arm exists because the first two model prose and nothing else. A
    BPE vocabulary packs roughly four letters into a token, but gives most
    punctuation marks and most digits a token each, so a table of numbers costs
    far more than its character count suggests. Measured on 2,780 chunks of real
    papers, the worst case was a results table of 1,534 characters that the old
    estimate put at 383 tokens and `cl100k_base` actually tokenised at **1,005**
    - a chunk that would silently overflow a 512-token budget by 2x.

    Counting punctuation and digits at one token each is the model, not a fit:
    the coefficients are 1.0 rather than tuned decimals so this does not quietly
    specialise to the corpus it was measured on.

    Effect on those 2,780 chunks, against `cl100k_base` with a 512-token budget:
    chunks the estimate called in-budget while they really were not fell from
    **959 to 231**, a 76% reduction. The cost is a median real/estimated ratio of
    0.97 instead of 1.03, so chunks now run about 3% under budget rather than 3%
    over - which is the direction the budget wants to err in, and what this
    function's docstring already claimed it did.
    """
    words = len(text.split())
    symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
    digits = sum(1 for char in text if char.isdigit())
    return max(
        len(text) / 4.0,
        words * 1.3,
        words * 0.9 + symbols + digits,
    )


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping abbreviations intact.

    A candidate boundary is rejected when the text before it ends in a known
    abbreviation, so 'approx. 40V' and 'Dr. Chen' stay whole.
    """
    text = " ".join(text.split())
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        candidate = text[start : match.start()]
        if _ABBREVIATION.search(candidate.strip()):
            continue
        pieces.append(candidate.strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces or [text]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


class _Unit:
    """One sentence, with the provenance and heading context it inherited."""

    __slots__ = ("text", "provenance", "heading_path", "starts_section", "block_type", "tokens")

    def __init__(
        self,
        text: str,
        provenance: Provenance | None,
        heading_path: tuple[str, ...],
        starts_section: bool,
        block_type: BlockType,
        tokens: int,
    ) -> None:
        self.text = text
        self.provenance = provenance
        self.heading_path = heading_path
        self.starts_section = starts_section
        self.block_type = block_type
        self.tokens = tokens


class ChunkerComponent:
    """Document in, retrievable chunks out, with provenance intact.

    Optionally takes an `Embedder` (the toolkit port) to enable semantic boundary
    detection. Without one it is pure text and geometry arithmetic, with no
    dependency and no model.
    """

    def __init__(self, embedder: object | None = None) -> None:
        self._embedder = embedder

    def execute(self, input_data: ChunkRequest) -> ChunkResult:
        cfg = input_data.config
        document = input_data.document
        count = cfg.token_counter or estimate_tokens

        units = self._units(document, cfg, count)
        if not units:
            return ChunkResult(chunks=[], oversized=[], token_estimates={})

        boundaries = self._semantic_boundaries(units, cfg)
        groups = self._pack(units, cfg, count, boundaries)
        groups = self._merge_runts(groups, cfg, count)
        groups = self._enforce_budget(groups, cfg, count)

        chunks: list[Chunk] = []
        oversized: list[str] = []
        estimates: dict[str, int] = {}

        for index, group in enumerate(groups):
            text = self._render(group, cfg)
            chunk_id = document.doc_id + "#" + str(index)
            tokens = count(text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    doc_id=document.doc_id,
                    index=index,
                    provenances=self._provenances(group),
                    metadata={
                        "heading_path": list(group[0].heading_path),
                        "block_types": sorted({u.block_type.value for u in group}),
                        "token_estimate": tokens,
                        # The content hash is metadata, not part of chunk_id, so
                        # re-ingesting an edited document overwrites slot N
                        # instead of leaving an orphan row behind.
                        "content_hash": hashlib.blake2b(
                            text.encode("utf-8"), digest_size=8
                        ).hexdigest(),
                        "source_uri": document.source_uri,
                    },
                )
            )
            estimates[chunk_id] = tokens
            if tokens > cfg.max_tokens:
                oversized.append(chunk_id)

        return ChunkResult(chunks=chunks, oversized=oversized, token_estimates=estimates)

    # --- unit construction ------------------------------------------------

    def _units(
        self, document: Document, cfg: ChunkConfig, count: object
    ) -> list[_Unit]:
        """Flatten blocks into sentences, tracking the heading stack.

        The heading stack is maintained by level, so a level-2 heading after a
        level-3 one correctly pops the deeper entry. Documents with erratic
        levels still behave, because the stack is truncated by level rather than
        assumed to nest perfectly.
        """
        blocks = document.blocks if cfg.keep_furniture else document.content_blocks()
        stack: list[tuple[int, str]] = []
        units: list[_Unit] = []

        for block in blocks:
            if block.type is BlockType.HEADING:
                level = block.level or 1
                stack = [entry for entry in stack if entry[0] < level]
                stack.append((level, block.text))
                # The heading itself becomes a unit so it is never dropped, and
                # it marks the start of a section.
                units.append(
                    _Unit(
                        text=block.text,
                        provenance=block.provenance,
                        heading_path=tuple(name for _, name in stack),
                        starts_section=True,
                        block_type=block.type,
                        tokens=count(block.text),  # type: ignore[operator]
                    )
                )
                continue

            path = tuple(name for _, name in stack)
            for sentence in self._block_sentences(block):
                for piece in self._fit(sentence, cfg, count):
                    units.append(
                        _Unit(
                            text=piece,
                            provenance=block.provenance,
                            heading_path=path,
                            starts_section=False,
                            block_type=block.type,
                            tokens=count(piece),  # type: ignore[operator]
                        )
                    )
        return units

    def _fit(self, text: str, cfg: ChunkConfig, count: object) -> list[str]:
        """Hard-split a unit that alone exceeds the budget, on word boundaries.

        Sentence splitting cannot help with a 3,000-token table row or a wall of
        text with no punctuation, and letting such a unit through would produce a
        chunk that an embedding endpoint rejects outright. Splitting on words
        keeps every character while guaranteeing the budget.
        """
        if count(text) <= cfg.max_tokens:  # type: ignore[operator]
            return [text]
        words = text.split()
        if len(words) <= 1:
            return [text]
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = current + [word]
            if current and count(" ".join(candidate)) > cfg.max_tokens:  # type: ignore[operator]
                pieces.append(" ".join(current))
                current = [word]
            else:
                current = candidate
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _block_sentences(self, block: Block) -> list[str]:
        """A list item or a table row is one unit even if it contains two
        sentences; splitting it would strip the structure that made it meaningful."""
        if block.type in (BlockType.LIST_ITEM, BlockType.TABLE, BlockType.CODE):
            text = block.text.strip()
            return [text] if text else []
        return split_sentences(block.text)

    # --- semantic boundaries ----------------------------------------------

    def _semantic_boundaries(self, units: Sequence[_Unit], cfg: ChunkConfig) -> set[int]:
        """Indices where a forced break is required by semantic dissimilarity.

        Only consulted when both a threshold and an embedder are present. Returns
        an empty set otherwise, so the caller needs no special case.
        """
        if cfg.semantic_threshold is None or self._embedder is None or len(units) < 2:
            return set()
        vectors = self._embedder.embed([u.text for u in units])  # type: ignore[attr-defined]
        if len(vectors) != len(units):
            return set()
        return {
            index
            for index in range(1, len(units))
            if _cosine(vectors[index - 1], vectors[index]) < cfg.semantic_threshold
        }

    # --- packing ----------------------------------------------------------

    def _pack(
        self,
        units: Sequence[_Unit],
        cfg: ChunkConfig,
        count: object,
        boundaries: set[int],
    ) -> list[list[_Unit]]:
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        current_tokens = 0
        heading_budget = 0

        for index, unit in enumerate(units):
            forced = index in boundaries or (
                cfg.split_on_heading and unit.starts_section and bool(current)
            )
            would_exceed = current and (
                current_tokens + unit.tokens + heading_budget > cfg.max_tokens
            )

            if forced or would_exceed:
                groups.append(current)
                carry = [] if forced else self._overlap(current, cfg, count)
                current = list(carry)
                current_tokens = sum(u.tokens for u in current)

            if not current:
                # The heading prefix consumes part of the budget, so account for
                # it once per chunk rather than discovering the overrun later.
                heading_budget = (
                    count(cfg.heading_separator.join(unit.heading_path))  # type: ignore[operator]
                    if cfg.include_heading_path and unit.heading_path
                    else 0
                )
            current.append(unit)
            current_tokens += unit.tokens

        if current:
            groups.append(current)
        return [g for g in groups if g]

    def _overlap(
        self, group: Sequence[_Unit], cfg: ChunkConfig, count: object
    ) -> list[_Unit]:
        """Take whole sentences from the tail of `group` up to the overlap budget.

        Never carries the heading unit itself: repeating a heading as overlap adds
        no information, because the breadcrumb already carries it.
        """
        if cfg.overlap <= 0:
            return []
        carried: list[_Unit] = []
        total = 0
        for unit in reversed(group):
            if unit.starts_section:
                continue
            if total + unit.tokens > cfg.overlap:
                break
            carried.insert(0, unit)
            total += unit.tokens
        return carried

    def _merge_runts(
        self, groups: Sequence[list[_Unit]], cfg: ChunkConfig, count: object
    ) -> list[list[_Unit]]:
        """Fold an undersized chunk into its predecessor when that is safe.

        Refuses to merge across a section boundary, since the reason the chunk is
        small is usually that its section is short — and merging two different
        sections to hit a token target is exactly the mistake `split_on_heading`
        exists to avoid.
        """
        out: list[list[_Unit]] = []
        for group in groups:
            tokens = sum(u.tokens for u in group)
            if (
                out
                and tokens < cfg.minimum
                and not group[0].starts_section
                and group[0].heading_path == out[-1][0].heading_path
                and sum(u.tokens for u in out[-1]) + tokens <= cfg.max_tokens
            ):
                out[-1].extend(group)
                continue
            out.append(list(group))
        return out

    def _enforce_budget(
        self, groups: Sequence[list[_Unit]], cfg: ChunkConfig, count: object
    ) -> list[list[_Unit]]:
        """Re-split any group whose *rendered* text exceeds the budget.

        Packing sums per-unit token counts, but the chunk that ships is the
        rendered string: joining separators, list bullets and the heading
        breadcrumb all add tokens that the per-unit sum never saw. With the
        default heuristic counter the two also disagree because `max(chars/4,
        words*1.3)` can switch which term dominates once units are concatenated.

        Rather than trying to predict the discrepancy — impossible in general,
        since `token_counter` is caller-supplied — this measures the real thing
        and moves trailing units into a follow-on chunk until the budget holds.
        A group of one unit is left alone: it was already word-split as far as it
        can be, and reporting it as oversized is more honest than losing text.
        """
        out: list[list[_Unit]] = []
        worklist = [list(group) for group in groups]
        while worklist:
            group = worklist.pop(0)
            if len(group) <= 1 or count(self._render(group, cfg)) <= cfg.max_tokens:  # type: ignore[operator]
                out.append(group)
                continue
            overflow: list[_Unit] = []
            while len(group) > 1 and count(self._render(group, cfg)) > cfg.max_tokens:  # type: ignore[operator]
                overflow.insert(0, group.pop())
            out.append(group)
            if overflow:
                worklist.insert(0, overflow)
        return out

    # --- rendering --------------------------------------------------------

    def _render(self, group: Sequence[_Unit], cfg: ChunkConfig) -> str:
        has_list = any(u.block_type is BlockType.LIST_ITEM for u in group)
        parts = [
            ("- " + u.text) if u.block_type is BlockType.LIST_ITEM else u.text
            for u in group
            if u.text
        ]
        # Lists keep one item per line; prose reads as a paragraph.
        body = "\n".join(parts) if has_list else " ".join(parts)

        if cfg.include_heading_path and group[0].heading_path:
            breadcrumb = cfg.heading_separator.join(group[0].heading_path)
            return breadcrumb + cfg.heading_prefix_separator + body
        return body

    def _provenances(self, group: Sequence[_Unit]) -> list[Provenance]:
        """One merged provenance per page the chunk drew from.

        Merging per page rather than producing one box for the whole chunk is the
        honest representation: a chunk spanning pages 7 and 8 has two regions,
        not one impossible box straddling the break.
        """
        by_page: dict[int, Provenance] = {}
        for unit in group:
            prov = unit.provenance
            if prov is None:
                continue
            existing = by_page.get(prov.page)
            by_page[prov.page] = prov if existing is None else existing.merge(prov)
        return [by_page[page] for page in sorted(by_page)]


__all__ = ["ChunkerComponent", "estimate_tokens", "split_sentences"]
