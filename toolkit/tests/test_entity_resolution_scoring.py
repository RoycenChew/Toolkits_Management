"""TK-6: scoring one record against candidates, and two new comparators.

`execute` resolves a batch: it learns blocking predicates, generates candidate
pairs, and estimates m and u probabilities by EM over those pairs. That is the
right shape for deduplicating a corpus and the wrong shape for the question an
application actually asks at runtime — "here is one new record, which of these
existing ones is it?" — because EM over a batch of one has nothing to learn
from and would return parameters that say whatever the seed said.

So the model is separated from the scoring. Train once with `execute`, keep the
`TrainedModel`, and score single records against candidates with it for as long
as it holds. The weight arithmetic is identical, which the first test here
asserts directly rather than trusting.

The two comparators exist because the default is affine-gap string distance,
and for a number or a date that is simply the wrong question: `1000` and `9999`
share three characters of length and nothing else, and `2026-01-31` and
`2026-02-01` are one day apart and look nothing alike.
"""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit.entity_resolution import (  # noqa: E402
    NO_MATCH_LEVEL,
    ComparisonLevel,
    EntityResolutionComponent,
    FieldComparison,
    ResolutionConfig,
    ResolutionRequest,
    ScoredPair,
    date_comparator,
    date_similarity,
    days_apart,
    default_model,
    numeric_comparator,
    numeric_similarity,
)

_LEFT = {
    "a1": {"name": "Robert Smith", "city": "London", "ref": "A-100"},
    "a2": {"name": "Alice Nakamura", "city": "Osaka", "ref": "A-205"},
    "a3": {"name": "Wei Chen", "city": "Taipei", "ref": "A-310"},
    "a4": {"name": "Priya Raman", "city": "Chennai", "ref": "A-412"},
    "a5": {"name": "Jonas Weber", "city": "Berlin", "ref": "A-515"},
    "a6": {"name": "Maria Santos", "city": "Lisbon", "ref": "A-620"},
    "b1": {"name": "Robert J Smith", "city": "London", "ref": "A-100"},
    "b2": {"name": "Alice Nakamura", "city": "Osaka", "ref": "A-205"},
    "b3": {"name": "Wei Chen", "city": "Taipei", "ref": "A-310"},
    "b4": {"name": "Priya Raman", "city": "Chennai", "ref": "A-412"},
    "b5": {"name": "Jonas Weber", "city": "Berlin", "ref": "A-515"},
    "b6": {"name": "Maria Santos", "city": "Lisbon", "ref": "A-620"},
}


def _config() -> ResolutionConfig:
    return ResolutionConfig(
        comparisons=[
            FieldComparison("name"),
            FieldComparison("city"),
            FieldComparison("ref"),
        ]
    )


def _trained():
    component = EntityResolutionComponent()
    config = _config()
    result = component.execute(ResolutionRequest(records=_LEFT, config=config))
    return component, config, result


# --------------------------------------------------------------------------
# Numeric comparator
# --------------------------------------------------------------------------


def test_numeric_similarity_is_relative_difference() -> None:
    """Relative, not absolute: a 10 unit difference is nothing on a million and
    everything on a dozen, and one threshold cannot serve both if the scale is
    absolute."""
    assert numeric_similarity("100", "100") == 1.0
    assert numeric_similarity("100", "110") == pytest.approx(1 - 10 / 110)
    assert numeric_similarity("1000000", "1000010") > 0.999
    assert numeric_similarity("10", "20") == pytest.approx(0.5)
    assert numeric_similarity("0", "0") == 1.0


def test_numeric_similarity_is_symmetric_and_handles_signs() -> None:
    assert numeric_similarity("110", "100") == numeric_similarity("100", "110")
    # Opposite signs are not nearly equal, however close the magnitudes.
    assert numeric_similarity("100", "-100") == 0.0
    assert numeric_similarity("-100", "-100") == 1.0
    assert numeric_similarity("0", "50") == 0.0


def test_numeric_similarity_reads_the_formats_records_actually_hold() -> None:
    assert numeric_similarity("1,240.50", "1240.5") == 1.0
    assert numeric_similarity("$1,240.50", "1240.50") == 1.0
    assert numeric_similarity(" 42 ", "42") == 1.0


def test_a_value_that_is_not_a_number_is_no_evidence_either_way() -> None:
    """0.0, not an exception: a record with a missing or junk field is the
    normal case in the data this exists for, and a comparator that raises takes
    the whole batch down with it."""
    assert numeric_similarity("abc", "42") == 0.0
    assert numeric_similarity("", "42") == 0.0
    assert numeric_similarity("abc", "def") == 0.0


def test_an_absolute_scale_is_available_when_the_field_has_a_natural_one() -> None:
    """A relative difference is wrong for a quantity that legitimately passes
    through zero, so the scale can be stated instead."""
    close = numeric_comparator(scale=10.0)
    assert close("100", "100") == 1.0
    assert close("100", "105") == pytest.approx(0.5)
    assert close("100", "110") == 0.0
    assert close("0", "1") == pytest.approx(0.9)


def test_the_numeric_comparator_plugs_into_a_field_comparison() -> None:
    comparison = FieldComparison(
        "amount",
        levels=(ComparisonLevel("exact", 1.0), ComparisonLevel("close", 0.95)),
        comparator=numeric_similarity,
    )
    component = EntityResolutionComponent()
    level = component._level_for(comparison, {"amount": "1240.50"}, {"amount": "1,240.50"})
    assert level == "exact"
    # Affine-gap would call these similar; they are 20% apart.
    far = component._level_for(comparison, {"amount": "1000"}, {"amount": "1200"})
    assert far == NO_MATCH_LEVEL


# --------------------------------------------------------------------------
# Date comparator
# --------------------------------------------------------------------------


def test_days_apart_counts_days_not_characters() -> None:
    assert days_apart("2026-01-31", "2026-02-01") == 1
    assert days_apart("2026-02-01", "2026-01-31") == 1
    assert days_apart("2026-01-01", "2026-01-01") == 0
    assert days_apart("2025-01-01", "2026-01-01") == 365
    assert days_apart("not a date", "2026-01-01") is None


def test_days_apart_reads_the_written_forms_too() -> None:
    assert days_apart("14 March 2024", "2024-03-14") == 0
    assert days_apart("March 14, 2024", "2024-03-15") == 1
    assert days_apart("2024/03/14", "2024-03-14") == 0


def test_date_similarity_decays_over_a_window() -> None:
    assert date_similarity("2026-01-01", "2026-01-01") == 1.0
    assert date_similarity("2026-01-01", "2026-01-02") > 0.99
    # The default window is a year, so half a year apart is about half.
    assert date_similarity("2026-01-01", "2026-07-02") == pytest.approx(0.5, abs=0.01)
    assert date_similarity("2020-01-01", "2026-01-01") == 0.0
    assert date_similarity("not a date", "2026-01-01") == 0.0


def test_the_date_window_is_configurable_because_fields_differ() -> None:
    """A date of birth that is three days out is a transcription error; an
    invoice date three days out is a different invoice."""
    tight = date_comparator(window_days=7)
    assert tight("2026-01-01", "2026-01-04") == pytest.approx(1 - 3 / 7)
    assert tight("2026-01-01", "2026-02-01") == 0.0
    assert date_comparator(window_days=3650)("2020-01-01", "2026-01-01") > 0.3


def test_a_zero_window_is_refused_rather_than_dividing_by_zero() -> None:
    with pytest.raises(ValueError):
        date_comparator(window_days=0)
    with pytest.raises(ValueError):
        numeric_comparator(scale=-1.0)


# --------------------------------------------------------------------------
# Single-record scoring
# --------------------------------------------------------------------------


def test_scoring_one_record_reproduces_the_batch_weight_exactly() -> None:
    """The claim worth testing: separating the model from the scoring changes
    nothing about the arithmetic. If it did, a cached model would quietly mean
    something different from the run that produced it."""
    component, config, result = _trained()
    batch = {(p.left, p.right): p for p in result.scored_pairs}
    assert ("a1", "b1") in batch, "the batch must have scored this pair"

    scored = component.score_record(
        _LEFT["a1"],
        {"b1": _LEFT["b1"]},
        config,
        result.model,
        record_id="a1",
    )

    [one] = scored
    expected = batch[("a1", "b1")]
    assert one.match_weight == pytest.approx(expected.match_weight)
    assert one.match_probability == pytest.approx(expected.match_probability)
    assert one.pattern == expected.pattern


def test_scoring_one_record_does_not_retrain() -> None:
    """No EM, by construction: the model that goes in is the model that is
    used. A batch of one candidate has nothing to estimate from, and EM over it
    would return whatever the seed said while looking like a measurement."""
    component, config, result = _trained()
    before = (
        result.model.iterations,
        dict(result.model.m_probabilities),
        result.model.lambda_prior,
    )
    component.score_record(_LEFT["a1"], _LEFT, config, result.model)
    after = (
        result.model.iterations,
        dict(result.model.m_probabilities),
        result.model.lambda_prior,
    )
    assert before == after


def test_the_best_candidate_comes_first_and_is_the_right_one() -> None:
    component, config, result = _trained()
    incoming = {"name": "Robert Smith", "city": "London", "ref": "A-100"}

    scored = component.score_record(incoming, _LEFT, config, result.model)

    assert [s.right for s in scored[:2]] == sorted(["a1", "b1"])[:2] or scored[0].right in (
        "a1",
        "b1",
    )
    assert scored[0].match_probability >= scored[-1].match_probability
    assert scored[0].right in ("a1", "b1")
    assert scored[0].match_probability > 0.9
    # Everything that is not Robert Smith scores far lower.
    others = [s for s in scored if s.right not in ("a1", "b1")]
    assert all(s.match_probability < 0.5 for s in others)


def test_every_result_carries_its_per_field_pattern() -> None:
    """The pattern is the explanation. A score with no pattern is a number
    nobody can argue with, which in a matching system is a liability."""
    component, config, result = _trained()
    scored = component.score_record(_LEFT["a1"], _LEFT, config, result.model, "a1")

    for pair in scored:
        assert isinstance(pair, ScoredPair)
        assert set(pair.pattern) == {"name", "city", "ref"}
    best = next(s for s in scored if s.right == "b1")
    assert best.pattern["city"] == "exact"
    assert best.pattern["ref"] == "exact"


def test_the_record_is_not_scored_against_itself() -> None:
    """Passing the whole corpus as candidates is the obvious call site, and a
    record matching itself at probability 1.0 would sit at the top of every
    result and mean nothing."""
    component, config, result = _trained()
    scored = component.score_record(_LEFT["a1"], _LEFT, config, result.model, "a1")
    assert "a1" not in [s.right for s in scored]
    assert len(scored) == len(_LEFT) - 1


def test_no_candidates_is_an_empty_list_not_an_error() -> None:
    component, config, result = _trained()
    assert component.score_record(_LEFT["a1"], {}, config, result.model) == []


# --------------------------------------------------------------------------
# Fixed weights, with nothing trained
# --------------------------------------------------------------------------


def test_a_default_model_scores_without_any_training_data() -> None:
    """The first run of a new deployment has no corpus to learn from, and
    waiting for one means shipping nothing. These are the same seed parameters
    EM starts from, and they are honest about it: iterations=0."""
    config = _config()
    model = default_model(config)
    assert model.iterations == 0
    assert model.converged is False

    component = EntityResolutionComponent()
    scored = component.score_record(
        {"name": "Robert Smith", "city": "London", "ref": "A-100"},
        _LEFT,
        config,
        model,
    )
    best = scored[0]
    assert best.right in ("a1", "b1")
    assert best.match_weight > 0
    assert scored[-1].match_weight < 0


def test_the_match_rate_moves_the_prior_the_way_it_should() -> None:
    config = _config()
    rare = default_model(config, match_rate=0.001)
    common = default_model(config, match_rate=0.2)
    assert rare.lambda_prior < common.lambda_prior

    component = EntityResolutionComponent()
    record = {"name": "Priya Raman", "city": "Chennai", "ref": "A-412"}
    rare_best = component.score_record(record, _LEFT, config, rare)[0]
    common_best = component.score_record(record, _LEFT, config, common)[0]
    # A rarer prior needs more evidence for the same conclusion.
    assert rare_best.match_weight < common_best.match_weight


def test_an_impossible_match_rate_is_refused() -> None:
    config = _config()
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            default_model(config, match_rate=bad)


def test_a_model_that_does_not_know_a_field_says_so() -> None:
    """Silently scoring an unknown field as m/u = 1 would make it contribute
    exactly nothing, so adding a comparison to the config and forgetting to
    retrain would look like it worked."""
    component, config, result = _trained()
    widened = ResolutionConfig(
        comparisons=[*config.comparisons, FieldComparison("postcode")]
    )
    with pytest.raises(ValueError) as caught:
        component.score_record(_LEFT["a1"], _LEFT, widened, result.model)
    assert "postcode" in str(caught.value)


def test_scoring_needs_comparisons_and_a_model() -> None:
    component, config, result = _trained()
    with pytest.raises(ValueError):
        component.score_record(
            _LEFT["a1"], _LEFT, ResolutionConfig(comparisons=[]), result.model
        )
    with pytest.raises(ValueError):
        component.score_record(_LEFT["a1"], _LEFT, config, None)


def test_a_trained_model_round_trips_through_a_dict() -> None:
    """A model you cannot persist is a model you must retrain on every process
    start, which defeats the point of separating it from the scoring."""
    component, config, result = _trained()
    revived = type(result.model)(
        lambda_prior=result.model.lambda_prior,
        m_probabilities={k: dict(v) for k, v in result.model.m_probabilities.items()},
        u_probabilities={k: dict(v) for k, v in result.model.u_probabilities.items()},
        iterations=result.model.iterations,
        converged=result.model.converged,
    )
    original = component.score_record(_LEFT["a1"], {"b1": _LEFT["b1"]}, config, result.model)
    copied = component.score_record(_LEFT["a1"], {"b1": _LEFT["b1"]}, config, revived)
    assert original[0].match_weight == pytest.approx(copied[0].match_weight)
