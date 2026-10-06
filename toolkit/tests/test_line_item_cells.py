"""Row matching on description, then every cell compared.

The first version matched rows on `(description, amount)` together. That is a
defensible pairing for precision and recall - it answers "did you return this
row correctly" in one number - and it is the wrong pairing for anything
finer-grained, because a row with a wrong amount comes back *unmatched*. The
detail is destroyed exactly where it is most wanted: you learn that a row is
missing, when what happened is that one of its four cells is wrong.

So the two questions are separated:

* **matching** is on the description, with position as the tiebreak when a
  description repeats. It answers "is this row present at all";
* **cell comparison** runs over the matched rows and answers "and is each of
  its cells right", per column.

A changed `quantity` was invisible before and is caught now, which was the
measured gap: row keys ignored it entirely.
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toolkit.extraction_eval import (  # noqa: E402
    CELL_COLUMNS,
    compare_line_items,
    line_item_scores,
)

_EXPECTED = [
    {"description": "Hex bolt stainless A", "quantity": 2,
     "unit_price": "25.00", "amount": "50.00"},
    {"description": "Washer flat B", "quantity": 3,
     "unit_price": "5.00", "amount": "15.00"},
    {"description": "Nylon lock nut C", "quantity": 1,
     "unit_price": "7.25", "amount": "7.25"},
]


def _perfect():
    return [dict(row) for row in _EXPECTED]


# --------------------------------------------------------------------------
# Matching on the description
# --------------------------------------------------------------------------


def test_a_perfect_table_matches_every_row_and_every_cell() -> None:
    result = compare_line_items(_EXPECTED, _perfect())
    assert result.matched_rows == 3
    assert result.expected_rows == 3 and result.extracted_rows == 3
    assert result.wrong_cells == []
    assert result.any_cell_wrong is False
    for column in CELL_COLUMNS:
        assert result.cell_accuracy(column) == 1.0
    assert result.cell_accuracy() == 1.0


def test_a_row_with_a_wrong_amount_is_matched_and_the_cell_is_wrong() -> None:
    """The point of the change. Under `(description, amount)` matching this row
    was *missing*; it is present with one wrong cell."""
    rows = _perfect()
    rows[1]["amount"] = "99.00"
    result = compare_line_items(_EXPECTED, rows)

    assert result.matched_rows == 3, "the row is present"
    assert result.cell_accuracy("amount") == 2 / 3
    assert result.cell_accuracy("description") == 1.0
    assert result.any_cell_wrong is True
    assert [(c.row, c.column) for c in result.wrong_cells] == [(1, "amount")]


def test_a_changed_quantity_is_caught() -> None:
    """Measured gap: the row key ignored quantity entirely, so a wrong one was
    invisible to every metric."""
    rows = _perfect()
    rows[0]["quantity"] = 7
    result = compare_line_items(_EXPECTED, rows)

    assert result.matched_rows == 3
    assert result.cell_accuracy("quantity") == 2 / 3
    assert result.any_cell_wrong is True
    assert result.wrong_cells[0].column == "quantity"
    assert result.wrong_cells[0].expected == 2
    assert result.wrong_cells[0].actual == 7


def test_a_changed_unit_price_is_caught() -> None:
    rows = _perfect()
    rows[2]["unit_price"] = "7.52"
    result = compare_line_items(_EXPECTED, rows)
    assert result.cell_accuracy("unit_price") == 2 / 3
    assert result.any_cell_wrong is True


def test_money_is_compared_by_value_not_by_spelling() -> None:
    """`RM1,240.50`, `1240.5` and `Decimal("1240.50")` are one value, and a
    harness that scores two of them wrong measures formatting."""
    rows = _perfect()
    rows[0]["amount"] = Decimal("50.000")
    rows[1]["amount"] = "15"
    rows[2]["unit_price"] = "RM7.25"
    result = compare_line_items(_EXPECTED, rows)
    assert result.any_cell_wrong is False


# --------------------------------------------------------------------------
# Position as the tiebreak
# --------------------------------------------------------------------------


def test_repeated_descriptions_pair_in_order() -> None:
    """A real invoice repeats a description with different quantities. Pairing
    them in document order is the only rule that does not need the cells it is
    about to compare."""
    expected = [
        {"description": "Cable tie", "quantity": 1, "unit_price": "2.00", "amount": "2.00"},
        {"description": "Cable tie", "quantity": 5, "unit_price": "2.00", "amount": "10.00"},
    ]
    actual = [
        {"description": "Cable tie", "quantity": 1, "unit_price": "2.00", "amount": "2.00"},
        {"description": "Cable tie", "quantity": 5, "unit_price": "2.00", "amount": "10.00"},
    ]
    assert compare_line_items(expected, actual).any_cell_wrong is False

    # Swapped, so pairing by position makes both rows wrong rather than
    # silently pairing each with its own twin.
    swapped = [actual[1], actual[0]]
    result = compare_line_items(expected, swapped)
    assert result.matched_rows == 2
    assert result.cell_accuracy("quantity") == 0.0


def test_a_missing_row_is_unmatched_and_not_counted_against_the_cells() -> None:
    """A dropped row is a recall failure. Scoring its four cells as wrong too
    would charge the same mistake twice and make cell accuracy track recall."""
    result = compare_line_items(_EXPECTED, _perfect()[:2])
    assert result.matched_rows == 2
    assert result.expected_rows == 3 and result.extracted_rows == 2
    assert result.cell_accuracy("amount") == 1.0
    assert result.missing_rows == 1
    # It is still a wrong table.
    assert result.is_exact is False


def test_a_hallucinated_row_is_reported_as_extra() -> None:
    rows = _perfect()
    rows.append({"description": "Grommet Z", "quantity": 1,
                 "unit_price": "99.00", "amount": "99.00"})
    result = compare_line_items(_EXPECTED, rows)
    assert result.matched_rows == 3
    assert result.extra_rows == 1
    assert result.is_exact is False
    assert result.any_cell_wrong is False, "the matched cells are all correct"


def test_a_misread_description_is_a_missing_row_and_an_extra_one() -> None:
    """Honest rather than clever. Matching on the description means a misread
    description cannot be paired, and guessing which row it meant would be the
    harness inventing ground truth."""
    rows = _perfect()
    rows[0]["description"] = "Hex b0lt stainless A"
    result = compare_line_items(_EXPECTED, rows)
    assert result.missing_rows == 1 and result.extra_rows == 1
    assert result.matched_rows == 2


def test_case_and_punctuation_do_not_break_a_match() -> None:
    rows = _perfect()
    rows[0]["description"] = "HEX BOLT STAINLESS A,"
    assert compare_line_items(_EXPECTED, rows).matched_rows == 3


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


def test_an_empty_table_does_not_divide_by_zero() -> None:
    empty = compare_line_items([], [])
    assert empty.matched_rows == 0
    assert empty.cell_accuracy() == 0.0
    assert empty.any_cell_wrong is False
    assert empty.is_exact is True, "nothing expected, nothing returned"

    assert compare_line_items(_EXPECTED, []).cell_accuracy() == 0.0
    assert compare_line_items([], _perfect()).extra_rows == 3


def test_a_cell_the_document_does_not_state_is_not_a_wrong_cell() -> None:
    """Expected null and actual null agree. A column absent from the schema
    should not drag every row down."""
    expected = [{"description": "Item A", "quantity": None,
                 "unit_price": None, "amount": "10.00"}]
    actual = [{"description": "Item A", "quantity": None,
               "unit_price": None, "amount": "10.00"}]
    result = compare_line_items(expected, actual)
    assert result.any_cell_wrong is False
    assert result.cell_accuracy("quantity") == 1.0


def test_a_cell_the_model_omitted_is_wrong() -> None:
    rows = _perfect()
    rows[0].pop("quantity")
    result = compare_line_items(_EXPECTED, rows)
    assert result.any_cell_wrong is True
    assert result.cell_accuracy("quantity") == 2 / 3


# --------------------------------------------------------------------------
# The old scores still work
# --------------------------------------------------------------------------


def test_precision_and_recall_now_follow_the_description_match() -> None:
    """`line_item_scores` keeps its shape and changes its meaning: it answers
    "were the rows found", with correctness living in the cell accuracy. Both
    halves are needed - a model that finds every row and fills it with rubbish
    scores 1.0 on one and badly on the other, which is the honest report."""
    rows = _perfect()
    rows[1]["amount"] = "99.00"
    score = line_item_scores(_EXPECTED, rows)
    assert score.matched == 3
    assert score.precision == 1.0 and score.recall == 1.0

    result = compare_line_items(_EXPECTED, rows)
    assert result.cell_accuracy() < 1.0
