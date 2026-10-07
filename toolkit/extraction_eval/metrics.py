"""The comparison rules, separated from the harness that applies them.

Two decisions carry this module.

**What equality means.** A model that returns the right fact in a different
shape is right: `"USD 35.00"`, `35`, `Decimal("35.00")` and `"35.0"` are one
value, and a harness that scores three of them wrong measures formatting rather
than extraction. So comparison is by normalised form - numbers as `Decimal`,
strings casefolded with their surrounding punctuation dropped - and never by
`repr`. What it deliberately does *not* do is fuzzy matching: `"ACME Corp"` and
`"ACME Corporation"` are different answers, and a threshold that calls them
equal is a threshold that hides the error it was meant to find.

**What identifies a table row.** Rows are matched on the **description**, with
position as the tiebreak when one repeats.

The first version matched on `(description, amount)` together. That is a
defensible pairing for precision and recall - it answers "did you return this
row correctly" in one number - and it destroys the detail wherever anything
finer is wanted: a row with a wrong amount comes back *unmatched*, so the
report says a row is missing when in truth one of its four cells is wrong. It
also could not see a wrong `quantity` at all, because the key ignored it.

So the two questions are separated. `line_item_scores` answers "is the row
present" and `compare_line_items` answers "and is each of its cells right",
per column. Both are needed: a model that finds every row and fills it with
rubbish scores 1.0 on the first and badly on the second, which is the honest
report.

Matching on the description means a *misread* description cannot be paired, and
is reported as one missing row and one extra. Guessing which row it meant would
be the harness inventing ground truth.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

_EDGE_PUNCTUATION = ":,;()[]{}.\"'!?*|"
_CURRENCY = re.compile(
    r"[$£€¥₹]"
    r"|\b(?:usd|eur|gbp|jpy|myr|sgd|aud|cad|chf|cny|inr)\b"
    # Symbols written against the digits, so no trailing word boundary: there
    # is none between the `M` and the `4` of `RM4,094.28`. The lookahead is
    # what keeps `RMS` and `ROOM12` from being read as numbers.
    r"|\b(?:rm|s\$|hk\$|a\$|nz\$|c\$)(?=\s*[-\d(])",
    re.IGNORECASE,
)

DESCRIPTION_KEYS = ("description", "desc", "item", "name", "label")
"""Field names a row's distinguishing text is likely to be under. Checked in
order; the first one present wins. A schema that calls it something else passes
its own names to `line_item_scores`."""

AMOUNT_KEYS = ("amount", "total", "line_total", "value", "price")


def mean(values: Sequence[float]) -> float:
    """0.0 for an empty sequence, not a ZeroDivisionError.

    A metric over zero cases is genuinely zero information, and a harness that
    raises here cannot report a run where one stage produced nothing - which is
    exactly the run you most want a report for.
    """
    return (sum(values) / len(values)) if values else 0.0


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # str() first: Decimal(35.1) is not 35.1.
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    if not isinstance(value, str):
        return None
    text = _CURRENCY.sub("", value).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = text.replace(",", "").replace(" ", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return None
    return -parsed if negative else parsed


def normalise_text(value: Any) -> str:
    """Casefold, collapse whitespace, drop punctuation at the edges of tokens."""
    tokens = str(value).split()
    stripped = [token.strip(_EDGE_PUNCTUATION).casefold() for token in tokens]
    return " ".join(token for token in stripped if token)


def values_equal(expected: Any, actual: Any) -> bool:
    """Normalised exact match.

    `None` is only equal to `None`: a field the model omitted is a wrong answer,
    not a missing measurement, and treating it as absent lets a model score well
    by answering less.
    """
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    expected_number = _as_decimal(expected)
    actual_number = _as_decimal(actual)
    if expected_number is not None and actual_number is not None:
        return expected_number == actual_number
    return normalise_text(expected) == normalise_text(actual)


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def canonical_number(value: Decimal) -> str:
    """A Decimal rendered so that equal values give equal strings.

    `str()` does not: `Decimal("1")` and `Decimal("1.00")` are equal and print
    differently, so hashing on `str` sent two identical amounts to two
    different keys and the rows never matched. `normalize()` fixes the trailing
    zeros but renders 20 as `2E+1`, so the result is formatted with `f`.
    """
    return format(value.normalize(), "f")


def row_key(
    row: Mapping[str, Any],
    description_keys: Sequence[str] = DESCRIPTION_KEYS,
    amount_keys: Sequence[str] = AMOUNT_KEYS,
) -> str:
    """What identifies one row: its normalised description.

    `amount_keys` is still accepted so every existing call keeps working, and
    is no longer part of the key - see this module's docstring for why pairing
    on the amount hid a wrong cell as a missing row.
    """
    description = _first_present(row, description_keys)
    return normalise_text("" if description is None else description)


@dataclass(frozen=True)
class LineItemScore:
    """Row-level precision, recall and F1 for one case.

    All three, because they fail in opposite directions and only reporting one
    hides it: a model that returns a single confident row has high precision and
    useless recall, and one that returns every line twice has the reverse.
    """

    expected_rows: int
    extracted_rows: int
    matched: int

    @property
    def precision(self) -> float:
        return (self.matched / self.extracted_rows) if self.extracted_rows else 0.0

    @property
    def recall(self) -> float:
        return (self.matched / self.expected_rows) if self.expected_rows else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return (2 * self.precision * self.recall / total) if total else 0.0


def line_item_scores(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    description_keys: Sequence[str] = DESCRIPTION_KEYS,
    amount_keys: Sequence[str] = AMOUNT_KEYS,
) -> LineItemScore:
    """Match rows as multisets of descriptions: is the row present at all?

    Correctness is `compare_line_items`' question, per cell. Both are needed -
    a model that finds every row and fills it with rubbish scores 1.0 here and
    badly there, which is the honest report.

    Multisets, not sets: a correct row returned twice is one match and one false
    positive. With sets, a model that repeats every row would score the same as
    one that does not, and precision could exceed 1.
    """
    remaining: dict[str, int] = {}
    for row in expected:
        key = row_key(row, description_keys, amount_keys)
        remaining[key] = remaining.get(key, 0) + 1

    matched = 0
    for row in actual:
        key = row_key(row, description_keys, amount_keys)
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
            matched += 1
    return LineItemScore(
        expected_rows=len(expected), extracted_rows=len(actual), matched=matched
    )


CELL_COLUMNS: tuple[str, ...] = ("description", "quantity", "unit_price", "amount")
"""The cells compared on every matched row. A schema naming its columns
differently passes its own tuple."""


@dataclass(frozen=True)
class WrongCell:
    row: int
    """Index into the *expected* rows, so a report can name the row a human
    would count to."""
    column: str
    expected: Any
    actual: Any

    def render(self) -> str:
        return "line_items[%d].%s: expected %r, got %r" % (
            self.row, self.column, self.expected, self.actual
        )


@dataclass(frozen=True)
class LineItemComparison:
    """Rows matched by description, then every cell compared.

    `cell_accuracy` is over *matched* rows only. A dropped row is a recall
    failure and is counted as one; scoring its four cells wrong as well would
    charge the same mistake twice and make cell accuracy track recall.
    """

    expected_rows: int
    extracted_rows: int
    matched_rows: int
    wrong_cells: list[WrongCell] = field(default_factory=list)
    cells_compared: Mapping[str, int] = field(default_factory=dict)
    cells_correct: Mapping[str, int] = field(default_factory=dict)

    @property
    def missing_rows(self) -> int:
        return self.expected_rows - self.matched_rows

    @property
    def extra_rows(self) -> int:
        return self.extracted_rows - self.matched_rows

    @property
    def any_cell_wrong(self) -> bool:
        return bool(self.wrong_cells)

    @property
    def is_exact(self) -> bool:
        """Every expected row present, nothing extra, every cell right."""
        return (
            self.missing_rows == 0
            and self.extra_rows == 0
            and not self.wrong_cells
        )

    def cell_accuracy(self, column: str | None = None) -> float:
        """Per column, or over every column when none is named."""
        if column is not None:
            compared = self.cells_compared.get(column, 0)
            return (self.cells_correct.get(column, 0) / compared) if compared else 0.0
        compared = sum(self.cells_compared.values())
        if not compared:
            return 0.0
        return sum(self.cells_correct.values()) / compared


def compare_line_items(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    columns: Sequence[str] = CELL_COLUMNS,
    description_keys: Sequence[str] = DESCRIPTION_KEYS,
) -> LineItemComparison:
    """Pair rows by description, in order, then compare every cell.

    Pairing in order is what makes a repeated description work: a real invoice
    lists the same item twice with different quantities, and the only rule that
    does not need the cells it is about to compare is document order.
    """
    buckets: dict[str, list[int]] = {}
    for index, row in enumerate(actual):
        buckets.setdefault(row_key(row, description_keys), []).append(index)

    wrong: list[WrongCell] = []
    compared: dict[str, int] = {column: 0 for column in columns}
    correct: dict[str, int] = {column: 0 for column in columns}
    matched = 0

    for index, want in enumerate(expected):
        candidates = buckets.get(row_key(want, description_keys))
        if not candidates:
            continue
        got = actual[candidates.pop(0)]
        matched += 1
        for column in columns:
            if column not in want:
                # The ground truth is silent about this column, so it is
                # unmeasured rather than wrong. Comparing an absent expectation
                # against an extracted value penalises the model for returning
                # something nobody recorded - the "fires on correct work"
                # failure, which ends with the metric switched off. An expected
                # value of None is different: that is a recorded "the document
                # does not state it", and it is compared.
                continue
            compared[column] += 1
            if values_equal(want.get(column), got.get(column)):
                correct[column] += 1
            else:
                wrong.append(
                    WrongCell(index, column, want.get(column), got.get(column))
                )

    return LineItemComparison(
        expected_rows=len(expected),
        extracted_rows=len(actual),
        matched_rows=matched,
        wrong_cells=wrong,
        cells_compared=compared,
        cells_correct=correct,
    )


def field_accuracy(outcomes: Sequence[Any]) -> float:
    """Share of compared paths that were right. 0.0 when nothing was compared."""
    return mean([1.0 if o.correct else 0.0 for o in outcomes])


def grounding_rate(outcomes: Sequence[Any]) -> float:
    """Share of non-null extracted values found in the document.

    Counted over values the extractor actually produced. Including the ones it
    left null would make a model that answers nothing look perfectly grounded.
    """
    produced = [o for o in outcomes if o.actual is not None]
    return mean([1.0 if o.grounded else 0.0 for o in produced])


__all__ = [
    "AMOUNT_KEYS",
    "CELL_COLUMNS",
    "DESCRIPTION_KEYS",
    "LineItemComparison",
    "LineItemScore",
    "WrongCell",
    "compare_line_items",
    "canonical_number",
    "field_accuracy",
    "grounding_rate",
    "line_item_scores",
    "mean",
    "normalise_text",
    "row_key",
    "values_equal",
]
