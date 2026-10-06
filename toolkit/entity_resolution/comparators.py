"""Comparators for fields that are not strings.

The default comparator is normalised affine-gap distance, which is the right
tool for a name and the wrong question for a number or a date. `1000` and
`9999` are the same length with no digits in common and score as unrelated,
which is correct by accident; `1000` and `1001` differ in one character and
score as nearly identical, which is correct by accident too. `100` and `1000`
share three characters and score high, and that one is simply wrong. For dates
it is worse: `2026-01-31` and `2026-02-01` are one day apart and share almost
no characters.

Both comparators here return similarity in [0, 1] and both return **0.0** for a
value they cannot read, rather than raising. A record with a missing or junk
field is the normal case in the data this component exists for, and a
comparator that raises takes the whole batch down with it. The Fellegi-Sunter
machinery then does the right thing by itself: a field that reaches no level
contributes its `NO_MATCH_LEVEL` evidence, which is what "we learned nothing
here" already means in that model.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date

_NUMBER_NOISE = re.compile(r"[,\s$£€¥₹]")
_NUMBER_SHAPE = re.compile(r"^-?\d+(?:\.\d+)?$")

_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_YEAR_FIRST = re.compile(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})$")
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTHS = {name: index for index, name in enumerate(_MONTH_NAMES, start=1)}
_MONTHS.update({name[:3]: index for index, name in enumerate(_MONTH_NAMES, start=1)})
_DAY_MONTH = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})$")
_MONTH_DAY = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$")

DEFAULT_DATE_WINDOW_DAYS = 365.0
"""A year. Deliberately wide: this is the generic default, and a field with a
tighter meaning should say so with `date_comparator(window_days=...)`."""


def _to_number(text: str) -> float | None:
    cleaned = _NUMBER_NOISE.sub("", str(text).strip())
    if not _NUMBER_SHAPE.match(cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def numeric_similarity(a: str, b: str) -> float:
    """Relative difference, as similarity in [0, 1].

    Relative rather than absolute because one threshold cannot serve both
    scales otherwise: a difference of 10 is nothing on a million and everything
    on a dozen. `1 - |a - b| / max(|a|, |b|)`.

    Opposite signs score 0.0 however close the magnitudes. Two amounts of +500
    and -500 are not a near match; they are a credit and a debit, and treating
    them as similar is how a reconciliation quietly pairs the wrong rows.
    """
    left = _to_number(a)
    right = _to_number(b)
    if left is None or right is None:
        return 0.0
    if left == right:
        return 1.0
    if (left < 0) != (right < 0):
        return 0.0
    scale = max(abs(left), abs(right))
    if scale == 0:
        return 1.0
    return max(0.0, 1.0 - abs(left - right) / scale)


def numeric_comparator(scale: float | None = None) -> Callable[[str, str], float]:
    """A numeric comparator, optionally on an absolute scale.

    `scale` is the difference at which similarity reaches zero, and it is the
    right choice for a quantity that legitimately passes through zero — a
    relative difference is undefined there and nearly meaningless just above
    it. Without a scale this is `numeric_similarity`.
    """
    if scale is None:
        return numeric_similarity
    if scale <= 0:
        raise ValueError("scale must be positive")

    def compare(a: str, b: str) -> float:
        left = _to_number(a)
        right = _to_number(b)
        if left is None or right is None:
            return 0.0
        return max(0.0, 1.0 - abs(left - right) / scale)

    return compare


def parse_date(text: str) -> date | None:
    """ISO first, then the forms records are actually typed in.

    Deliberately narrower than `extraction.dates`: this unit must not depend on
    that one, and an ambiguous `03/10/2026` has no document-level hint to
    resolve it here, so it is not accepted at all rather than guessed.
    """
    raw = str(text).strip()
    for pattern in (_ISO, _YEAR_FIRST):
        match = pattern.match(raw)
        if match:
            year, month, day = (int(part) for part in match.groups())
            try:
                return date(year, month, day)
            except ValueError:
                return None
    for pattern in (_DAY_MONTH, _MONTH_DAY):
        match = pattern.match(raw)
        if not match:
            continue
        groups = list(match.groups())
        if groups[0].isdigit():
            day_text, month_text, year_text = groups
        else:
            month_text, day_text, year_text = groups
        lowered = month_text.lower().rstrip(".")
        named = _MONTHS.get(lowered) or _MONTHS.get(lowered[:3])
        if not named:
            return None
        try:
            return date(int(year_text), named, int(day_text))
        except ValueError:
            return None
    return None


def days_apart(a: str, b: str) -> int | None:
    """Absolute difference in days, or None if either side is unreadable.

    Exposed because it is often the number a human wants in a report, and
    "similarity 0.97" is not.
    """
    left = parse_date(a)
    right = parse_date(b)
    if left is None or right is None:
        return None
    return abs((left - right).days)


def date_comparator(
    window_days: float = DEFAULT_DATE_WINDOW_DAYS,
) -> Callable[[str, str], float]:
    """Linear decay over `window_days`, reaching zero at the window edge.

    Linear rather than exponential on purpose: the threshold on a
    `ComparisonLevel` is then readable as a number of days. At a 7-day window,
    a threshold of 0.57 is "within three days", which someone can check. With
    an exponential decay the same threshold is a number nobody can picture, and
    a comparison level nobody can picture is one nobody will tune.

    The window is per field because fields differ: a date of birth three days
    out is a transcription error, an invoice date three days out is a different
    invoice.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    def compare(a: str, b: str) -> float:
        days = days_apart(a, b)
        if days is None:
            return 0.0
        return max(0.0, 1.0 - days / window_days)

    return compare


date_similarity = date_comparator()
"""Date comparator on the default one-year window."""


__all__ = [
    "DEFAULT_DATE_WINDOW_DAYS",
    "date_comparator",
    "date_similarity",
    "days_apart",
    "numeric_comparator",
    "numeric_similarity",
    "parse_date",
]
