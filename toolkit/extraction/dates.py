"""Date coercion and the reverse: how a page might have spelled an ISO date.

Two directions, one table of conventions, because they have to agree. If
coercion reads `03/10/2026` as 3 October and grounding looks for `10/03/2026`,
a correctly extracted date is reported as fabricated — a signal that fires on
correct work, which is worse than no signal.

`date_order` is a document-level fact the caller knows and this module cannot:
a Malaysian invoice writes day first, a US one month first. Without it an
ambiguous date stays refused, which is the only honest default.
"""
from __future__ import annotations

import re
from datetime import date

DateOrder = str
"""One of "DMY", "MDY", "YMD". Typed as `Literal` on `ExtractionConfig`; kept
loose here so this module imports nothing."""

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_FIRST = re.compile(r"(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})")
_YEAR_LAST = re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})")

_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_MONTHS = {name.lower(): index for index, name in enumerate(_MONTH_NAMES, start=1)}
# Documents write "Mar 16, 2024" far more often than "March 16, 2024", and a
# full-name-only lookup silently failed on every abbreviation.
_MONTHS.update({name.lower()[:3]: index for index, name in enumerate(_MONTH_NAMES, 1)})

_DAY_MONTH_NAME = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})")
_MONTH_NAME_DAY = re.compile(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})")


def is_iso(text: str) -> bool:
    return bool(_ISO.fullmatch(text.strip()))


def _iso(year: int, month: int, day: int) -> str | None:
    """Render, but only if the date exists. 2026-02-30 is a model error worth a
    repair round, not something to pass downstream."""
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def coerce_date(text: str, date_order: DateOrder | None = None) -> str | None:
    """Rewrite a written date as YYYY-MM-DD, or return None if it is ambiguous.

    Deliberately refuses `01/02/2024` without a `date_order`: it is 1 February
    or 2 January depending on where the document came from, and guessing
    silently corrupts data. A repair round that asks the model is the correct
    cost there.
    """
    text = text.strip()
    if is_iso(text):
        return text

    match = _YEAR_FIRST.fullmatch(text)
    if match:
        year, first, second = (int(g) for g in match.groups())
        # Year first means month then day in every convention that writes it.
        return _iso(year, first, second)

    match = _YEAR_LAST.fullmatch(text)
    if match:
        first, second, year = (int(g) for g in match.groups())
        if date_order == "DMY":
            return _iso(year, second, first)
        if date_order == "MDY":
            return _iso(year, first, second)
        # No usable hint: resolve only when one component cannot be a month.
        if first > 12 and second <= 12:
            return _iso(year, second, first)
        if second > 12 and first <= 12:
            return _iso(year, first, second)
        return None

    for pattern in (_DAY_MONTH_NAME, _MONTH_NAME_DAY):
        match = pattern.fullmatch(text)
        if not match:
            continue
        groups = list(match.groups())
        if groups[0].isdigit():
            day_text, month_text, year_text = groups
        else:
            month_text, day_text, year_text = groups
        lowered = month_text.lower().rstrip(".")
        month = _MONTHS.get(lowered) or _MONTHS.get(lowered[:3])
        if month:
            return _iso(int(year_text), month, int(day_text))
    return None


def date_variants(iso_text: str, date_order: DateOrder | None = None) -> list[str]:
    """Every spelling of an ISO date that this module would read back as it.

    Used by grounding, so the rule is strict: a numeric day-first or
    month-first form is only included when reading it back is unambiguous —
    either because `date_order` says which convention the document uses, or
    because the day is past the 12th and no other reading exists. Including
    `03/10/2026` for 3 October without a hint would claim the page as evidence
    when the page may well have meant 10 March.
    """
    if not is_iso(iso_text):
        return []
    year, month, day = (int(part) for part in iso_text.split("-"))
    name = _MONTH_NAMES[month - 1]

    variants = [
        iso_text,
        "%04d/%02d/%02d" % (year, month, day),
        "%04d.%02d.%02d" % (year, month, day),
        "%d %s %d" % (day, name, year),
        "%02d %s %d" % (day, name, year),
        "%s %d, %d" % (name, day, year),
        "%d %s %d" % (day, name[:3], year),
        "%s %d, %d" % (name[:3], day, year),
    ]

    day_first = date_order == "DMY" or (date_order is None and day > 12)
    month_first = date_order == "MDY" or (date_order is None and day > 12)
    for separator in ("/", ".", "-"):
        if day_first:
            variants.append("%02d%s%02d%s%04d" % (day, separator, month, separator, year))
            variants.append("%d%s%d%s%04d" % (day, separator, month, separator, year))
        if month_first:
            variants.append("%02d%s%02d%s%04d" % (month, separator, day, separator, year))
            variants.append("%d%s%d%s%04d" % (month, separator, day, separator, year))

    seen: set[str] = set()
    ordered: list[str] = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            ordered.append(variant)
    return ordered


__all__ = ["coerce_date", "date_variants", "is_iso"]
