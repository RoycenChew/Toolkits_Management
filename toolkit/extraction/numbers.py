"""Number parsing shared by validation and grounding.

Split out of `component` for one reason: a `Decimal` built from a `float` has
already lost the thing it was chosen for. `Decimal(1240.50)` is
1240.5000000000000454747350886464118957519531250, because 1240.50 is not
representable in binary. Both parsers therefore share one normaliser that
returns a *string*, and each builds its own type from that string.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_CURRENCY = re.compile(
    r"[$£€¥₹]"
    r"|\b(?:usd|eur|gbp|jpy|myr|sgd|aud|cad|chf|cny|inr)\b"
    # Symbols written against the digits, so no trailing word boundary: there
    # is none between the `M` and the `4` of `RM4,094.28`. The lookahead is
    # what keeps `RMS` and `ROOM12` from being read as numbers.
    r"|\b(?:rm|s\$|hk\$|a\$|nz\$|c\$)(?=\s*[\d(])",
    re.IGNORECASE,
)
"""Currency symbols and ISO codes, stripped before numeric parsing. The word
boundaries matter: without them "inr" would match inside an ordinary word."""

_DECIMAL_SHAPE = re.compile(r"^-?\d+(?:\.\d+)?$")


def normalise_number_text(raw: str) -> str | None:
    """Reduce a money-shaped string to a plain signed decimal string.

    Handles three things the naive strip-commas approach gets wrong, all found
    in the stress corpus:

    * **European notation.** "2.450,75" means 2450.75, not 2.45075. The rule
      that disambiguates is positional, not locale-based: when both separators
      appear, the **rightmost** one is the decimal point. That holds for both
      conventions and needs no locale guess.
    * **Accounting negatives.** "($310.00)" is -310. Parentheses are how
      finance writes a negative, and dropping them inverts the sign of every
      credit note.
    * **Currency words, codes and symbols.** "EUR 2.450,75", "USD 1,240.50"
      and "RM4,094.28". A symbol written against the digits needs no trailing
      word boundary and must not have one: there is none between the `M` and
      the `4`.

    A single separator is ambiguous in principle — "1.234" is 1234 in Germany
    and 1.234 elsewhere. Resolved by digit grouping: exactly three digits after
    a lone separator is read as a thousands group, which is the convention that
    makes "1.234" and "1,234" both mean 1234.
    """
    text = raw.strip()
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    text = _CURRENCY.sub("", text).strip()
    if text.startswith("-"):
        negative = not negative
        text = text[1:].strip()
    text = text.replace(" ", "").replace(" ", "").replace(" ", "")
    if not text:
        return None

    last_dot = text.rfind(".")
    last_comma = text.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        # Both present: the rightmost separator is the decimal point.
        if last_comma > last_dot:
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif last_comma >= 0:
        tail = text[last_comma + 1 :]
        text = (
            text.replace(",", "")
            if len(tail) == 3 and tail.isdigit()
            else text.replace(",", ".")
        )
    elif last_dot >= 0:
        tail = text[last_dot + 1 :]
        if len(tail) == 3 and tail.isdigit() and text.count(".") >= 1 and len(text) > 4:
            # "2.450" with no other separator: a thousands group.
            text = text.replace(".", "")

    if not _DECIMAL_SHAPE.fullmatch(text):
        return None
    return ("-" + text) if negative else text


def parse_number(raw: str) -> float | None:
    """Parse a money-shaped string into a float."""
    text = normalise_number_text(raw)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_decimal(raw: str) -> Decimal | None:
    """Parse a money-shaped string into a `Decimal`, never via `float`."""
    text = normalise_number_text(raw)
    if text is None:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def as_decimal(value: object) -> Decimal | None:
    """Best-effort `Decimal` view of any extracted scalar, for comparison.

    `str(value)` is deliberate for floats: it gives the shortest repr that
    round-trips, so 6511.05 compares equal to the document's "6,511.05" instead
    of to its binary expansion.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    if isinstance(value, str):
        return parse_decimal(value)
    return None


__all__ = [
    "as_decimal",
    "normalise_number_text",
    "parse_decimal",
    "parse_number",
]
