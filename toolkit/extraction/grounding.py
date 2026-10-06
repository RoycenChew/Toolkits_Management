"""Word-level grounding: did this value actually appear on the page, and where?

The version this replaces was substring matching over normalised block text. It
reported a fabricated `quantity: 7` as found, because `7` sits inside
`INV-2026-0417` and inside the unit price `7.25`. A grounding signal with false
positives in it is not a weak signal, it is a misleading one: the whole purpose
is to be the set a human checks first.

Three things fix it:

* **Token boundaries.** A value matches runs of *consecutive whole words*, never
  a substring. When the document has no words (a plain text source) the same
  rule is applied to block text with `(?<!\\w)…(?!\\w)` boundaries.
* **Comparison by parsed value, per token.** `7` is compared against each
  token's parsed number, so it matches neither `7.25` nor an invoice number.
  Equally, a model returning `6511.05` still matches a page reading
  `MYR 6,511.05` — the match is `normalized` rather than `exact`, and saying
  which is the point.
* **Row locality.** Inside a table, a cell is only accepted on its own row's
  line. Without that, `quantity: 1` in row 7 grounds happily on row 1's `1`,
  and the evidence box points at the wrong row - which is a worse failure than
  no box, because it looks right.

`fuzzy_ocr` is the one match class that is deliberately **not** grounded. A
low-confidence OCR word one edit away from the value is worth reporting, since
it is the likeliest explanation of a near miss, but it is not evidence that the
document says what the model claims.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from ..core.models import BBox, Provenance, Word, WordSource
from .dates import date_variants, is_iso
from .models import Evidence, FieldType, MatchClass
from .numbers import as_decimal, parse_decimal

_STRIP = ":,;()[]{}.\"'“”‘’!?*|"
"""Punctuation stripped from a token's edges before comparison. Interior
characters stay: removing the hyphens from `INV-2026-0417` would let it match a
different reference."""

_SPLIT = re.compile(r"\S+")

DEFAULT_OCR_CONFIDENCE = 0.75
"""Below this, an OCR word is unreliable enough that a one-edit difference is
more likely a misread than a different word. Above it, a near miss is treated
as a genuine mismatch."""

_VERTICAL_OVERLAP = 0.5
"""Fraction of the shorter box's height that must overlap for two words to be
called the same line. Chosen because digits and letters on one baseline differ
in height by less than half (no descenders on `4`), while adjacent table rows
16 points apart do not overlap at all."""


@dataclass(frozen=True)
class _Token:
    """One comparable unit of the document, with where it sits."""

    text: str
    norm: str
    number: Decimal | None
    page: int
    bbox: BBox | None
    confidence: float | None
    source: WordSource


@dataclass(frozen=True)
class Match:
    """Where a value was found, and how exactly."""

    match: MatchClass
    evidence: list[Evidence]
    start: int = -1
    length: int = 0

    @property
    def grounded(self) -> bool:
        return self.match in (MatchClass.EXACT, MatchClass.NORMALIZED)


NOT_CHECKED = Match(MatchClass.NOT_CHECKED, [])
NOT_FOUND = Match(MatchClass.NOT_FOUND, [])

_CONTAINER_TYPES = frozenset({FieldType.OBJECT, FieldType.ARRAY})


def normalise_token(text: str) -> str:
    """Lowercase and drop edge punctuation. `ACME,` and `acme` are one word."""
    return text.strip().strip(_STRIP).lower()


def _token_from_word(word: Word) -> _Token:
    return _Token(
        text=word.text,
        norm=normalise_token(word.text),
        number=parse_decimal(word.text),
        page=word.page,
        bbox=word.bbox,
        confidence=word.confidence,
        source=word.source,
    )


def _tokens_from_segments(
    segments: Sequence[tuple[str, Provenance | None]],
) -> list[_Token]:
    """The fallback layer: block text split on whitespace.

    Each token carries the *block's* box, not its own, because that is all this
    layer knows. A consumer can tell the difference: the evidence box is the
    same for every token in the block.
    """
    tokens: list[_Token] = []
    for text, provenance in segments:
        page = provenance.page if provenance else 1
        bbox = provenance.bbox if provenance else None
        for piece in _SPLIT.findall(text):
            tokens.append(
                _Token(
                    text=piece,
                    norm=normalise_token(piece),
                    number=parse_decimal(piece),
                    page=page,
                    bbox=bbox,
                    confidence=None,
                    source=WordSource.TEXT_LAYER,
                )
            )
    return tokens


def _same_line(a: BBox, b: BBox) -> bool:
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap <= 0:
        return False
    shortest = min(a.height, b.height)
    return shortest <= 0 or overlap / shortest >= _VERTICAL_OVERLAP


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    """True when `a` becomes `b` with one substitution, insertion or deletion.

    Bounded rather than general: the question is only ever "is this one OCR
    slip away", and a bounded check is linear and cannot drift into matching
    genuinely different words.
    """
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    return any(
        longer[:index] + longer[index + 1 :] == shorter for index in range(len(longer))
    )


class Grounder:
    """Searches one document's tokens for extracted values.

    Built once per extraction. `has_words` says which layer is in use, because
    the precision of every box it hands out depends on it.
    """

    def __init__(
        self,
        segments: Sequence[tuple[str, Provenance | None]],
        words: Sequence[Word] = (),
        date_order: str | None = None,
        ocr_confidence: float = DEFAULT_OCR_CONFIDENCE,
    ) -> None:
        self.has_words = bool(words)
        self._tokens = (
            [_token_from_word(word) for word in words]
            if self.has_words
            else _tokens_from_segments(segments)
        )
        self._date_order = date_order
        self._ocr_confidence = ocr_confidence
        self._by_norm: dict[str, list[int]] = {}
        for index, token in enumerate(self._tokens):
            if token.norm:
                self._by_norm.setdefault(token.norm, []).append(index)

    # --- the public question ---------------------------------------------

    def locate(
        self,
        value: object,
        field_type: FieldType,
        *,
        line: BBox | None = None,
        page: int | None = None,
        exclude: Iterable[int] = (),
    ) -> Match:
        """Find `value` among the tokens.

        `line` and `page` restrict the search to one row of one page, which is
        how a table cell is kept from grounding on another row. `exclude` skips
        token runs already claimed, so two identical rows ground on their own
        words rather than both on the first.
        """
        if field_type in _CONTAINER_TYPES or isinstance(value, bool):
            return NOT_CHECKED
        if value is None or field_type is FieldType.BOOLEAN:
            return NOT_CHECKED
        if not self._tokens:
            return NOT_FOUND

        skip = set(exclude)
        if field_type is FieldType.DATE and isinstance(value, str) and is_iso(value):
            return self._locate_date(value, line, page, skip)

        number = as_decimal(value) if not isinstance(value, str) else None
        if number is not None:
            return self._locate_number(number, str(value), line, page, skip)

        if isinstance(value, Decimal):
            return self._locate_number(value, str(value), line, page, skip)

        text = str(value).strip()
        if not text:
            return NOT_CHECKED
        if field_type in (FieldType.NUMBER, FieldType.INTEGER, FieldType.DECIMAL):
            parsed = parse_decimal(text)
            if parsed is not None:
                return self._locate_number(parsed, text, line, page, skip)
        return self._locate_text(text, line, page, skip)

    # --- the three searches ----------------------------------------------

    def _locate_number(
        self,
        number: Decimal,
        rendered: str,
        line: BBox | None,
        page: int | None,
        skip: set[int],
    ) -> Match:
        """Compare parsed value per token, which is what kills the false
        positives: `7` never equals `7.25`, and `INV-2026-0417` parses to
        nothing at all."""
        candidates = [
            index
            for index, token in enumerate(self._tokens)
            if index not in skip
            and token.number is not None
            and token.number == number
            and self._in_scope(token, line, page)
        ]
        best = self._nearest(candidates, line)
        if best is None:
            return self._fuzzy(rendered, line, page, skip)
        token = self._tokens[best]
        match = (
            MatchClass.EXACT
            if token.text == rendered or normalise_token(token.text) == rendered.lower()
            else MatchClass.NORMALIZED
        )
        return Match(match, [self._evidence([best])], best, 1)

    def _locate_text(
        self,
        text: str,
        line: BBox | None,
        page: int | None,
        skip: set[int],
    ) -> Match:
        pieces = [normalise_token(piece) for piece in _SPLIT.findall(text)]
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            return NOT_CHECKED
        starts = [
            index
            for index in self._by_norm.get(pieces[0], ())
            if self._run_matches(index, pieces, line, page, skip)
        ]
        best = self._nearest(starts, line)
        if best is None:
            return self._fuzzy(text, line, page, skip)
        run = list(range(best, best + len(pieces)))
        joined = " ".join(self._tokens[i].text for i in run)
        match = MatchClass.EXACT if joined == text.strip() else MatchClass.NORMALIZED
        return Match(match, [self._evidence(run)], best, len(pieces))

    def _locate_date(
        self,
        iso_text: str,
        line: BBox | None,
        page: int | None,
        skip: set[int],
    ) -> Match:
        """A date is only ever `exact` when the page itself is ISO; every other
        spelling the document chose is a `normalized` match."""
        for variant in date_variants(iso_text, self._date_order):
            found = self._locate_text(variant, line, page, skip)
            if found.grounded:
                match = (
                    MatchClass.EXACT
                    if variant == iso_text and found.match is MatchClass.EXACT
                    else MatchClass.NORMALIZED
                )
                return Match(match, found.evidence, found.start, found.length)
        return NOT_FOUND

    def _fuzzy(
        self, rendered: str, line: BBox | None, page: int | None, skip: set[int]
    ) -> Match:
        """One edit away from a low-confidence OCR word: reported, not trusted."""
        if not self.has_words:
            return NOT_FOUND
        target = normalise_token(rendered)
        if len(target) < 3:
            return NOT_FOUND
        for index, token in enumerate(self._tokens):
            if index in skip or token.source is not WordSource.OCR:
                continue
            if token.confidence is None or token.confidence >= self._ocr_confidence:
                continue
            if not self._in_scope(token, line, page):
                continue
            if _edit_distance_at_most_one(token.norm, target):
                return Match(MatchClass.FUZZY_OCR, [self._evidence([index])], index, 1)
        return NOT_FOUND

    # --- scoping and evidence --------------------------------------------

    def _in_scope(self, token: _Token, line: BBox | None, page: int | None) -> bool:
        if page is not None and token.page != page:
            return False
        if line is None:
            return True
        return token.bbox is not None and _same_line(token.bbox, line)

    def _run_matches(
        self,
        start: int,
        pieces: Sequence[str],
        line: BBox | None,
        page: int | None,
        skip: set[int],
    ) -> bool:
        if start + len(pieces) > len(self._tokens):
            return False
        first = self._tokens[start]
        for offset, piece in enumerate(pieces):
            index = start + offset
            token = self._tokens[index]
            if index in skip or token.norm != piece or token.page != first.page:
                return False
            if not self._in_scope(token, line, page):
                return False
        return True

    def _nearest(self, candidates: Sequence[int], line: BBox | None) -> int | None:
        """First in reading order, or — when a row is given and the value
        repeats on it — the one closest to that row's anchor."""
        if not candidates:
            return None
        if line is None:
            return candidates[0]

        def distance(index: int) -> tuple[float, int]:
            bbox = self._tokens[index].bbox
            if bbox is None:
                return (float("inf"), index)
            centre = (line.y0 + line.y1) / 2
            token_centre = (bbox.y0 + bbox.y1) / 2
            return (abs(token_centre - centre), index)

        return min(candidates, key=distance)

    def _evidence(self, indices: Sequence[int]) -> Evidence:
        tokens = [self._tokens[index] for index in indices]
        boxes = [token.bbox for token in tokens if token.bbox is not None]
        merged: BBox | None = None
        for box in boxes:
            merged = box if merged is None else merged.merge(box)
        confidences = [t.confidence for t in tokens if t.confidence is not None]
        return Evidence(
            page=tokens[0].page,
            bbox=merged,
            words=[token.text for token in tokens],
            source=tokens[0].source,
            min_confidence=min(confidences) if confidences else None,
        )


__all__ = [
    "DEFAULT_OCR_CONFIDENCE",
    "Grounder",
    "Match",
    "normalise_token",
]
