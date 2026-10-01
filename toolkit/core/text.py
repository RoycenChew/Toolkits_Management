"""Text normalisation applied at the document boundary.

Real documents carry typographic artefacts that break exact matching in ways
that are invisible when you read the output:

* **Ligatures.** A PDF produced by LaTeX stores "configuration" as `conﬁguration`
  with a single ﬁ glyph. A user typing "configuration" matches nothing, and the
  bug is undetectable by eye because both render identically.
* **Non-breaking spaces.** `Maximum voltage` does not match `Maximum voltage`.
  These are everywhere in Word exports and HTML conversions.
* **Zero-width characters.** Invisible, and they split tokens.
* **Soft hyphens.** `re­sistance` is one word that looks like one word and
  compares as two.
* **Decomposed accents.** `Café` and `Café` are different strings.

NFKC composes accents, expands ligatures and folds the exotic space characters,
which handles most of this in one pass. The remainder are removed explicitly
because NFKC preserves them.

Applied in the `DocumentSource` adapters — the boundary where untrusted text
enters — so every downstream component works on normalised text and no component
has to remember to do it.
"""
from __future__ import annotations

import unicodedata

_ZERO_WIDTH = {
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "⁠",  # word joiner
    "﻿",  # BOM / zero-width no-break space
}

_SOFT_HYPHEN = "­"


def normalise_text(text: str, collapse_whitespace: bool = True) -> str:
    """NFKC-normalise and strip invisible characters.

    `collapse_whitespace=False` preserves newlines, which matters for sources
    whose structure is line-based (Markdown) and not for sources that have
    already resolved layout into blocks.
    """
    if not text:
        return ""

    # NFKC first: it folds ligatures, non-breaking and other exotic spaces, and
    # composes combining marks. Doing it before removal means fewer special
    # cases afterwards.
    out = unicodedata.normalize("NFKC", text)

    if _SOFT_HYPHEN in out:
        # A soft hyphen is a *suggested* break point, not a character. Removing
        # it rejoins the word, which is what a reader sees anyway.
        out = out.replace(_SOFT_HYPHEN, "")

    for char in _ZERO_WIDTH:
        if char in out:
            out = out.replace(char, "")

    if collapse_whitespace:
        return " ".join(out.split())
    # Preserve line structure, but normalise runs of spaces/tabs within a line.
    return "\n".join(" ".join(line.split()) for line in out.splitlines())


def dehyphenate(left: str, right: str) -> tuple[str, bool]:
    """Rejoin a word split across a line break by justification.

    PDFs break `custo-\\nmer` and `conse-\\nquential` constantly; left unjoined,
    neither half matches anything and the chunk text reads as broken. Returns
    the possibly-modified left fragment and whether a join should happen without
    an intervening space.

    Deliberately conservative. A trailing hyphen is only treated as a break when
    the next line starts lowercase, which leaves genuine compounds alone:
    `self-\\nService` keeps its hyphen, and `well-known` on one line is never
    touched. A hyphenated proper noun split across lines is the acceptable loss.
    """
    if not left.endswith("-") or len(left) < 2:
        return left, False
    stem = left[:-1]
    if not stem or not stem[-1].isalpha():
        # An em-dash substitute or a numeric range like "40-" is not a word break.
        return left, False
    if not right or not right[0].isalpha() or not right[0].islower():
        return left, False
    return stem, True


__all__ = ["dehyphenate", "normalise_text"]
