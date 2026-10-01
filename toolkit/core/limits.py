"""Resource limits for parsing untrusted input.

Every `DocumentSource` runs a parser over bytes it did not create. Without caps,
a decompression bomb, a 40,000-page PDF, or a file crafted to make a layout
parser quadratic will take the process down — and `skip_failed=True` does not
save you from an OOM kill, because the kill happens below the exception handler.

These defaults are deliberately generous enough that real documents pass and
tight enough that pathological ones do not. They are on by default, including in
the hackathon profile: the habit of trusting your own files is exactly how an
untrusted file eventually gets parsed unguarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import ToolkitError


class ScreeningFailure(str, Enum):
    """Why a document was refused. All of these are poison, never retryable."""

    TOO_LARGE = "too_large"
    TOO_MANY_PAGES = "too_many_pages"
    TOO_SLOW = "too_slow"
    UNREADABLE = "unreadable"
    ENCRYPTED = "encrypted"
    EMPTY = "empty"


class ScreeningRejected(ToolkitError):
    """Raised when a document fails screening and the caller asked to raise.

    Distinct from `AdapterError` because it is *never* worth retrying, and a
    pipeline that treats it as transient will loop on the same bad file forever.
    """

    def __init__(self, reason: ScreeningFailure, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__("screening rejected (" + reason.value + "): " + detail)


@dataclass(frozen=True)
class ScreeningLimits:
    max_bytes: int = 64 * 1024 * 1024
    """64 MB. A text document above this is almost always embedded media or a
    bomb; either way it is not what the pipeline is for."""

    max_pages: int = 2_000
    """Checked after the container is opened but before any page is processed,
    which is the only point where it is both known and still cheap to refuse."""

    max_seconds: float = 300.0
    """Wall-clock budget for one document, checked between pages.

    Deliberately not a hard interrupt: killing a parser mid-page needs a
    subprocess or signal handling, neither of which is portable or safe inside a
    library. A between-pages check cannot stop a single pathological page, and
    that limitation is stated rather than hidden. For genuinely hostile input,
    run the parser in a subprocess with an OS-level timeout."""

    min_bytes: int = 1
    """Reject empty files early rather than producing an empty Document."""

    def __post_init__(self) -> None:
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if self.max_pages < 1:
            raise ValueError("max_pages must be positive")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")

    @staticmethod
    def unlimited() -> ScreeningLimits:
        """Escape hatch for a trusted batch job. Explicit, so it shows up in a
        diff and in review rather than being the quiet default."""
        return ScreeningLimits(
            max_bytes=1 << 62, max_pages=1 << 30, max_seconds=float("inf")
        )


@dataclass(frozen=True)
class ScreeningResult:
    passed: bool
    reason: ScreeningFailure | None = None
    detail: str = ""
    size_bytes: int = 0
    page_count: int | None = None
    """None when the format carries no page concept, or when it was not opened."""

    def raise_if_rejected(self) -> None:
        if not self.passed and self.reason is not None:
            raise ScreeningRejected(self.reason, self.detail)


__all__ = [
    "ScreeningFailure",
    "ScreeningLimits",
    "ScreeningRejected",
    "ScreeningResult",
]
