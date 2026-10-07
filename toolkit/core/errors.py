"""The toolkit's error taxonomy.

One base class so a caller can catch everything from the toolkit without
catching their own bugs, and a small set of subclasses that distinguish the
failures a pipeline actually has to treat differently: a missing optional
dependency is a setup problem, a rate limit is worth retrying, a validation
failure is not.
"""
from __future__ import annotations


class ToolkitError(Exception):
    """Base class for every error raised by this toolkit."""


class MissingDependency(ToolkitError):
    """An optional backend was used without its package installed.

    Carries the exact install command, because the whole point of optional
    extras is lost if the error does not say how to satisfy them.
    """

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            "this adapter needs '"
            + package
            + "', which is not installed. Install it with:"
            + "  pip install 'toolkit["
            + extra
            + "]'"
        )


class AdapterError(ToolkitError):
    """A backend was reachable but failed or returned something unusable."""


class PermanentFailure(AdapterError):
    """A provider failure that will fail identically on a retry.

    A subclass of `AdapterError`, so every existing handler keeps working, and
    distinct so `governor` can stop retrying it. The measured case: a run died
    with "giving up after 3 attempts: DeepSeek API error 402: Insufficient
    Balance". An empty account does not fill itself between attempts, so two of
    those three round trips were waste, and the retry loop hid the one fact
    that mattered behind a generic message.

    Use it for the configuration failures: a rejected key, an unknown model, a
    malformed request, an exhausted balance. Anything that might succeed on a
    second attempt - a timeout, a reset connection, a 503 - stays an ordinary
    `AdapterError` and is still retried.
    """


class RateLimited(AdapterError):
    """The provider refused the call for rate or quota reasons.

    Distinguished from AdapterError because it is the one backend failure that
    is always worth retrying with backoff rather than surfacing.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class ValidationFailed(ToolkitError):
    """Output did not satisfy the requested schema or contract.

    Never retried blindly: retrying identical input against a deterministic
    validator produces the identical failure.
    """


__all__ = [
    "PermanentFailure",
    "ToolkitError",
    "MissingDependency",
    "AdapterError",
    "RateLimited",
    "ValidationFailed",
]
