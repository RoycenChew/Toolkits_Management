"""Credential and endpoint resolution: what to call, and with which key.

Separate from the client that does the calling, because the two fail for
different reasons and at different times. "I cannot find a key" is a setup
problem a human fixes once; "the provider returned 503" is a runtime problem a
retry handles. Collapsing them into one error type is why so much LLM code
reports a misconfiguration as a transient failure and retries it three times.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ApiStyle(str, Enum):
    """The request/response *shape* a provider speaks.

    The useful abstraction is two styles, not N providers. Almost every vendor
    worth calling exposes an OpenAI-compatible chat-completions endpoint, so
    supporting DeepSeek, Gemini, Groq, Mistral, Qwen, Kimi, GLM and a dozen
    self-hosted servers costs one adapter and a base URL - while Anthropic's
    Messages API is different enough to need its own.

    Two concrete shapes beat one leaky universal abstraction, and the day a
    third appears it is a third member here rather than a rewrite.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class SetupError(Exception):
    """The AI settings are missing or contradictory.

    Never raised for anything a retry could fix. If you see this, a human has
    to change a setting, so the message says which one and where to get it.
    """


@dataclass(frozen=True)
class Provider:
    """Everything needed to make a call, resolved from the environment.

    Frozen so a resolved provider cannot be mutated behind a client's back.
    """

    style: ApiStyle
    model: str
    label: str
    """What a human should see: "Anthropic", "DeepSeek", or an endpoint host."""
    base_url: str | None = None
    """`None` means the style's own official endpoint."""
    api_key: str = ""
    source: str = ""
    """Which setting supplied the key, e.g. `"ANTHROPIC_API_KEY"`. Names only."""

    def __repr__(self) -> str:
        """Redacted, deliberately.

        A dataclass's generated `repr` would print `api_key` in full, and the
        places a provider object ends up - a debug log line, an unhandled
        traceback, a crash reporter, a CI job's captured output - are all places
        a credential must never reach. One `logging.debug(provider)` is enough
        to leak a key into a log aggregator for its whole retention period, and
        that is not retrofittable: the key has to be rotated.

        So the only way to obtain the secret is to ask for it by name, which is
        greppable at review time.
        """
        return (
            "Provider(style=%s, model=%r, label=%r, base_url=%r, api_key=%s, source=%r)"
            % (
                self.style.value,
                self.model,
                self.label,
                self.base_url,
                self.redacted_key,
                self.source,
            )
        )

    __str__ = __repr__

    @property
    def redacted_key(self) -> str:
        """The key as it is safe to display: length and last four characters.

        Enough to tell two keys apart and to confirm which one is loaded, which
        is the only thing anyone actually needs from a log line.
        """
        if not self.api_key:
            return "<none>"
        if len(self.api_key) <= 8:
            return "<set:%d chars>" % len(self.api_key)
        return "<set:%d chars ...%s>" % (len(self.api_key), self.api_key[-4:])

    @property
    def endpoint(self) -> str:
        """The base URL to call, falling back to the style's official host."""
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.style is ApiStyle.ANTHROPIC:
            return "https://api.anthropic.com"
        return "https://api.openai.com/v1"

    def describe(self) -> str:
        """One line fit for a log or a `--version`-style banner."""
        return "%s %s via %s (key %s from %s)" % (
            self.label,
            self.model,
            self.endpoint,
            self.redacted_key,
            self.source or "explicit argument",
        )
