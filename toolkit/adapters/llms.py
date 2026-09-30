"""LLM adapters.

`ScriptedLLM` is not only a test double. A deterministic, offline model is what
lets you build and debug a pipeline's control flow — the retries, the repair
loop, the branching — without paying for a token or waiting on a network. Get
the flow right against it, then swap in the real one.

`LiteLLMClient` is the production path. LiteLLM is the one piece of this that
should never be hand-rolled: provider auth, parameter naming, error shapes and
retry semantics all differ, and normalising them is a full-time job someone else
is already doing.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..core.errors import AdapterError, MissingDependency, RateLimited
from ..core.models import Completion, Message, Usage


def _estimate_tokens(text: str) -> int:
    """Rough token count for budgeting when a provider reports none.

    Deliberately crude: about four characters per token. It is a budget signal,
    not an invoice, and pulling in a tokenizer to improve it would defeat the
    zero-dependency property of this module.
    """
    return max(1, len(text) // 4)


class ScriptedLLM:
    """Deterministic offline model.

    Three modes, in order of precedence: a `responses` queue consumed in order,
    a `handler` callable for logic-dependent replies, or the default echo of the
    last user message. Records every call in `.calls` so a test can assert on
    what the pipeline actually sent — which is usually the thing that is wrong.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        handler: Callable[[Sequence[Message]], str] | None = None,
        model: str = "scripted",
    ) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self._model = model
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        if self._responses:
            text = self._responses.pop(0)
        elif self._handler is not None:
            text = self._handler(messages)
        else:
            user = [m for m in messages if m.role == "user"]
            text = user[-1].content if user else ""
        prompt_tokens = sum(_estimate_tokens(m.content) for m in messages)
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=_estimate_tokens(text),
                cost_usd=0.0,
            ),
            model=self._model,
            finish_reason="stop",
        )


class LiteLLMClient:
    """Any of ~100 providers through one interface.

    Rate-limit failures are translated to `RateLimited` so the governor layer
    can treat them as retryable without string-matching provider error text.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        completion_fn: Callable[..., Any] | None = None,
        **defaults: Any,
    ) -> None:
        self._model = model
        self._completion = completion_fn
        self._defaults = defaults

    def _ensure(self) -> Callable[..., Any]:
        if self._completion is None:
            try:
                from litellm import completion  # type: ignore
            except ImportError as exc:
                raise MissingDependency("litellm", "llm") from exc
            self._completion = completion
        return self._completion

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        completion = self._ensure()
        payload: dict[str, Any] = dict(self._defaults)
        payload.update(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
        )
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = completion(**payload)
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            name = type(exc).__name__.lower()
            if "ratelimit" in name or "quota" in name:
                raise RateLimited(str(exc)) from exc
            raise AdapterError("litellm call failed: " + str(exc)) from exc

        return Completion(
            text=self._text_of(response),
            usage=self._usage_of(response, messages),
            model=str(self._get(response, "model") or self._model),
            finish_reason=str(self._finish_of(response) or ""),
        )

    # LiteLLM returns an object that is usually dict-like and sometimes
    # attribute-like depending on the provider, so every read goes through this.
    def _get(self, obj: Any, key: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key)
        return getattr(obj, key, None)

    def _first_choice(self, response: Any) -> Any:
        choices = self._get(response, "choices") or []
        return choices[0] if choices else None

    def _text_of(self, response: Any) -> str:
        choice = self._first_choice(response)
        if choice is None:
            raise AdapterError("litellm returned no choices")
        message = self._get(choice, "message")
        content = self._get(message, "content") if message is not None else None
        if content is None:
            content = self._get(choice, "text")
        return str(content or "")

    def _finish_of(self, response: Any) -> Any:
        choice = self._first_choice(response)
        return self._get(choice, "finish_reason") if choice is not None else None

    def _usage_of(self, response: Any, messages: Sequence[Message]) -> Usage:
        usage = self._get(response, "usage")
        if usage is None:
            return Usage(
                input_tokens=sum(_estimate_tokens(m.content) for m in messages)
            )
        return Usage(
            input_tokens=int(self._get(usage, "prompt_tokens") or 0),
            output_tokens=int(self._get(usage, "completion_tokens") or 0),
            cost_usd=float(self._get(response, "response_cost") or 0.0),
        )


__all__ = ["LiteLLMClient", "ScriptedLLM"]
