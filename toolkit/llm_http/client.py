"""An `LLM` implementation over plain HTTP. No vendor SDK, no dependencies.

Two request shapes cover almost every provider worth calling: OpenAI's
chat-completions and Anthropic's messages. Everything else - DeepSeek, Gemini,
Groq, Mistral, Together, xAI, Qwen, Kimi, GLM, vLLM, Ollama, LM Studio - is the
OpenAI shape at a different base URL, which is why this is 2 code paths rather
than 12.

Why `urllib` rather than the official SDKs: this keeps the toolkit's base
install stdlib-only all the way through generation. Before this, the only way
to reach a real model was `LiteLLMClient`, so the whole generation half of the
pipeline was unreachable on a bare install. A chat completion is one POST with
a JSON body; an SDK is convenience, not capability.

What this deliberately does **not** do:

* **Retry.** `governor` already owns budgets, rate limiting and selective
  retry. This client's job is to *classify* a failure correctly - `RateLimited`
  for 429 and overload, `AdapterError` for everything else - so that the layer
  above can make the decision. Two components retrying the same call
  independently is how a rate limit becomes an outage.
* **Stream, call tools, or request structured output.** `ports.LLM` leaves
  those out on purpose, and a lowest-common-denominator version of any of them
  would be dishonest. Reach for the vendor SDK when you need them.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..core.errors import AdapterError, RateLimited
from ..core.models import Completion, Message, Usage
from ..provider import ApiStyle, Provider, resolve

#: A transport takes (url, headers, body, timeout) and returns (status, text).
#:
#: Injectable so every behaviour below is testable without a network or an API
#: key. The error translation *is* the valuable part of this unit, and error
#: translation that can only be exercised against a live provider is error
#: translation nobody tests.
Transport = Callable[[str, Mapping[str, str], dict, float], "tuple[int, str]"]

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})


def urllib_transport(
    url: str, headers: Mapping[str, str], body: dict, timeout: float
) -> tuple[int, str]:
    """The default transport. Returns the status even for an error response.

    `HTTPError` is caught and its body returned rather than raised, because the
    body carries the provider's own explanation and discarding it is how "HTTP
    400" becomes an unactionable error message.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **dict(headers)},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        raise AdapterError(
            "could not reach %s: %s" % (_host(url), exc.reason)
        ) from exc
    except TimeoutError as exc:
        raise AdapterError("timed out calling %s" % _host(url)) from exc


def _host(url: str) -> str:
    return url.split("/")[2] if "//" in url else url


def _provider_message(payload: Any) -> str:
    """Pull the human-readable reason out of an error body of unknown shape."""
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            for field in ("message", "detail", "type"):
                value = error.get(field)
                if isinstance(value, str) and value:
                    return value
        if isinstance(error, str) and error:
            return error
        for field in ("message", "detail"):
            value = payload.get(field)
            if isinstance(value, str) and value:
                return value
    return ""


class HttpLLM:
    """`ports.LLM` over HTTP for OpenAI-compatible and Anthropic-style APIs.

    Usage counts come from the provider's own `usage` block, so `governor`'s
    budget is as honest as the provider is. When a provider omits it the counts
    are zero and the governor will under-count - stated here rather than
    silently estimated, because a made-up token count is worse than a missing
    one.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        timeout: float = 120.0,
        max_output_tokens: int = 4096,
        transport: Transport | None = None,
    ) -> None:
        self.provider = provider if provider is not None else resolve()
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self._transport: Transport = transport or urllib_transport

    @property
    def model_version(self) -> str:
        return "%s:%s" % (self.provider.style.value, self.provider.model)

    def __repr__(self) -> str:
        # Delegates to Provider.__repr__, which redacts the key.
        return "HttpLLM(%r, timeout=%s)" % (self.provider, self.timeout)

    # -- the port ----------------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        if not messages:
            raise AdapterError("complete() needs at least one message")
        limit = max_tokens or self.max_output_tokens
        if self.provider.style is ApiStyle.ANTHROPIC:
            url, headers, body = self._anthropic_request(messages, temperature, limit)
        else:
            url, headers, body = self._openai_request(messages, temperature, limit)

        status, text = self._transport(url, headers, body, self.timeout)
        try:
            payload = json.loads(text) if text.strip() else {}
        except ValueError:
            payload = {}

        if status >= 400:
            self._raise_for_status(status, payload, text)

        if self.provider.style is ApiStyle.ANTHROPIC:
            return self._anthropic_completion(payload)
        return self._openai_completion(payload)

    # -- request shaping ---------------------------------------------------
    def _openai_request(
        self, messages: Sequence[Message], temperature: float, limit: int
    ) -> tuple[str, dict[str, str], dict]:
        return (
            self.provider.endpoint + "/chat/completions",
            {"Authorization": "Bearer " + self.provider.api_key},
            {
                "model": self.provider.model,
                "messages": [
                    {"role": m.role, "content": self._openai_content(m)}
                    for m in messages
                ],
                "temperature": temperature,
                "max_tokens": limit,
            },
        )

    @staticmethod
    def _openai_content(message: Message) -> Any:
        """A plain string for a text-only turn, a parts list when there is an image.

        The string case is not an optimisation. This style accepts a parts list
        for text too, but switching every existing call to one would change the
        request body of every pipeline already running against it, to no end.
        """
        if not message.images:
            return message.content
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"type": "text", "text": message.content})
        parts.extend(
            {"type": "image_url", "image_url": {"url": image.data_url}}
            for image in message.images
        )
        return parts

    def _anthropic_request(
        self, messages: Sequence[Message], temperature: float, limit: int
    ) -> tuple[str, dict[str, str], dict]:
        """Anthropic takes the system prompt as a top-level field, not a message.

        Sending `role: "system"` inside `messages` is rejected, so the system
        turns are lifted out and joined. Getting this wrong produces a 400 whose
        message does not obviously point at the cause.
        """
        for message in messages:
            if message.role == "system" and message.images:
                # The system prompt is a top-level string in this API, so an
                # image on it has nowhere to go. Dropping it quietly would send
                # a request that cannot answer the question it was asked.
                raise AdapterError(
                    "an Anthropic system prompt cannot carry images; put them on"
                    " a user message"
                )
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": self._anthropic_content(m)}
            for m in messages
            if m.role != "system"
        ]
        if not turns:
            raise AdapterError(
                "an Anthropic request needs at least one user message, not only a system prompt"
            )
        body: dict[str, Any] = {
            "model": self.provider.model,
            "messages": turns,
            "max_tokens": limit,
            "temperature": temperature,
        }
        if system:
            body["system"] = system
        return (
            self.provider.endpoint + "/v1/messages",
            {
                "x-api-key": self.provider.api_key,
                "anthropic-version": "2023-06-01",
            },
            body,
        )

    @staticmethod
    def _anthropic_content(message: Message) -> Any:
        """Image blocks first, then the text.

        That order is what Anthropic documents and uses in every example, and
        it is not cosmetic: a question asked before the image it refers to
        measurably degrades the answer.
        """
        if not message.images:
            return message.content
        blocks: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": image.base64,
                },
            }
            for image in message.images
        ]
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        return blocks

    # -- failure classification -------------------------------------------
    def _raise_for_status(self, status: int, payload: Any, raw: str) -> None:
        """Translate a status code into the port's documented exceptions.

        Messages name the provider and say what to change, because the common
        failures here are all configuration: a key from the wrong account, a
        model name that does not exist on this endpoint, a base URL copied with
        a trailing path. "HTTP 404" sends people to the wrong place.
        """
        body = raw.strip()
        # An empty JSON object carries no information, so do not append "{}" to
        # an otherwise clean message.
        detail = _provider_message(payload) or (
            body[:300] if body not in ("", "{}", "[]", "null") else ""
        )
        label = self.provider.label

        if status in (401, 403):
            raise AdapterError(
                "%s rejected the API key (HTTP %d). The key in %s may be invalid, "
                "revoked, or for a different account.%s"
                % (label, status, self.provider.source or "the environment",
                   " " + detail if detail else "")
            )
        if status == 404:
            raise AdapterError(
                "%s has no model %r at %s (HTTP 404). Check the model name and the "
                "base URL.%s"
                % (label, self.provider.model, self.provider.endpoint,
                   " " + detail if detail else "")
            )
        if status == 400:
            raise AdapterError(
                "%s rejected the request (HTTP 400): %s" % (label, detail or "no detail given")
            )
        if status in _RETRYABLE_STATUS:
            raise RateLimited(
                "%s is rate limited or overloaded (HTTP %d)%s"
                % (label, status, ": " + detail if detail else ""),
                retry_after=None,
            )
        raise AdapterError("%s API error %d%s" % (label, status, ": " + detail if detail else ""))

    # -- response parsing --------------------------------------------------
    def _openai_completion(self, payload: Any) -> Completion:
        choices = payload.get("choices") if isinstance(payload, Mapping) else None
        if not choices:
            raise AdapterError(
                "%s returned no choices; the response carried no completion"
                % self.provider.label
            )
        first = choices[0] or {}
        finish = str(first.get("finish_reason") or "")
        text = str((first.get("message") or {}).get("content") or "")

        # A truncated answer that is returned as a success is the quietest way
        # for a pipeline to produce a wrong result, so it is an error here.
        if finish == "length" and not text.strip():
            raise AdapterError(
                "%s truncated the answer before any text was produced "
                "(finish_reason=length); raise max_tokens" % self.provider.label
            )
        if finish == "content_filter":
            raise AdapterError("%s filtered the response (content_filter)" % self.provider.label)
        if not text.strip():
            raise AdapterError(
                "%s returned an empty completion (finish_reason=%r)"
                % (self.provider.label, finish or "unset")
            )
        usage = payload.get("usage") or {}
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
            model=str(payload.get("model") or self.provider.model),
            finish_reason=finish,
        )

    def _anthropic_completion(self, payload: Any) -> Completion:
        if not isinstance(payload, Mapping):
            raise AdapterError("%s returned a malformed response" % self.provider.label)
        stop = str(payload.get("stop_reason") or "")
        if stop == "refusal":
            raise AdapterError("%s declined this request (stop_reason=refusal)" % self.provider.label)
        blocks = payload.get("content") or []
        text = "".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        if stop == "max_tokens" and not text.strip():
            raise AdapterError(
                "%s truncated the answer before any text was produced "
                "(stop_reason=max_tokens); raise max_tokens" % self.provider.label
            )
        if not text.strip():
            raise AdapterError(
                "%s returned an empty completion (stop_reason=%r)"
                % (self.provider.label, stop or "unset")
            )
        usage = payload.get("usage") or {}
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            ),
            model=str(payload.get("model") or self.provider.model),
            finish_reason=stop,
        )
