# HTTP LLM Component

**Layer 2 · depends on `core`, `provider` · `copy_tier: needs_package`**

## What It Does

Implements the `LLM` port over plain HTTP, for OpenAI-compatible and
Anthropic-style APIs. No vendor SDK, no third-party package.

## Why It Is Useful

Before this, the only route to a real model was `LiteLLMClient`, so the entire
generation half of the pipeline was unreachable on the stdlib-only base
install. A chat completion is one POST with a JSON body; an SDK is convenience,
not capability. **This makes the toolkit end-to-end usable with nothing
installed.**

Two request shapes cover almost every provider worth calling — DeepSeek,
Gemini, Groq, Mistral, Together, xAI, Qwen, Kimi, GLM, vLLM, Ollama, LM Studio
are all the OpenAI shape at a different base URL — which is why this is 2 code
paths rather than 12.

## The division of labour that matters

**This client classifies failures. It never retries them.**

`governor` already owns budgets, rate limiting and selective retry. The client's
job is to decide *correctly* whether a failure is worth retrying — `RateLimited`
for 429 and overload, `AdapterError` for everything else — so the layer above
can act. Two components retrying the same call independently is how a rate
limit becomes an outage.

The second thing it does is refuse to return a quietly wrong answer. A
truncated completion handed back as a success is the most silent way for a
pipeline to produce a wrong result, so an empty answer with
`finish_reason=length` is an error naming `max_tokens`, an Anthropic
`stop_reason=refusal` is an error, and a `content_filter` is an error. A
*partial* answer is returned intact with the reason on the `Completion`, because
that is usable.

## Architecture

```
Message[]  ──▶  style?
                 │
   ┌─────────────┴──────────────┐
   ▼                            ▼
OPENAI                       ANTHROPIC
POST {base}/chat/completions  POST {base}/v1/messages
Authorization: Bearer <key>   x-api-key, anthropic-version
system stays in messages      system LIFTED OUT of messages
images -> image_url parts     images -> image blocks, BEFORE the text
  (base64 data URL)             (base64 source + media_type)
   └─────────────┬──────────────┘
                 ▼
          transport(url, headers, body, timeout) -> (status, text)
                 ▼
        status >= 400 ?  ──▶  401/403 names the env var that supplied the key
                              404    names the model and the endpoint
                              400    carries the provider's own message
                              429/5xx -> RateLimited, for `governor` to retry
                 ▼
        finish_reason / stop_reason sanity
                 ▼
        Completion(text, Usage(provider-reported), model, finish_reason)
```

Anthropic takes the system prompt as a **top-level field**, not a message.
Sending `role: "system"` inside `messages` is rejected with a 400 whose text
does not obviously point at the cause, so the lifting happens here rather than
in every caller.

## Installation

```bash
pip install -e .
```

Standard library only, but it depends on `core` for the error taxonomy and
contracts and on `provider` for resolution, so install the package rather than
copying the directory. `toolkit/provider` on its own **is** standalone.

## Input / Output

```python
from toolkit.llm_http import HttpLLM
from toolkit.provider import resolve

llm = HttpLLM(resolve())                      # or HttpLLM() to resolve implicitly
llm = HttpLLM(resolve(), timeout=60.0, max_output_tokens=2048)
completion = llm.complete([Message(role="user", content="...")], temperature=0.0)
```

| Raises | When |
|---|---|
| `RateLimited` | 429, 500, 502, 503, 504, 529 — the retryable set |
| `AdapterError` | auth, unknown model, bad request, unreachable host, timeout, empty or filtered or refused completion |

`Completion.usage` carries the **provider's own** token counts. When a provider
omits the `usage` block the counts are zero, and that is reported rather than
estimated — `governor`'s budget is only as honest as this field, and a made-up
token count is worse than a missing one.

## Testability is a design feature

The constructor takes an injectable `transport`:

```python
def fake(url, headers, body, timeout):
    return 429, '{"error": {"message": "slow down"}}'

HttpLLM(provider, transport=fake)   # asserts RateLimited, offline
```

The error translation **is** the valuable part of this unit, and error handling
that can only be exercised against a live provider is error handling nobody
tests. All 58 tests for this unit, `provider` and the image shapes run offline
with no key.

## Sending an image

A scanned page is sometimes better read by a vision model than by OCR.
`Message.images` is a sequence of `core.ImagePart` (bytes plus a media type),
empty by default, and the two styles serialise it differently:

```python
from toolkit.core import ImagePart, Message

page = ImagePart(png_bytes, media_type="image/png")
llm.complete([Message("user", "What is the invoice total?", images=[page])])
```

- **OpenAI style** sends `content` as a list: a `text` part, then one
  `image_url` part per image holding a base64 **data URL**.
- **Anthropic style** sends `image` blocks with a base64 `source` and the media
  type as its own field, **before** the text. That order is what Anthropic
  documents and uses, and it is not cosmetic: a question asked before the image
  it refers to measurably degrades the answer.
- **A text-only message still sends `content` as a plain string.** Both styles
  accept a parts list for text, but switching every existing call to one would
  change the request body of every pipeline already running, to no end.
- An image on an **Anthropic system prompt** raises `AdapterError`. The system
  prompt is a top-level string in that API, so the image has nowhere to go, and
  dropping it quietly would send a request that cannot answer the question it
  was asked.

`CachedLLM` keys on each image's `fingerprint()`, so two requests differing only
in the image do not serve each other's answers.

## Limitations

- **No streaming, tool calling or structured output.** `ports.LLM` leaves these
  out deliberately — they differ enough between providers that a
  lowest-common-denominator version would be dishonest. Use the vendor SDK.
- **No retry and no backoff**, by design. Wrap in `GovernedLLM`.
- **No `retry_after`.** The transport returns a status and a body, not headers,
  so a provider's `Retry-After` is not surfaced; `governor` falls back to its
  own backoff.
- **No prices ship with the toolkit**, deliberately: a table that goes stale in a
  library nobody updates weekly is worse than none, because a stale number is
  believed. `Usage.cost_usd` therefore stays 0.0 **unless you pass a
  `provider.PriceBook`**, which is the rate per million tokens you already know,
  with an optional off-peak window. The omission used to be load-bearing in a bad
  way: `GovernedLLM`'s `max_cost_usd` is checked against that field, so a ceiling
  on an unpriced adapter never fired. `governor` now refuses that configuration
  rather than pretending — see `require_priced_calls`.
- **Two styles only**, and a provider that is only *nearly* OpenAI-compatible
  may still need its own branch.
- **Images are sent, not checked.** There is no size limit, no re-encoding and
  no format conversion here: the bytes go as they are. Every provider caps
  image size and resolution, and exceeding it is a 400 carrying the provider's
  own message. Resize before sending.
- **Images are base64 in the request body**, which inflates them by a third and
  is held in memory whole. A 30-page scan is a request measured in megabytes;
  send pages one at a time.
- **Synchronous.** One call, one thread. Use `concurrency.bounded_map` for
  fan-out.
- **Gemini via the OpenAI-compatible endpoint**, not its native API, so
  Gemini-specific features are unavailable.

## Integration Guide

1. **Wrap it**: `CachedLLM(GovernedLLM(HttpLLM(resolve()), config), cache)`.
   This unit is the innermost layer on purpose — caching and budgeting are
   someone else's job.
2. **Set `max_output_tokens` to what you actually need.** It is the default for
   every call, and an empty truncated answer is an error rather than a silent
   shrug.
3. **Let `AdapterError` reach the user.** The messages name the setting, the
   model and the endpoint because those are the three things that are actually
   wrong in practice.
4. **Inject a transport in tests.** Never point a test suite at a live
   provider.
5. For a self-hosted server (vLLM, Ollama, LM Studio), set
   `TOOLKIT_BASE_URL` and `TOOLKIT_MODEL`; the OpenAI style is almost always
   the right one.

## Extraction Notes

- **Preserved** from PRism (`prism.py`, same author): the two-shape split, the
  stdlib `urllib` POST, mapping vendor failures onto human-readable actionable
  messages, and treating `stop_reason` of `refusal`/`max_tokens` as errors
  rather than as text.
- **Removed:** the Anthropic SDK dependency and its server-side fallback beta
  (SDK-specific), and the project's own `ModelError`, replaced by the toolkit's
  `AdapterError`/`RateLimited` split so `governor` can already act on it.
- **Added:** the injectable transport, so every failure path is testable
  offline; the `RateLimited`-vs-`AdapterError` classification instead of one
  error type; system-prompt lifting for Anthropic; auth errors that name the
  environment variable the key came from; usage parsing into `core.Usage`; and
  redaction of the key from every error message and `repr`.
- **Isolated:** no vendor SDK, so nothing here can break a stdlib-only install.
