"""Model identity in caches and provider defaults (TK-5).

Two defects, both found while planning the FDIP document pipeline:

* `CachedLLM` keyed its entries on the wrapped object's *class name* unless a
  `model_name` was passed. Every `HttpLLM` has the same class name, so two
  different models behind one cache silently shared answers: switching from a
  cheap model to a strong one returned the cheap model's cached output, with no
  error anywhere. `CachedEmbedder` already delegated to `model_version`; the LLM
  wrapper now does the same.
* The DeepSeek shortcut defaulted to `deepseek-chat`, a model name the provider's
  current pricing page no longer lists. A shortcut default that is stale turns
  "I set the key" into a 404.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toolkit.adapters import ScriptedLLM  # noqa: E402
from toolkit.cache import CachedLLM, SqliteCache  # noqa: E402
from toolkit.core import Message  # noqa: E402
from toolkit.llm_http import HttpLLM  # noqa: E402
from toolkit.provider import resolve  # noqa: E402


def _replying(text: str):
    def transport(url, headers, body, timeout):
        transport.calls += 1
        return 200, json.dumps({
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })

    transport.calls = 0
    return transport


def _provider(model: str):
    return resolve(env_file=None, environ={"DEEPSEEK_API_KEY": "sk-test-123456", "TOOLKIT_MODEL": model})


def test_two_models_behind_one_cache_do_not_share_answers() -> None:
    cache = SqliteCache()
    question = [Message("user", "what is the total?")]
    cheap = CachedLLM(HttpLLM(_provider("model-a"), transport=_replying("A")), cache)
    strong_transport = _replying("B")
    strong = CachedLLM(HttpLLM(_provider("model-b"), transport=strong_transport), cache)

    assert cheap.complete(question).text == "A"
    answer = strong.complete(question)

    assert answer.text == "B", "model-b was served model-a's cached answer"
    assert strong_transport.calls == 1


def test_the_same_model_still_hits_the_cache() -> None:
    cache = SqliteCache()
    question = [Message("user", "q")]
    transport = _replying("A")
    first = CachedLLM(HttpLLM(_provider("model-a"), transport=transport), cache)
    second = CachedLLM(HttpLLM(_provider("model-a"), transport=transport), cache)

    first.complete(question)
    again = second.complete(question)

    assert again.usage.cached and transport.calls == 1


def test_cached_llm_reports_the_wrapped_models_identity() -> None:
    llm = HttpLLM(_provider("model-a"), transport=_replying("A"))
    wrapped = CachedLLM(llm, SqliteCache())
    assert wrapped.model_version == llm.model_version == "openai:model-a"


def test_scripted_llm_has_a_model_version() -> None:
    """The offline model needs an identity too, or two scripted models with
    different scripts would collide in a shared cache exactly like above."""
    assert ScriptedLLM(model="fixture-1").model_version == "scripted:fixture-1"
    assert ScriptedLLM(model="a").model_version != ScriptedLLM(model="b").model_version


def test_an_explicit_model_name_still_wins() -> None:
    llm = HttpLLM(_provider("model-a"), transport=_replying("A"))
    assert CachedLLM(llm, SqliteCache(), model_name="pinned").model_version == "pinned"


def test_deepseek_shortcut_uses_a_current_model_name() -> None:
    provider = resolve(env_file=None, environ={"DEEPSEEK_API_KEY": "sk-test-123456"})
    assert provider.model == "deepseek-flash"
