"""`provider` and `llm_http`: find a model, call it, classify the failure.

Extracted from a working project (PRism) where the provider-detection and
error-translation patterns had already been used in anger, then reimplemented
against this toolkit's error taxonomy and `ports.LLM` contract.

Every test here runs offline. `HttpLLM` takes an injectable transport
specifically so the error translation - the valuable part - is testable without
a network or a key, because error handling that can only be exercised against a
live provider is error handling nobody tests.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toolkit.core import Message  # noqa: E402
from toolkit.core.errors import AdapterError, RateLimited  # noqa: E402
from toolkit.llm_http import HttpLLM  # noqa: E402
from toolkit.ports import LLM  # noqa: E402
from toolkit.provider import (  # noqa: E402
    ApiStyle,
    Provider,
    SetupError,
    available,
    diagnose,
    load_settings,
    resolve,
)

SECRET = "sk-super-secret-key-9999"


def _transport(status: int, payload: dict):
    """A fake transport that records what it was asked to send."""

    def transport(url, headers, body, timeout):
        transport.seen = {"url": url, "headers": dict(headers), "body": body}
        return status, json.dumps(payload)

    transport.seen = {}
    return transport


def _openai_ok(text: str = "42", finish: str = "stop") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 2},
        "model": "deepseek-chat",
    }


def _deepseek() -> Provider:
    return resolve(env_file=None, environ={"DEEPSEEK_API_KEY": SECRET})


def _claude() -> Provider:
    return resolve(env_file=None, environ={"ANTHROPIC_API_KEY": SECRET})


# --------------------------------------------------------------------------- #
# provider: resolution
# --------------------------------------------------------------------------- #
def test_a_single_shortcut_key_is_enough() -> None:
    """A key with no model is still unusable, so a shortcut carries one."""
    provider = _deepseek()
    assert provider.style is ApiStyle.OPENAI
    assert provider.model == "deepseek-flash"
    assert provider.endpoint == "https://api.deepseek.com"
    assert provider.source == "DEEPSEEK_API_KEY"


def test_shortcut_order_is_the_precedence() -> None:
    """Two keys set is normal on a developer machine; the order must be stable."""
    both = resolve(env_file=None, environ={"ANTHROPIC_API_KEY": SECRET, "OPENAI_API_KEY": SECRET})
    assert both.label == "Anthropic", "SHORTCUTS order decides, and Anthropic is first"


def test_explicit_settings_beat_every_shortcut() -> None:
    provider = resolve(
        env_file=None,
        environ={
            "DEEPSEEK_API_KEY": SECRET,
            "TOOLKIT_API_KEY": "sk-explicit",
            "TOOLKIT_BASE_URL": "https://my-gateway.internal/v1",
            "TOOLKIT_MODEL": "house-model",
        },
    )
    assert provider.endpoint == "https://my-gateway.internal/v1"
    assert provider.model == "house-model"
    assert provider.label == "my-gateway.internal"


def test_a_provider_base_url_override_keeps_the_other_defaults() -> None:
    """How a proxy or a regional endpoint is used without losing the defaults."""
    provider = resolve(
        env_file=None,
        environ={"OPENAI_API_KEY": SECRET, "OPENAI_BASE_URL": "https://proxy.local/v1"},
    )
    assert provider.endpoint == "https://proxy.local/v1"
    assert provider.model == "gpt-4.1"


def test_the_environment_beats_the_dotfile(tmp_path) -> None:
    """CI injects variables and must win, or every deploy needs the file deleted."""
    env_file = os.path.join(str(tmp_path), ".toolkit.env")
    with open(env_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# a comment\nDEEPSEEK_API_KEY=\"from-file\"\n")

    from_file = load_settings(env_file=env_file, environ={})
    assert from_file["DEEPSEEK_API_KEY"] == "from-file"

    overridden = load_settings(env_file=env_file, environ={"DEEPSEEK_API_KEY": "from-env"})
    assert overridden["DEEPSEEK_API_KEY"] == "from-env"


def test_a_blank_value_does_not_unset_a_real_one(tmp_path) -> None:
    """`KEY=` in a shell profile is a common accident; it must not win."""
    env_file = os.path.join(str(tmp_path), ".toolkit.env")
    with open(env_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("DEEPSEEK_API_KEY=real-key\n")
    settings = load_settings(env_file=env_file, environ={"DEEPSEEK_API_KEY": ""})
    assert settings["DEEPSEEK_API_KEY"] == "real-key"


# --------------------------------------------------------------------------- #
# provider: the failure path is the feature
# --------------------------------------------------------------------------- #
def test_no_configuration_raises_a_setup_error_that_says_what_to_set() -> None:
    with pytest.raises(SetupError) as caught:
        resolve(env_file=None, environ={})
    message = str(caught.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "DEEPSEEK_API_KEY" in message
    assert "TOOLKIT_BASE_URL" in message


def test_a_key_without_an_endpoint_is_named_precisely() -> None:
    """The single most common misconfiguration deserves its own message."""
    with pytest.raises(SetupError) as caught:
        resolve(env_file=None, environ={"TOOLKIT_API_KEY": SECRET})
    assert "TOOLKIT_BASE_URL is missing" in str(caught.value)


def test_a_custom_endpoint_refuses_to_guess_a_model() -> None:
    with pytest.raises(SetupError) as caught:
        resolve(
            env_file=None,
            environ={"TOOLKIT_API_KEY": SECRET, "TOOLKIT_BASE_URL": "https://x/v1"},
        )
    assert "TOOLKIT_MODEL is missing" in str(caught.value)


def test_an_unknown_style_is_rejected_not_defaulted() -> None:
    with pytest.raises(SetupError) as caught:
        resolve(
            env_file=None,
            environ={
                "TOOLKIT_API_KEY": SECRET,
                "TOOLKIT_BASE_URL": "https://x/v1",
                "TOOLKIT_API_STYLE": "cohere",
            },
        )
    assert "'openai' or 'anthropic'" in str(caught.value)


def test_diagnose_reports_names_and_never_values() -> None:
    """A diagnosis that leaked the values would be worse than no diagnosis."""
    report = diagnose({"DEEPSEEK_API_KEY": SECRET, "TOOLKIT_BASE_URL": "https://x/v1"})
    rendered = report.render()
    assert "DEEPSEEK_API_KEY" in report.present
    assert SECRET not in rendered
    assert "https://x/v1" not in rendered


def test_available_lets_a_caller_degrade_instead_of_failing() -> None:
    assert available(env_file=None, environ={"DEEPSEEK_API_KEY": SECRET}) is True
    assert available(env_file=None, environ={}) is False


# --------------------------------------------------------------------------- #
# provider: the key must not leak
# --------------------------------------------------------------------------- #
def test_repr_and_str_redact_the_key() -> None:
    """One `logging.debug(provider)` would otherwise put a key in a log forever.

    A dataclass's generated repr prints every field, so this is overridden
    rather than left to discipline.
    """
    provider = _deepseek()
    for rendered in (repr(provider), str(provider), provider.describe()):
        assert SECRET not in rendered, rendered
    assert provider.redacted_key.endswith("9999>")
    assert provider.api_key == SECRET, "the real key must still be reachable by name"


def test_a_formatted_container_also_redacts() -> None:
    """`%s` on a list calls repr on the members; that path must be safe too."""
    assert SECRET not in "%s" % ([_deepseek()],)


def test_the_client_repr_redacts_too() -> None:
    assert SECRET not in repr(HttpLLM(_deepseek(), transport=_transport(200, _openai_ok())))


# --------------------------------------------------------------------------- #
# llm_http: request shaping
# --------------------------------------------------------------------------- #
def test_it_satisfies_the_llm_port() -> None:
    assert isinstance(HttpLLM(_deepseek(), transport=_transport(200, _openai_ok())), LLM)


def test_openai_style_posts_chat_completions_with_a_bearer_token() -> None:
    transport = _transport(200, _openai_ok())
    completion = HttpLLM(_deepseek(), transport=transport).complete(
        [Message(role="user", content="what is 6*7?")]
    )

    assert completion.text == "42"
    assert completion.usage.input_tokens == 11
    assert completion.usage.output_tokens == 2
    assert completion.finish_reason == "stop"
    assert transport.seen["url"] == "https://api.deepseek.com/chat/completions"
    assert transport.seen["headers"]["Authorization"] == "Bearer " + SECRET


def test_anthropic_style_lifts_the_system_prompt_out_of_messages() -> None:
    """Anthropic rejects `role: system` inside `messages`.

    Sending it anyway produces a 400 whose text does not obviously point at the
    cause, so the lifting is done here rather than left to every caller.
    """
    payload = {
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
        "model": "claude-opus-5",
    }
    transport = _transport(200, payload)
    HttpLLM(_claude(), transport=transport).complete(
        [Message(role="system", content="Be brief."), Message(role="user", content="hi")]
    )

    assert transport.seen["url"] == "https://api.anthropic.com/v1/messages"
    assert transport.seen["body"]["system"] == "Be brief."
    assert [m["role"] for m in transport.seen["body"]["messages"]] == ["user"]
    assert transport.seen["headers"]["x-api-key"] == SECRET
    assert "Authorization" not in transport.seen["headers"]


def test_a_system_only_conversation_is_refused_for_anthropic() -> None:
    transport = _transport(200, {})
    with pytest.raises(AdapterError) as caught:
        HttpLLM(_claude(), transport=transport).complete(
            [Message(role="system", content="Be brief.")]
        )
    assert "at least one user message" in str(caught.value)


def test_no_messages_is_a_programming_error() -> None:
    with pytest.raises(AdapterError):
        HttpLLM(_deepseek(), transport=_transport(200, _openai_ok())).complete([])


# --------------------------------------------------------------------------- #
# llm_http: failure classification, which is the point of the unit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_transient_statuses_raise_rate_limited_so_the_governor_can_retry(status: int) -> None:
    """Classification here, retry in `governor`.

    Two components retrying the same call independently is how a rate limit
    becomes an outage, so this client never retries - it only says whether a
    retry is worth attempting.
    """
    with pytest.raises(RateLimited):
        HttpLLM(_deepseek(), transport=_transport(status, {})).complete(
            [Message(role="user", content="x")]
        )


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_configuration_failures_are_not_retryable(status: int) -> None:
    """A wrong key is not a transient condition; retrying it wastes a budget."""
    with pytest.raises(AdapterError) as caught:
        HttpLLM(_deepseek(), transport=_transport(status, {})).complete(
            [Message(role="user", content="x")]
        )
    assert not isinstance(caught.value, RateLimited)


def test_an_auth_failure_names_the_setting_that_supplied_the_key() -> None:
    """"HTTP 401" sends people to the wrong place; the variable name does not."""
    with pytest.raises(AdapterError) as caught:
        HttpLLM(_deepseek(), transport=_transport(401, {"error": {"message": "bad key"}})).complete(
            [Message(role="user", content="x")]
        )
    message = str(caught.value)
    assert "DEEPSEEK_API_KEY" in message
    assert "DeepSeek" in message
    assert SECRET not in message, "an error message must never echo the key"


def test_a_404_names_the_model_and_the_endpoint() -> None:
    with pytest.raises(AdapterError) as caught:
        HttpLLM(_deepseek(), transport=_transport(404, {})).complete(
            [Message(role="user", content="x")]
        )
    message = str(caught.value)
    assert "deepseek-flash" in message
    assert "https://api.deepseek.com" in message


def test_the_providers_own_explanation_is_preserved() -> None:
    """Discarding the response body is how "HTTP 400" becomes unactionable."""
    with pytest.raises(AdapterError) as caught:
        HttpLLM(
            _deepseek(),
            transport=_transport(400, {"error": {"message": "max_tokens must be <= 8192"}}),
        ).complete([Message(role="user", content="x")])
    assert "max_tokens must be <= 8192" in str(caught.value)


def test_an_empty_error_body_does_not_append_braces() -> None:
    with pytest.raises(RateLimited) as caught:
        HttpLLM(_deepseek(), transport=_transport(503, {})).complete(
            [Message(role="user", content="x")]
        )
    assert "{}" not in str(caught.value)


# --------------------------------------------------------------------------- #
# llm_http: a quietly wrong answer is worse than an error
# --------------------------------------------------------------------------- #
def test_truncation_with_no_text_is_an_error_not_an_empty_success() -> None:
    """A truncated answer returned as success is the quietest way to be wrong."""
    with pytest.raises(AdapterError) as caught:
        HttpLLM(_deepseek(), transport=_transport(200, _openai_ok("", "length"))).complete(
            [Message(role="user", content="x")]
        )
    assert "max_tokens" in str(caught.value)


def test_a_content_filter_is_reported() -> None:
    with pytest.raises(AdapterError) as caught:
        HttpLLM(_deepseek(), transport=_transport(200, _openai_ok("", "content_filter"))).complete(
            [Message(role="user", content="x")]
        )
    assert "filtered" in str(caught.value)


def test_no_choices_is_an_error() -> None:
    with pytest.raises(AdapterError):
        HttpLLM(_deepseek(), transport=_transport(200, {"choices": []})).complete(
            [Message(role="user", content="x")]
        )


def test_an_anthropic_refusal_is_an_error() -> None:
    payload = {"content": [], "stop_reason": "refusal"}
    with pytest.raises(AdapterError) as caught:
        HttpLLM(_claude(), transport=_transport(200, payload)).complete(
            [Message(role="user", content="x")]
        )
    assert "declined" in str(caught.value)


def test_a_partial_answer_is_returned_rather_than_discarded() -> None:
    """Truncated *with* text is a usable result; the reason is on the Completion."""
    completion = HttpLLM(
        _deepseek(), transport=_transport(200, _openai_ok("partial answer", "length"))
    ).complete([Message(role="user", content="x")])

    assert completion.text == "partial answer"
    assert completion.finish_reason == "length"


def test_missing_usage_reports_zero_rather_than_a_guess() -> None:
    """A made-up token count is worse than a missing one: `governor` trusts it."""
    payload = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
    completion = HttpLLM(_deepseek(), transport=_transport(200, payload)).complete(
        [Message(role="user", content="x")]
    )
    assert completion.usage.input_tokens == 0
    assert completion.usage.output_tokens == 0


# --------------------------------------------------------------------------- #
# composition with what already exists
# --------------------------------------------------------------------------- #
def test_it_composes_with_the_governor() -> None:
    """The division of labour: this client classifies, `governor` decides."""
    from toolkit.governor import GovernedLLM, GovernorConfig

    governed = GovernedLLM(
        HttpLLM(_deepseek(), transport=_transport(200, _openai_ok())),
        GovernorConfig(max_total_tokens=1000),
    )
    completion = governed.complete([Message(role="user", content="what is 6*7?")])
    assert completion.text == "42"


def test_it_composes_with_the_cache() -> None:
    from toolkit.cache import CachedLLM, SqliteCache

    transport = _transport(200, _openai_ok())
    cached = CachedLLM(HttpLLM(_deepseek(), transport=transport), SqliteCache())
    messages = [Message(role="user", content="what is 6*7?")]

    first = cached.complete(messages)
    second = cached.complete(messages)

    assert first.text == second.text == "42"
    assert second.usage.cached is True
