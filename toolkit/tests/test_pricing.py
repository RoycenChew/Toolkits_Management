"""Model pricing, and the guard that stops a cost ceiling being decorative.

`Usage.cost_usd` has existed since the first commit and nothing ever filled it
in. `llm_http` listed that as a limitation and was right about the reason -
prices change weekly and a stale table shipped inside a library is worse than
an absent one - but the consequence was never followed through:

    GovernedLLM(HttpLLM(provider), GovernorConfig(max_cost_usd=5.0))

enforces a ceiling against a field that is always 0.0. The governor never
fires. Every caller that believed it had a budget had nothing, and nothing
anywhere said so. Found on a live benchmark run that reported a spend of $0.00
for documents that had demonstrably cost money.

Two halves to the fix, and the second matters more than the first:

* a `PriceBook` the caller supplies, so `HttpLLM` can populate the field. The
  toolkit still ships **no prices**, for exactly the reason the README gave;
* a guard, so a ceiling that is being enforced against nothing says so instead
  of passing silently. A limit that cannot bind is worse than no limit, because
  it is believed.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toolkit.core import Message, Usage  # noqa: E402
from toolkit.core.errors import ValidationFailed  # noqa: E402
from toolkit.governor import GovernedLLM, GovernorConfig  # noqa: E402
from toolkit.llm_http import HttpLLM  # noqa: E402
from toolkit.provider import (  # noqa: E402
    ModelPrice,
    OffPeakWindow,
    PriceBook,
    Provider,
    resolve,
)

SECRET = "sk-super-secret-key-9999"


def _transport(payload: dict):
    def transport(url, headers, body, timeout):
        transport.seen = {"url": url, "body": body}
        return 200, json.dumps(payload)

    transport.seen = {}
    return transport


def _transport_status(status: int, payload: dict):
    """A transport that returns an arbitrary status, for classification tests."""

    def transport(url, headers, body, timeout):
        return status, json.dumps(payload)

    return transport


def _ok(prompt=1000, completion=2000) -> dict:
    return {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        "model": "deepseek-flash",
    }


def _deepseek() -> Provider:
    return resolve(env_file=None, environ={"DEEPSEEK_API_KEY": SECRET})


# --------------------------------------------------------------------------
# The price book
# --------------------------------------------------------------------------


def test_a_price_is_usd_per_million_tokens() -> None:
    price = ModelPrice(input_per_million=0.30, output_per_million=1.20)
    assert price.cost(1_000_000, 0) == pytest.approx(0.30)
    assert price.cost(0, 1_000_000) == pytest.approx(1.20)
    assert price.cost(1137, 1924) == pytest.approx(
        1137 * 0.30 / 1e6 + 1924 * 1.20 / 1e6
    )


def test_a_price_refuses_a_negative_rate() -> None:
    with pytest.raises(ValueError):
        ModelPrice(input_per_million=-1.0, output_per_million=1.0)


def test_a_book_prices_by_model_name() -> None:
    book = PriceBook({"deepseek-flash": ModelPrice(0.30, 1.20)})
    assert book.cost("deepseek-flash", 1_000_000, 0) == pytest.approx(0.30)


def test_an_unknown_model_costs_nothing_unless_a_default_is_given() -> None:
    """Returning a guess for a model nobody priced would be inventing a number.
    The *guard* is what stops that silence being dangerous, not a fabricated
    price."""
    book = PriceBook({"deepseek-flash": ModelPrice(0.30, 1.20)})
    assert book.cost("some-other-model", 1_000_000, 1_000_000) == 0.0
    assert book.for_model("some-other-model") is None

    guarded = PriceBook(
        {"deepseek-flash": ModelPrice(0.30, 1.20)},
        default=ModelPrice(1.00, 4.00),
    )
    assert guarded.cost("some-other-model", 1_000_000, 0) == pytest.approx(1.00)


def test_a_versioned_model_name_prices_as_its_family() -> None:
    """Providers append dates and revisions to a model name without changing
    what it costs."""
    book = PriceBook({"deepseek-flash": ModelPrice(0.30, 1.20)})
    assert book.cost("deepseek-flash-2026-10-01", 1_000_000, 0) == pytest.approx(0.30)


def test_an_empty_book_is_the_toolkit_default_and_changes_nothing() -> None:
    """Shipping prices would mean shipping a table that goes stale in a library
    nobody updates weekly. An empty book keeps today's behaviour exactly."""
    assert PriceBook().cost("deepseek-flash", 1_000_000, 1_000_000) == 0.0


# --------------------------------------------------------------------------
# Peak and off-peak
# --------------------------------------------------------------------------


def test_an_off_peak_window_discounts_within_its_hours() -> None:
    """DeepSeek halves its rate off-peak, and a benchmark run that straddles
    the boundary is mispriced by a factor of two without this."""
    from datetime import datetime, timezone

    book = PriceBook({
        "deepseek-flash": ModelPrice(
            0.30, 1.20,
            off_peak=OffPeakWindow(start_hour_utc=16, end_hour_utc=24, multiplier=0.5),
        )
    })
    peak = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
    off = datetime(2026, 10, 7, 18, 0, tzinfo=timezone.utc)

    assert book.cost("deepseek-flash", 1_000_000, 0, at=peak) == pytest.approx(0.30)
    assert book.cost("deepseek-flash", 1_000_000, 0, at=off) == pytest.approx(0.15)


def test_a_window_that_wraps_midnight_is_handled() -> None:
    """22:00 to 06:00 is one window, not two, and the naive start < hour < end
    comparison gets it backwards."""
    from datetime import datetime, timezone

    book = PriceBook({
        "m": ModelPrice(1.0, 1.0,
                        off_peak=OffPeakWindow(22, 6, multiplier=0.25)),
    })
    at_23 = datetime(2026, 10, 7, 23, 0, tzinfo=timezone.utc)
    at_03 = datetime(2026, 10, 7, 3, 0, tzinfo=timezone.utc)
    at_12 = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)

    assert book.cost("m", 1_000_000, 0, at=at_23) == pytest.approx(0.25)
    assert book.cost("m", 1_000_000, 0, at=at_03) == pytest.approx(0.25)
    assert book.cost("m", 1_000_000, 0, at=at_12) == pytest.approx(1.0)


def test_an_impossible_window_is_refused() -> None:
    with pytest.raises(ValueError):
        OffPeakWindow(start_hour_utc=25, end_hour_utc=3, multiplier=0.5)
    with pytest.raises(ValueError):
        OffPeakWindow(start_hour_utc=1, end_hour_utc=3, multiplier=-0.5)


# --------------------------------------------------------------------------
# HttpLLM fills the field in
# --------------------------------------------------------------------------


def test_http_llm_leaves_cost_at_zero_without_a_book() -> None:
    """Unchanged behaviour for every existing caller."""
    llm = HttpLLM(_deepseek(), transport=_transport(_ok()))
    assert llm.complete([Message("user", "hi")]).usage.cost_usd == 0.0


def test_http_llm_prices_the_call_when_given_a_book() -> None:
    book = PriceBook({"deepseek-flash": ModelPrice(0.30, 1.20)})
    llm = HttpLLM(_deepseek(), transport=_transport(_ok(1000, 2000)), prices=book)

    usage = llm.complete([Message("user", "hi")]).usage
    assert usage.input_tokens == 1000 and usage.output_tokens == 2000
    assert usage.cost_usd == pytest.approx(1000 * 0.30 / 1e6 + 2000 * 1.20 / 1e6)


def test_the_response_model_is_what_gets_priced() -> None:
    """A provider may answer with a different model than it was asked for - an
    alias, a fallback, a silent upgrade - and the bill follows the model that
    answered."""
    payload = _ok()
    payload["model"] = "deepseek-reasoner"
    book = PriceBook({
        "deepseek-flash": ModelPrice(0.30, 1.20),
        "deepseek-reasoner": ModelPrice(0.55, 2.19),
    })
    llm = HttpLLM(_deepseek(), transport=_transport(payload), prices=book)
    usage = llm.complete([Message("user", "hi")]).usage
    assert usage.cost_usd == pytest.approx(1000 * 0.55 / 1e6 + 2000 * 2.19 / 1e6)


# --------------------------------------------------------------------------
# The guard: a ceiling that cannot bind must say so
# --------------------------------------------------------------------------


def test_a_cost_ceiling_against_an_unpriced_model_is_refused() -> None:
    """The defect, as a test. A ceiling enforced against a field nothing fills
    in is believed and does nothing, which is worse than no ceiling at all."""

    class Unpriced:
        model_version = "unpriced:1"

        def complete(self, messages, temperature=0.0, max_tokens=None):
            from toolkit.core import Completion

            return Completion(
                text="{}",
                usage=Usage(input_tokens=100_000, output_tokens=100_000, cost_usd=0.0),
            )

    governed = GovernedLLM(Unpriced(), GovernorConfig(max_cost_usd=1.0))
    with pytest.raises(ValidationFailed) as caught:
        for _ in range(10):
            governed.complete([Message("user", "x")])
    message = str(caught.value).lower()
    assert "cost" in message
    assert "price" in message or "0.0" in message


def test_the_guard_does_not_fire_when_costs_are_real() -> None:
    class Priced:
        model_version = "priced:1"

        def complete(self, messages, temperature=0.0, max_tokens=None):
            from toolkit.core import Completion

            return Completion(
                text="{}",
                usage=Usage(input_tokens=10, output_tokens=10, cost_usd=0.001),
            )

    governed = GovernedLLM(Priced(), GovernorConfig(max_cost_usd=1.0))
    for _ in range(10):
        governed.complete([Message("user", "x")])
    assert governed.state.usage.cost_usd == pytest.approx(0.01)


def test_the_guard_is_silent_when_no_cost_ceiling_was_set() -> None:
    """Most callers do not set one, and a free model reporting zero is not a
    misconfiguration."""

    class Unpriced:
        model_version = "unpriced:1"

        def complete(self, messages, temperature=0.0, max_tokens=None):
            from toolkit.core import Completion

            return Completion(text="{}", usage=Usage(input_tokens=10, output_tokens=10))

    governed = GovernedLLM(Unpriced(), GovernorConfig(max_total_tokens=10_000_000))
    for _ in range(10):
        governed.complete([Message("user", "x")])
    assert governed.state.calls == 10


def test_the_guard_can_be_switched_off_for_a_genuinely_free_model() -> None:
    """A local model really does cost nothing, and the escape hatch has to be
    explicit rather than the default."""

    class Free:
        model_version = "local:1"

        def complete(self, messages, temperature=0.0, max_tokens=None):
            from toolkit.core import Completion

            return Completion(text="{}", usage=Usage(input_tokens=10, output_tokens=10))

    governed = GovernedLLM(
        Free(), GovernorConfig(max_cost_usd=1.0, require_priced_calls=False)
    )
    for _ in range(10):
        governed.complete([Message("user", "x")])
    assert governed.state.calls == 10

# --------------------------------------------------------------------------
# Provider-specific request parameters
# --------------------------------------------------------------------------


def test_extra_body_parameters_reach_the_request() -> None:
    """Every provider has a knob the port does not model.

    DeepSeek's `thinking` is the measured case: `deepseek-flash` is a reasoning
    model, and `{"type": "disabled"}` cut output tokens by 4.4x and latency by
    1.7x on an identical prompt. `ports.LLM` should not grow a `thinking`
    argument - the next provider calls it something else - but refusing to pass
    anything through means the only way to use it is to stop using the port.
    """
    transport = _transport(_ok())
    llm = HttpLLM(
        _deepseek(),
        transport=transport,
        extra_body={"thinking": {"type": "disabled"}},
    )
    llm.complete([Message("user", "hi")])

    body = transport.seen["body"]
    assert body["thinking"] == {"type": "disabled"}
    # And it did not disturb anything the port does model.
    assert body["model"] == "deepseek-flash"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.0


def test_extra_body_cannot_overwrite_the_fields_the_port_owns() -> None:
    """Otherwise a stray key silently changes the model or drops the messages,
    and the call that results is not the one the caller asked for."""
    transport = _transport(_ok())
    llm = HttpLLM(
        _deepseek(),
        transport=transport,
        extra_body={"model": "something-else", "messages": [], "thinking": {"type": "disabled"}},
    )
    llm.complete([Message("user", "hi")])

    body = transport.seen["body"]
    assert body["model"] == "deepseek-flash"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["thinking"] == {"type": "disabled"}


def test_extra_body_reaches_the_anthropic_shape_too() -> None:
    anthropic = resolve(env_file=None, environ={"ANTHROPIC_API_KEY": SECRET})
    transport = _transport({
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 5, "output_tokens": 5},
        "model": "claude-opus-5",
        "stop_reason": "end_turn",
    })
    HttpLLM(anthropic, transport=transport, extra_body={"top_k": 5}).complete(
        [Message("user", "hi")]
    )
    assert transport.seen["body"]["top_k"] == 5


def test_the_model_version_records_the_extra_parameters() -> None:
    """`CachedLLM` keys on `model_version`. Two runs that differ only in
    `thinking` are different requests and must not share cached answers - the
    TK-5 defect, in a new field."""
    plain = HttpLLM(_deepseek(), transport=_transport(_ok()))
    thinking_off = HttpLLM(
        _deepseek(),
        transport=_transport(_ok()),
        extra_body={"thinking": {"type": "disabled"}},
    )
    assert plain.model_version != thinking_off.model_version

def test_a_cached_replay_does_not_trip_the_unpriced_guard() -> None:
    """A false positive in the guard, found on a full-cache replay.

    `CachedLLM` reports a replayed completion at `cost_usd=0.0` and
    `cached=True`, and it is right to: the call really did cost nothing. The
    guard counted those zeros as evidence that nothing was priced, so a run
    that replayed every call from cache was killed by its own budget guard
    after five documents.

    That is not a benchmark problem. FDIP caches model calls by design
    (CLAUDE.md 3.2, record-replay), so a re-run of a cached job would have
    taken itself down in production.

    A cached call is therefore not evidence either way. The guard counts only
    calls that actually reached a provider and still reported nothing.
    """
    from toolkit.cache import CachedLLM, SqliteCache

    class Priced:
        model_version = "priced:1"

        def __init__(self):
            self.calls = 0

        def complete(self, messages, temperature=0.0, max_tokens=None):
            from toolkit.core import Completion

            self.calls += 1
            return Completion(
                text="{}",
                usage=Usage(input_tokens=10, output_tokens=10, cost_usd=0.002),
            )

    # The cache is populated by an earlier run, as it is in a real replay.
    store = SqliteCache()
    warm = CachedLLM(Priced(), store)
    prompts = [[Message("user", "doc %d" % n)] for n in range(12)]
    for messages in prompts:
        warm.complete(messages)

    # A fresh governor over the warm cache: every call is a replay, so the
    # total cost stays exactly zero for the whole run. This is the case the
    # first version of the test missed, because its first call was real and
    # priced, which hid the bug.
    inner = Priced()
    governed = GovernedLLM(
        CachedLLM(inner, store), GovernorConfig(max_cost_usd=5.0)
    )
    for messages in prompts:
        governed.complete(messages)

    assert inner.calls == 0, "every call should have been a replay"
    assert governed.state.calls == 12
    assert governed.state.usage.cost_usd == 0.0, "replays are free, correctly"


def test_an_unpriced_live_call_still_trips_the_guard_after_replays() -> None:
    """The guard must not be disarmed by the fix: a genuinely unpriced
    provider is still caught, even if some replays came first."""
    from toolkit.cache import CachedLLM, SqliteCache

    class Unpriced:
        model_version = "unpriced:1"

        def complete(self, messages, temperature=0.0, max_tokens=None):
            from toolkit.core import Completion

            return Completion(
                text="{}", usage=Usage(input_tokens=1000, output_tokens=1000)
            )

    cached = CachedLLM(Unpriced(), SqliteCache())
    governed = GovernedLLM(cached, GovernorConfig(max_cost_usd=5.0))

    with pytest.raises(ValidationFailed):
        # Each distinct prompt is a real call; none of them reports a cost.
        for n in range(12):
            governed.complete([Message("user", "prompt %d" % n)])

# --------------------------------------------------------------------------
# Terminal failures must not be retried
# --------------------------------------------------------------------------


def test_an_empty_balance_is_terminal_and_says_so() -> None:
    """Measured: a 175-document run died with

        giving up after 3 attempts: DeepSeek API error 402: Insufficient Balance

    Three attempts, because the governor retried. An empty account will not
    fill itself between attempts, so every retry was a wasted round trip and
    the message buried the one fact that mattered behind "API error 402".
    """
    from toolkit.core.errors import PermanentFailure

    llm = HttpLLM(_deepseek(), transport=_transport_status(402, {
        "error": {"message": "Insufficient Balance"}
    }))
    with pytest.raises(PermanentFailure) as caught:
        llm.complete([Message("user", "hi")])
    message = str(caught.value).lower()
    assert "balance" in message
    assert "top up" in message or "add credit" in message
    # And it is still an AdapterError, so every existing handler keeps working.
    from toolkit.core.errors import AdapterError as _AdapterError

    assert isinstance(caught.value, _AdapterError)


def test_the_governor_does_not_retry_a_permanent_failure() -> None:
    """The defect behind the wasted attempts. A bad key, a missing model, a
    malformed request and an empty balance will all fail identically on the
    second try."""
    from toolkit.core.errors import PermanentFailure

    class Refusing:
        model_version = "refusing:1"

        def __init__(self):
            self.calls = 0

        def complete(self, messages, temperature=0.0, max_tokens=None):
            self.calls += 1
            raise PermanentFailure("insufficient balance; top up the account")

    inner = Refusing()
    governed = GovernedLLM(inner, GovernorConfig(max_attempts=3))
    with pytest.raises(PermanentFailure):
        governed.complete([Message("user", "x")])
    assert inner.calls == 1, "a terminal failure must be attempted exactly once"
    assert governed.state.retries == 0


def test_a_transient_failure_is_still_retried() -> None:
    """The guard must not disarm the retry loop it sits in front of: a timeout
    or a 503 is worth another attempt."""
    from toolkit.core.errors import AdapterError

    class Flaky:
        model_version = "flaky:1"

        def __init__(self):
            self.calls = 0

        def complete(self, messages, temperature=0.0, max_tokens=None):
            from toolkit.core import Completion

            self.calls += 1
            if self.calls < 3:
                raise AdapterError("connection reset")
            return Completion(
                text="{}", usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.1)
            )

    inner = Flaky()
    governed = GovernedLLM(
        inner, GovernorConfig(max_attempts=3, initial_backoff=0.0, jitter=0.0)
    )
    assert governed.complete([Message("user", "x")]).text == "{}"
    assert inner.calls == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 402])
def test_every_configuration_failure_is_terminal(status) -> None:
    """None of these improves on a retry: a wrong key stays wrong."""
    from toolkit.core.errors import PermanentFailure

    llm = HttpLLM(_deepseek(), transport=_transport_status(status, {}))
    with pytest.raises(PermanentFailure):
        llm.complete([Message("user", "hi")])


@pytest.mark.parametrize("status", [429, 500, 503])
def test_an_overload_is_not_terminal(status) -> None:
    from toolkit.core.errors import RateLimited

    llm = HttpLLM(_deepseek(), transport=_transport_status(status, {}))
    with pytest.raises(RateLimited):
        llm.complete([Message("user", "hi")])


# --------------------------------------------------------------------------
# Malaysian currency notation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("RM1,234.50", "1234.50"),
        ("RM 1,234.50", "1234.50"),
        ("RM(310.00)", "-310.00"),
        ("(RM310.00)", "-310.00"),
        ("-RM310.00", "-310.00"),
        # The marker before the sign, which is the ordering the bracket fix did
        # not reach.
        ("RM-310.00", "-310.00"),
        ("(RM 310.00)", "-310.00"),
        ("rm1234.5", "1234.5"),
        ("RM 310", "310"),
    ],
)
def test_malaysian_currency_notation(text, expected) -> None:
    """Every form a Malaysian invoice actually writes. The marker may sit
    outside or inside the brackets, before or after the sign."""
    from decimal import Decimal

    from toolkit.extraction import parse_decimal

    assert parse_decimal(text) == Decimal(expected)


@pytest.mark.parametrize("text", ["RM", "RM-", "RM()", "RMx1.00", "1,234.50RM-"])
def test_notation_that_is_not_a_number_is_still_refused(text) -> None:
    from toolkit.extraction import parse_decimal

    assert parse_decimal(text) is None

def test_an_answer_truncated_before_any_text_is_terminal() -> None:
    """Measured on a scanned document at max_tokens=16000: the model spent the
    whole budget reasoning and returned empty content with
    `finish_reason=length`. The governor retried it three times.

    At temperature 0 the same request produces the same result, so the retries
    are waste - and worse, the retry wrapper hid the one actionable line
    ("raise max_tokens") behind "giving up after 3 attempts".
    """
    from toolkit.core.errors import PermanentFailure

    payload = {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 3000, "completion_tokens": 16000},
        "model": "deepseek-flash",
    }
    llm = HttpLLM(_deepseek(), transport=_transport(payload))
    with pytest.raises(PermanentFailure) as caught:
        llm.complete([Message("user", "hi")])
    assert "max_tokens" in str(caught.value)


def test_the_governor_does_not_retry_a_truncated_answer() -> None:
    from toolkit.core.errors import PermanentFailure

    payload = {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        "model": "deepseek-flash",
    }
    calls = {"n": 0}

    def counting(url, headers, body, timeout):
        calls["n"] += 1
        return 200, json.dumps(payload)

    governed = GovernedLLM(
        HttpLLM(_deepseek(), transport=counting), GovernorConfig(max_attempts=3)
    )
    with pytest.raises(PermanentFailure):
        governed.complete([Message("user", "x")])
    assert calls["n"] == 1
