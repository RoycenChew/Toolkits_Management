"""TK-3: images on a `Message`, and what each API style does with them.

A scanned page is sometimes better read by a vision model than by OCR, and the
toolkit had no way to send one: `Message` carried text and nothing else. The
addition is deliberately small — bytes plus a media type — because the two
request shapes that matter disagree about everything else, and a contract that
encodes either one of them stops being a contract.

Three things have to stay true, and each has a test here:

* a text-only message still sends `content` as a plain **string**. Both
  providers accept a parts list for text, but switching every existing call to
  one changes the request every pipeline already depends on for no benefit;
* OpenAI gets `image_url` parts holding a base64 data URL, Anthropic gets
  `image` blocks with a base64 `source`;
* `CachedLLM` keys on the images too. This is exactly the shape of the defect
  TK-5 fixed: two requests that differ only in a field the key ignores serve
  each other's answers, with no error anywhere.

Every test runs offline through the injectable transport.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toolkit.adapters import ScriptedLLM  # noqa: E402
from toolkit.cache import CachedLLM, SqliteCache  # noqa: E402
from toolkit.core import ImagePart, Message  # noqa: E402
from toolkit.core.errors import AdapterError  # noqa: E402
from toolkit.llm_http import HttpLLM  # noqa: E402
from toolkit.provider import Provider, resolve  # noqa: E402

SECRET = "sk-super-secret-key-9999"

# A one-pixel PNG. Real bytes, because a data URL built from text that is not an
# image is the kind of thing that only fails at the provider.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000050001" "0d0a2db4000000004945" "4e44ae426082"
)
JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")


def _transport(status: int, payload: dict):
    """A fake transport that records what it was asked to send."""

    def transport(url, headers, body, timeout):
        transport.seen = {"url": url, "headers": dict(headers), "body": body}
        return status, json.dumps(payload)

    transport.seen = {}
    return transport


def _openai_ok(text: str = "ok") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 2},
        "model": "deepseek-chat",
    }


def _anthropic_ok(text: str = "ok") -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 11, "output_tokens": 2},
        "model": "claude-sonnet-4-5",
        "stop_reason": "end_turn",
    }


def _deepseek() -> Provider:
    return resolve(env_file=None, environ={"DEEPSEEK_API_KEY": SECRET})


def _anthropic() -> Provider:
    return resolve(env_file=None, environ={"ANTHROPIC_API_KEY": SECRET})


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


def test_an_image_part_rejects_what_cannot_be_sent() -> None:
    with pytest.raises(ValueError):
        ImagePart(b"")
    with pytest.raises(ValueError):
        ImagePart(PNG, media_type="application/pdf")
    with pytest.raises(ValueError):
        ImagePart(PNG, media_type="")


def test_an_image_part_renders_base64_and_a_data_url() -> None:
    part = ImagePart(PNG, media_type="image/png")
    assert part.base64.isascii() and "\n" not in part.base64
    assert part.data_url == "data:image/png;base64," + part.base64
    assert part.media_type == "image/png"


def test_an_image_part_fingerprint_follows_the_bytes() -> None:
    """The cache keys on this, so two different images must never share one and
    the same image must always give the same one."""
    assert ImagePart(PNG).fingerprint() == ImagePart(PNG).fingerprint()
    assert ImagePart(PNG).fingerprint() != ImagePart(JPEG, "image/jpeg").fingerprint()
    # The media type is part of the identity: the same bytes declared as a
    # different type is a different request.
    assert ImagePart(PNG).fingerprint() != ImagePart(PNG, "image/jpeg").fingerprint()


def test_messages_without_images_are_unchanged() -> None:
    """Additive: every existing positional and keyword call keeps working."""
    assert list(Message("user", "hi").images) == []
    assert list(Message(role="system", content="be brief").images) == []


# --------------------------------------------------------------------------
# OpenAI style
# --------------------------------------------------------------------------


def test_a_text_only_message_still_sends_a_plain_string() -> None:
    """Both providers accept a parts list for text. Sending one anyway would
    change the request body of every existing pipeline for no benefit."""
    transport = _transport(200, _openai_ok())
    HttpLLM(_deepseek(), transport=transport).complete([Message("user", "hi")])
    sent = transport.seen["body"]["messages"]
    assert sent == [{"role": "user", "content": "hi"}]


def test_openai_style_sends_text_and_an_image_url_part() -> None:
    transport = _transport(200, _openai_ok())
    message = Message("user", "What is the total?", images=[ImagePart(PNG)])
    HttpLLM(_deepseek(), transport=transport).complete([message])

    [sent] = transport.seen["body"]["messages"]
    assert sent["role"] == "user"
    assert sent["content"] == [
        {"type": "text", "text": "What is the total?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + ImagePart(PNG).base64},
        },
    ]


def test_openai_style_sends_several_images_in_order() -> None:
    transport = _transport(200, _openai_ok())
    message = Message(
        "user", "", images=[ImagePart(PNG), ImagePart(JPEG, "image/jpeg")]
    )
    HttpLLM(_deepseek(), transport=transport).complete([message])

    content = transport.seen["body"]["messages"][0]["content"]
    # No empty text part: an image-only turn is legitimate, and a blank text
    # block is rejected by some providers.
    assert [part["type"] for part in content] == ["image_url", "image_url"]
    assert "image/png" in content[0]["image_url"]["url"]
    assert "image/jpeg" in content[1]["image_url"]["url"]


# --------------------------------------------------------------------------
# Anthropic style
# --------------------------------------------------------------------------


def test_anthropic_style_sends_a_base64_image_block_before_the_text() -> None:
    """Anthropic documents images first, text second, and follows that order
    itself in every example. Text-first measurably degrades its answers."""
    transport = _transport(200, _anthropic_ok())
    message = Message("user", "What is the total?", images=[ImagePart(PNG)])
    HttpLLM(_anthropic(), transport=transport).complete([message])

    [sent] = transport.seen["body"]["messages"]
    assert sent["content"] == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": ImagePart(PNG).base64,
            },
        },
        {"type": "text", "text": "What is the total?"},
    ]


def test_anthropic_style_keeps_a_text_only_turn_as_a_string() -> None:
    transport = _transport(200, _anthropic_ok())
    HttpLLM(_anthropic(), transport=transport).complete(
        [Message("system", "be brief"), Message("user", "hi")]
    )
    body = transport.seen["body"]
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["system"] == "be brief"


def test_an_image_on_an_anthropic_system_prompt_is_refused() -> None:
    """The system prompt is a top-level string in this API, so an image on it
    has nowhere to go. Dropping it silently would send a request that cannot
    answer the question it was asked."""
    transport = _transport(200, _anthropic_ok())
    llm = HttpLLM(_anthropic(), transport=transport)
    with pytest.raises(AdapterError) as caught:
        llm.complete(
            [Message("system", "read this", images=[ImagePart(PNG)]), Message("user", "?")]
        )
    assert "system" in str(caught.value).lower()


# --------------------------------------------------------------------------
# ScriptedLLM
# --------------------------------------------------------------------------


def test_the_scripted_model_records_the_images_it_was_sent() -> None:
    """Same reason it records prompts: the thing that is wrong is usually what
    the pipeline actually sent."""
    llm = ScriptedLLM(responses=["ok"])
    llm.complete([Message("user", "look", images=[ImagePart(PNG)])])

    assert llm.images == [[ImagePart(PNG)]]
    assert llm.calls[0][0].images[0].media_type == "image/png"


def test_the_scripted_model_records_an_empty_list_for_text_only_calls() -> None:
    llm = ScriptedLLM(responses=["a", "b"])
    llm.complete([Message("user", "one")])
    llm.complete([Message("user", "two", images=[ImagePart(JPEG, "image/jpeg")])])
    assert llm.images == [[], [ImagePart(JPEG, "image/jpeg")]]


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


def test_two_requests_differing_only_in_the_image_do_not_share_an_answer() -> None:
    """The TK-5 defect in a new field: a key that ignores part of the request
    lets two different requests serve each other, with no error anywhere."""
    inner = ScriptedLLM(responses=["first page", "second page"])
    cached = CachedLLM(inner, SqliteCache())

    one = cached.complete([Message("user", "read it", images=[ImagePart(PNG)])])
    two = cached.complete(
        [Message("user", "read it", images=[ImagePart(JPEG, "image/jpeg")])]
    )

    assert one.text == "first page"
    assert two.text == "second page"
    assert inner.calls and cached.calls == 2


def test_the_same_image_hits_the_cache() -> None:
    inner = ScriptedLLM(responses=["read once"])
    cached = CachedLLM(inner, SqliteCache())
    messages = [Message("user", "read it", images=[ImagePart(PNG)])]

    first = cached.complete(messages)
    second = cached.complete(messages)

    assert first.text == second.text == "read once"
    assert second.usage.cached is True
    assert cached.calls == 1


def test_an_image_bearing_request_does_not_collide_with_the_text_only_one() -> None:
    inner = ScriptedLLM(responses=["with picture", "without"])
    cached = CachedLLM(inner, SqliteCache())

    with_image = cached.complete([Message("user", "read it", images=[ImagePart(PNG)])])
    without = cached.complete([Message("user", "read it")])

    assert with_image.text == "with picture"
    assert without.text == "without"
