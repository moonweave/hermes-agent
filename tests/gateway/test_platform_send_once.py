from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendOnceOutcome,
)
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_base_send_once_is_explicitly_unsupported():
    result = await BasePlatformAdapter.send_once(
        cast(BasePlatformAdapter, object()), "chat", "final"
    )

    assert result.outcome is SendOnceOutcome.FAILED
    assert result.definitive is True
    assert result.error == "send_once_unsupported"


def _telegram_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_telegram_send_once_makes_one_transport_call_and_returns_receipt():
    adapter = _telegram_adapter()
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))

    result = await adapter.send_once("123", "final answer", metadata={"notify": True})

    assert bot.send_message.await_count == 1
    assert result.outcome is SendOnceOutcome.DELIVERED
    assert result.definitive is True
    assert result.message_id == "42"


@pytest.mark.asyncio
async def test_telegram_send_once_timeout_is_uncertain_and_never_retries():
    adapter = _telegram_adapter()
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock(side_effect=asyncio.TimeoutError("timed out"))

    result = await adapter.send_once("123", "final answer")

    assert bot.send_message.await_count == 1
    assert result.outcome is SendOnceOutcome.DELIVERY_UNCERTAIN
    assert result.definitive is False


@pytest.mark.asyncio
async def test_telegram_send_once_actual_ptb_bad_request_is_failed():
    from telegram.error import BadRequest  # type: ignore[reportMissingImports]

    adapter = _telegram_adapter()
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock(side_effect=BadRequest("message text is empty"))

    result = await adapter.send_once("123", "final answer")

    assert bot.send_message.await_count == 1
    assert result.outcome is SendOnceOutcome.FAILED
    assert result.definitive is True


@pytest.mark.asyncio
async def test_telegram_send_once_rejects_oversize_before_transport():
    adapter = _telegram_adapter()
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock()

    result = await adapter.send_once("123", "x" * 5000)

    bot.send_message.assert_not_awaited()
    assert result.outcome is SendOnceOutcome.FAILED
    assert result.error == "content_too_long_for_send_once"


@pytest.mark.asyncio
async def test_telegram_send_once_invalid_content_is_structured_preflight_failure():
    adapter = _telegram_adapter()
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock()

    result = await adapter.send_once("123", cast(Any, object()))

    bot.send_message.assert_not_awaited()
    assert result.outcome is SendOnceOutcome.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preflight_method",
    ["format_message", "_compute_single_send_routing", "_link_preview_kwargs"],
)
async def test_telegram_send_once_preflight_errors_are_structured_failed(
    monkeypatch, preflight_method
):
    adapter = _telegram_adapter()
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock()
    monkeypatch.setattr(
        adapter,
        preflight_method,
        MagicMock(side_effect=ValueError("invalid preflight input")),
    )

    result = await adapter.send_once("123", "final answer")

    bot.send_message.assert_not_awaited()
    assert result.outcome is SendOnceOutcome.FAILED
    assert result.definitive is True


@pytest.mark.asyncio
async def test_telegram_send_once_normalization_error_is_preflight_failed(monkeypatch):
    adapter = _telegram_adapter()
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.normalize_telegram_chat_id",
        MagicMock(side_effect=ValueError("invalid chat id")),
    )

    result = await adapter.send_once("bad", "final answer")

    bot.send_message.assert_not_awaited()
    assert result.outcome is SendOnceOutcome.FAILED


@pytest.mark.asyncio
async def test_telegram_send_once_value_error_after_invocation_is_uncertain():
    adapter = _telegram_adapter()
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock(side_effect=ValueError("post-request decode failed"))

    result = await adapter.send_once("123", "final answer")

    assert bot.send_message.await_count == 1
    assert result.outcome is SendOnceOutcome.DELIVERY_UNCERTAIN
    assert result.definitive is False
