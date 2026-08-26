from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import gc
import hashlib
import logging
import re
import threading
import time
import weakref
from types import SimpleNamespace
from typing import Any, TypedDict, cast

import pytest

from agent.route_capability import (
    CoordinatorRouteError,
    GatewayDeliveryService,
    GatewayDeliveryState,
    GatewayDeliveryStatus,
    activate_gateway_dispatch_binding,
    issue_gateway_dispatch_binding,
    reset_coordinator_registry_for_tests,
    revoke_gateway_dispatch_binding,
)
from gateway import plugin_delivery_ledger as ledger
from gateway.platforms.base import SendOnceOutcome, SendOnceResult
from gateway.plugin_delivery_service import (
    _METADATA_KEY,
    _persist_resolver,
    _recovery_context_digest,
    _transport_owner_digest,
    cancel_prepared,
    deliver_once,
    prepare_once,
    reconcile,
    reconcile_now,
    recover_pending,
    send_prepared_once,
)
from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    path = tmp_path / "profile" / "plugin-delivery.sqlite3"
    path.parent.mkdir(mode=0o700)
    monkeypatch.setattr(ledger, "_db_path", lambda: path)
    reset_coordinator_registry_for_tests()
    yield path
    reset_coordinator_registry_for_tests()


class Store:
    def __init__(self, entry, *more_entries):
        self.entry = entry
        self.entries = [entry, *more_entries]
        self.writes = []

    def lookup_by_session_key(self, session_key):
        return next(
            (entry for entry in self.entries if session_key == entry.session_key),
            None,
        )

    def list_sessions(self):
        return list(self.entries)

    def get_session_metadata(self, session_key, key, default=None):
        entry = self.lookup_by_session_key(session_key)
        return default if entry is None else entry.metadata.get(key, default)

    def set_session_metadata(self, session_key, key, value):
        entry = self.lookup_by_session_key(session_key)
        assert entry is not None
        entry.metadata[key] = value
        self.writes.append(SimpleNamespace(value=value))
        return True

    def update_session(self, session_key, *, touch_activity=True):
        assert self.lookup_by_session_key(session_key) is not None
        return None


class _DeliveryCall(TypedDict):
    runner: Any
    plugin_id: str
    binding_handle: str
    binding_expires_at: float
    session_id: str
    session_key: str
    source: SessionSource


class _DeliveryContext(TypedDict):
    runner: Any
    plugin_id: str
    binding_expires_at: float
    session_id: str
    session_key: str
    source: SessionSource


class Adapter:
    def __init__(self, outcome=SendOnceOutcome.DELIVERED):
        self.outcome = outcome
        self.calls = []

    async def send_once(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append((chat_id, content, reply_to, metadata))
        return SendOnceResult(self.outcome, message_id="receipt-secret")


def _target_metadata(
    platform, chat_id, thread_id, *, chat_type, reply_to_message_id, adapter
):
    assert platform is Platform.TELEGRAM
    assert chat_id == "private-chat-123"
    assert chat_type == "dm"
    assert reply_to_message_id is None
    assert adapter is not None
    return {
        "thread_id": thread_id,
        "telegram_dm_topic_reply_fallback": True,
        "direct_messages_topic_id": thread_id,
    }


def _recovery_runner(store, adapter, *, adapters=None, profile_adapters=None):
    return SimpleNamespace(
        session_store=store,
        _adapter_for_source=lambda _source: adapter,
        adapters=({Platform.TELEGRAM: adapter} if adapters is None else adapters),
        _profile_adapters=profile_adapters or {},
        _thread_metadata_for_target=_target_metadata,
    )


def _resolver_value(*, expires_at):
    return {
        "expires_at": expires_at,
        "recovery_context_digest": "a" * 64,
        "transport_owner_digest": "b" * 64,
    }


def test_resolver_persist_prunes_expired_and_malformed_entries_before_cap():
    _bound, _binding, store, _adapter = _delivery_fixture()
    now = time.time()
    mapping = {}
    for index in range(32):
        digest = f"{index:064x}"
        if index % 6 == 0:
            mapping[digest] = _resolver_value(expires_at=now - 1)
        elif index % 6 == 1:
            mapping[digest] = {"expires_at": now + 60}
        elif index % 6 == 2:
            mapping[digest] = {
                **_resolver_value(expires_at=now + 60),
                "transport_owner_digest": "not-a-digest",
            }
        elif index % 6 == 3:
            mapping[digest] = _resolver_value(expires_at=float("nan"))
        elif index % 6 == 4:
            mapping[digest] = _resolver_value(expires_at=True)
        else:
            mapping[f"invalid-{index}"] = _resolver_value(expires_at=now + 60)
    store.entry.metadata[_METADATA_KEY] = mapping

    assert _persist_resolver(
        store,
        session_key=store.entry.session_key,
        binding_digest="f" * 64,
        recovery_context_digest="c" * 64,
        transport_owner_digest="d" * 64,
        expires_at=now + 60,
    )

    assert store.entry.metadata[_METADATA_KEY] == {
        "f" * 64: {
            "expires_at": now + 60,
            "recovery_context_digest": "c" * 64,
            "transport_owner_digest": "d" * 64,
        }
    }


def test_resolver_persist_rejects_full_live_cap_without_mutation():
    _bound, _binding, store, _adapter = _delivery_fixture()
    now = time.time()
    mapping = {
        f"{index:064x}": _resolver_value(expires_at=now + 60) for index in range(32)
    }
    store.entry.metadata[_METADATA_KEY] = mapping
    writes_before = len(store.writes)

    assert not _persist_resolver(
        store,
        session_key=store.entry.session_key,
        binding_digest="f" * 64,
        recovery_context_digest="c" * 64,
        transport_owner_digest="d" * 64,
        expires_at=now + 60,
    )
    assert store.entry.metadata[_METADATA_KEY] == mapping
    assert len(store.writes) == writes_before


def test_concurrent_resolver_persists_never_exceed_cap():
    _bound, _binding, store, _adapter = _delivery_fixture()
    now = time.time()
    start = threading.Event()

    def persist(index):
        start.wait()
        return _persist_resolver(
            store,
            session_key=store.entry.session_key,
            binding_digest=f"{index:064x}",
            recovery_context_digest="c" * 64,
            transport_owner_digest="d" * 64,
            expires_at=now + 60,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        futures = [pool.submit(persist, index) for index in range(64)]
        start.set()
        results = [future.result(timeout=5) for future in futures]

    assert sum(results) == 32
    assert len(store.entry.metadata[_METADATA_KEY]) == 32


def _delivery_record(delivery_id):
    assert isinstance(delivery_id, str)
    record = ledger.get_delivery(delivery_id)
    assert record is not None
    return record


def _delivery_fixture(
    adapter=None, *, message_id="message-1", persisted_message_id=None
):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_type="dm",
        chat_id="private-chat-123",
        thread_id="topic-7",
        user_id="user-1",
        message_id=message_id,
    )
    entry = SimpleNamespace(
        session_key="session-key-1",
        session_id="session-1",
        origin=dataclasses.replace(
            source,
            message_id=(
                message_id if persisted_message_id is None else persisted_message_id
            ),
        ),
        metadata={},
    )
    store = Store(entry)
    adapter = adapter or Adapter()
    runner = SimpleNamespace(
        session_store=store,
        _adapter_for_source=lambda candidate: adapter if candidate == source else None,
        _thread_metadata_for_target=_target_metadata,
    )

    async def delivery(plugin_id, binding_handle, expires_at, delivery_key, content):
        return await deliver_once(
            runner,
            plugin_id=plugin_id,
            binding_handle=binding_handle,
            binding_expires_at=expires_at,
            session_id=entry.session_id,
            session_key=entry.session_key,
            source=source,
            delivery_key=delivery_key,
            content=content,
        )

    async def prepare(
        plugin_id,
        binding_handle,
        expires_at,
        reconciliation_id,
        delivery_key,
        content,
    ):
        return await prepare_once(
            runner,
            plugin_id=plugin_id,
            binding_handle=binding_handle,
            binding_expires_at=expires_at,
            session_id=entry.session_id,
            session_key=entry.session_key,
            source=source,
            reconciliation_id=reconciliation_id,
            delivery_key=delivery_key,
            content=content,
        )

    async def send_prepared(plugin_id, binding_handle, expires_at, reconciliation_id):
        return await send_prepared_once(
            runner,
            plugin_id=plugin_id,
            binding_handle=binding_handle,
            binding_expires_at=expires_at,
            session_id=entry.session_id,
            session_key=entry.session_key,
            source=source,
            reconciliation_id=reconciliation_id,
        )

    def cancel(plugin_id, binding_handle, reconciliation_id):
        return cancel_prepared(
            runner,
            plugin_id=plugin_id,
            binding_handle=binding_handle,
            session_key=entry.session_key,
            reconciliation_id=reconciliation_id,
        )

    binding = issue_gateway_dispatch_binding(
        issuer_plugin_id="fixture.delivery",
        parent=SimpleNamespace(session_id=entry.session_id),
        parent_session_id=entry.session_id,
        parent_turn_id="turn-1",
        user_message="question",
        platform="telegram",
        message_id=message_id,
        schedule=lambda *_args: "unused",
        validity_resolver=lambda: True,
        delivery=delivery,
        delivery_prepare=prepare,
        delivery_send_prepared=send_prepared,
        delivery_cancel_prepared=cancel,
    )
    bound = GatewayDeliveryService(
        "fixture.delivery", authorization_resolver=lambda: True
    ).for_gateway_binding(binding)
    activate_gateway_dispatch_binding(binding)
    return bound, binding, store, adapter


@pytest.mark.asyncio
async def test_two_phase_prepare_reconcile_and_send_without_plugin_content_replay():
    _bound, _binding, store, adapter = _delivery_fixture()
    runner = _recovery_runner(store, adapter)
    source = store.entry.origin
    common: _DeliveryCall = {
        "runner": runner,
        "plugin_id": "fixture.delivery",
        "binding_handle": "private-binding-handle",
        "binding_expires_at": time.time() + 60,
        "session_id": store.entry.session_id,
        "session_key": store.entry.session_key,
        "source": source,
    }

    assert await reconcile(
        plugin_id="fixture.delivery", reconciliation_id="consultation-1"
    ) == GatewayDeliveryStatus(GatewayDeliveryState.NOT_RESERVED)
    assert reconcile_now(
        plugin_id="fixture.delivery",
        reconciliation_id="consultation-1",
    ) == GatewayDeliveryStatus(GatewayDeliveryState.NOT_RESERVED)
    prepared = await prepare_once(
        **common,
        reconciliation_id="consultation-1",
        delivery_key="final",
        content="safe answer",
    )
    assert prepared == GatewayDeliveryStatus(GatewayDeliveryState.PENDING)
    assert adapter.calls == []
    assert await reconcile(
        plugin_id="fixture.delivery", reconciliation_id="consultation-1"
    ) == GatewayDeliveryStatus(GatewayDeliveryState.PENDING)
    assert reconcile_now(
        plugin_id="fixture.delivery",
        reconciliation_id="consultation-1",
    ) == GatewayDeliveryStatus(GatewayDeliveryState.PENDING)

    sent = await send_prepared_once(
        **common,
        reconciliation_id="consultation-1",
    )
    assert sent == GatewayDeliveryStatus(
        GatewayDeliveryState.DELIVERED,
        transport_attempted=True,
    )
    assert len(adapter.calls) == 1
    assert await send_prepared_once(
        **common,
        reconciliation_id="consultation-1",
    ) == GatewayDeliveryStatus(GatewayDeliveryState.DELIVERED)
    assert len(adapter.calls) == 1
    assert await reconcile(
        plugin_id="fixture.delivery", reconciliation_id="consultation-1"
    ) == GatewayDeliveryStatus(GatewayDeliveryState.DELIVERED)


@pytest.mark.asyncio
async def test_reconcile_read_failure_is_retryable_not_terminal(monkeypatch):
    def fail_read(**_kwargs):
        raise OSError("ledger unavailable")

    monkeypatch.setattr(
        "gateway.plugin_delivery_ledger.get_delivery_by_reconciliation_id",
        fail_read,
    )
    expected = GatewayDeliveryStatus(
        GatewayDeliveryState.RETRYABLE,
        error_code="reconciliation_unavailable",
    )

    assert (
        reconcile_now(
            plugin_id="fixture.delivery",
            reconciliation_id="consultation-retry",
        )
        == expected
    )
    assert (
        await reconcile(
            plugin_id="fixture.delivery",
            reconciliation_id="consultation-retry",
        )
        == expected
    )


@pytest.mark.asyncio
async def test_binding_revocation_cancels_prepared_pending_before_restart_recovery():
    bound, binding, store, adapter = _delivery_fixture()
    assert await bound.prepare_once(
        reconciliation_id="consultation-stop",
        delivery_key="final",
        content="safe answer",
    ) == GatewayDeliveryStatus(GatewayDeliveryState.PENDING)

    revoke_gateway_dispatch_binding(binding)

    with pytest.raises(CoordinatorRouteError, match="Unknown"):
        await bound.send_prepared_once(reconciliation_id="consultation-stop")
    assert await reconcile(
        plugin_id="fixture.delivery",
        reconciliation_id="consultation-stop",
    ) == GatewayDeliveryStatus(
        GatewayDeliveryState.FAILED,
        error_code="cancelled",
    )
    counts = await recover_pending(_recovery_runner(store, adapter))
    assert counts == {"pending": 0, "delivered": 0, "failed": 0, "uncertain": 0}
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_bound_cancel_prepared_once_is_terminal_idempotent_and_transport_free():
    bound, _binding, _store, adapter = _delivery_fixture()
    assert await bound.prepare_once(
        reconciliation_id="consultation-state-write-failed",
        delivery_key="final",
        content="safe answer",
    ) == GatewayDeliveryStatus(GatewayDeliveryState.PENDING)

    cancelled = bound.cancel_prepared_once(
        reconciliation_id="consultation-state-write-failed"
    )
    assert cancelled == GatewayDeliveryStatus(
        GatewayDeliveryState.FAILED,
        error_code="cancelled",
    )
    assert bound.cancel_prepared_once(
        reconciliation_id="consultation-state-write-failed"
    ) == GatewayDeliveryStatus(
        GatewayDeliveryState.FAILED,
        error_code="cancelled",
    )
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_two_phase_prepared_pending_is_the_only_restart_resumable_state():
    _bound, _binding, store, adapter = _delivery_fixture()
    common: _DeliveryCall = {
        "runner": _recovery_runner(store, adapter),
        "plugin_id": "fixture.delivery",
        "binding_handle": "private-binding-handle",
        "binding_expires_at": time.time() + 60,
        "session_id": store.entry.session_id,
        "session_key": store.entry.session_key,
        "source": store.entry.origin,
    }
    await prepare_once(
        **common,
        reconciliation_id="consultation-pending",
        delivery_key="final",
        content="safe answer",
    )

    counts = await recover_pending(_recovery_runner(store, adapter))

    assert counts == {"pending": 1, "delivered": 1, "failed": 0, "uncertain": 0}
    assert len(adapter.calls) == 1
    assert await reconcile(
        plugin_id="fixture.delivery", reconciliation_id="consultation-pending"
    ) == GatewayDeliveryStatus(GatewayDeliveryState.DELIVERED)
    second = await recover_pending(_recovery_runner(store, adapter))
    assert second == {"pending": 0, "delivered": 0, "failed": 0, "uncertain": 0}
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_reconcile_maps_claimed_uncertain_and_failed_reasons_without_secrets():
    _bound, _binding, store, adapter = _delivery_fixture()
    common: _DeliveryCall = {
        "runner": _recovery_runner(store, adapter),
        "plugin_id": "fixture.delivery",
        "binding_handle": "private-binding-handle",
        "binding_expires_at": time.time() + 60,
        "session_id": store.entry.session_id,
        "session_key": store.entry.session_key,
        "source": store.entry.origin,
    }
    await prepare_once(
        **common,
        reconciliation_id="consultation-claimed",
        delivery_key="claimed-final",
        content="secret claimed answer",
    )
    claim = ledger.claim_for_send(
        plugin_id="fixture.delivery",
        binding_handle="private-binding-handle",
        delivery_key="claimed-final",
    )
    assert claim is not None
    assert await reconcile(
        plugin_id="fixture.delivery", reconciliation_id="consultation-claimed"
    ) == GatewayDeliveryStatus(GatewayDeliveryState.SEND_CLAIMED)
    ledger.recover_after_restart()
    uncertain = await reconcile(
        plugin_id="fixture.delivery", reconciliation_id="consultation-claimed"
    )
    assert uncertain == GatewayDeliveryStatus(GatewayDeliveryState.DELIVERY_UNCERTAIN)

    await prepare_once(
        **common,
        reconciliation_id="consultation-cancelled",
        delivery_key="cancelled-final",
        content="secret cancelled answer",
    )
    cancelled_record = ledger.get_delivery_by_reconciliation_id(
        plugin_id="fixture.delivery",
        reconciliation_id="consultation-cancelled",
    )
    assert cancelled_record is not None
    assert isinstance(cancelled_record.recovery_context_digest, str)
    assert ledger.cancel_pending_delivery(
        delivery_id=cancelled_record.delivery_id,
        plugin_id=cancelled_record.plugin_id,
        binding_digest=cancelled_record.binding_digest,
        recovery_context_digest=cancelled_record.recovery_context_digest,
        reason="delivery_cancelled",
    )
    cancelled = await reconcile(
        plugin_id="fixture.delivery", reconciliation_id="consultation-cancelled"
    )
    assert cancelled == GatewayDeliveryStatus(
        GatewayDeliveryState.FAILED,
        error_code="cancelled",
    )
    serialized = repr((uncertain, cancelled))
    assert "secret" not in serialized
    assert "private-binding-handle" not in serialized


@pytest.mark.asyncio
async def test_prepared_binding_mismatch_never_claims_or_sends():
    _bound, _binding, store, adapter = _delivery_fixture()
    common: _DeliveryContext = {
        "runner": _recovery_runner(store, adapter),
        "plugin_id": "fixture.delivery",
        "binding_expires_at": time.time() + 60,
        "session_id": store.entry.session_id,
        "session_key": store.entry.session_key,
        "source": store.entry.origin,
    }
    await prepare_once(
        **common,
        binding_handle="original-private-binding",
        reconciliation_id="consultation-binding",
        delivery_key="final",
        content="safe answer",
    )

    result = await send_prepared_once(
        **common,
        binding_handle="different-private-binding",
        reconciliation_id="consultation-binding",
    )

    assert result == GatewayDeliveryStatus(
        GatewayDeliveryState.FAILED,
        error_code="binding_mismatch",
    )
    assert adapter.calls == []
    assert await reconcile(
        plugin_id="fixture.delivery", reconciliation_id="consultation-binding"
    ) == GatewayDeliveryStatus(GatewayDeliveryState.PENDING)


@pytest.mark.asyncio
async def test_simultaneous_prepared_send_has_one_transport_attempt():
    _bound, _binding, store, adapter = _delivery_fixture()
    common: _DeliveryCall = {
        "runner": _recovery_runner(store, adapter),
        "plugin_id": "fixture.delivery",
        "binding_handle": "private-binding-handle",
        "binding_expires_at": time.time() + 60,
        "session_id": store.entry.session_id,
        "session_key": store.entry.session_key,
        "source": store.entry.origin,
    }
    await prepare_once(
        **common,
        reconciliation_id="consultation-concurrent",
        delivery_key="final",
        content="safe answer",
    )

    results = await asyncio.gather(
        send_prepared_once(**common, reconciliation_id="consultation-concurrent"),
        send_prepared_once(**common, reconciliation_id="consultation-concurrent"),
    )

    assert len(adapter.calls) == 1
    assert {result.state for result in results} <= {
        GatewayDeliveryState.SEND_CLAIMED,
        GatewayDeliveryState.DELIVERED,
    }
    assert await reconcile(
        plugin_id="fixture.delivery", reconciliation_id="consultation-concurrent"
    ) == GatewayDeliveryStatus(GatewayDeliveryState.DELIVERED)


@pytest.mark.asyncio
async def test_cancelled_prepared_send_is_never_resumed_or_sent_later():
    started = asyncio.Event()

    class BlockingAdapter(Adapter):
        async def send_once(
            self, chat_id, content, reply_to=None, metadata=None
        ) -> SendOnceResult:
            self.calls.append((chat_id, content, reply_to, metadata))
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    adapter = BlockingAdapter()
    _bound, _binding, store, _adapter = _delivery_fixture(adapter)
    common: _DeliveryCall = {
        "runner": _recovery_runner(store, adapter),
        "plugin_id": "fixture.delivery",
        "binding_handle": "private-binding-handle",
        "binding_expires_at": time.time() + 60,
        "session_id": store.entry.session_id,
        "session_key": store.entry.session_key,
        "source": store.entry.origin,
    }
    await prepare_once(
        **common,
        reconciliation_id="consultation-cancel-active",
        delivery_key="final",
        content="safe answer",
    )
    task = asyncio.create_task(
        send_prepared_once(
            **common,
            reconciliation_id="consultation-cancel-active",
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(adapter.calls) == 1
    assert await reconcile(
        plugin_id="fixture.delivery",
        reconciliation_id="consultation-cancel-active",
    ) == GatewayDeliveryStatus(GatewayDeliveryState.SEND_CLAIMED)

    counts = await recover_pending(_recovery_runner(store, adapter))
    assert counts == {"pending": 0, "delivered": 0, "failed": 0, "uncertain": 0}
    assert len(adapter.calls) == 1
    assert await reconcile(
        plugin_id="fixture.delivery",
        reconciliation_id="consultation-cancel-active",
    ) == GatewayDeliveryStatus(GatewayDeliveryState.DELIVERY_UNCERTAIN)


@pytest.mark.asyncio
async def test_bound_delivery_sends_once_and_persists_only_digest_resolver():
    bound, binding, store, adapter = _delivery_fixture()

    result = await bound.deliver_once(delivery_key="final", content="safe answer")

    assert result.outcome == "DELIVERED"
    assert result.transport_attempted is True
    assert adapter.calls == [
        (
            "private-chat-123",
            "safe answer",
            None,
            {
                "thread_id": "topic-7",
                "telegram_dm_topic_reply_fallback": True,
                "direct_messages_topic_id": "topic-7",
            },
        )
    ]
    resolver = store.writes[0].value
    assert list(resolver) and all(
        re.fullmatch(r"[0-9a-f]{64}", key) for key in resolver
    )
    assert all(
        set(value)
        == {"expires_at", "recovery_context_digest", "transport_owner_digest"}
        for value in resolver.values()
    )
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value["recovery_context_digest"])
        for value in resolver.values()
    )
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value["transport_owner_digest"])
        for value in resolver.values()
    )
    serialized = repr(store.writes)
    for secret in (
        "private-chat-123",
        "message-1",
        "safe answer",
        binding.nonce,
        binding.signature,
        "receipt-secret",
    ):
        assert secret not in serialized


@pytest.mark.asyncio
async def test_reused_session_delivers_and_recovers_standalone_in_same_topic():
    bound, _binding, store, adapter = _delivery_fixture(
        message_id="message-2", persisted_message_id="message-1"
    )

    result = await bound.deliver_once(delivery_key="immediate", content="answer-1")

    assert result.outcome == "DELIVERED"
    assert store.entry.origin.message_id == "message-1"
    assert adapter.calls[-1][2] is None

    record = _reserve_restart_pending(store, adapter, handle="second-private-handle")
    counts = await recover_pending(_recovery_runner(store, adapter))

    assert counts == {"pending": 1, "delivered": 1, "failed": 0, "uncertain": 0}
    assert adapter.calls[-1][2] is None
    assert _delivery_record(record.delivery_id).state.value == "DELIVERED"
    assert store.entry.metadata[_METADATA_KEY] == {}
    assert not any("adapter" in name for name in vars(bound))


@pytest.mark.asyncio
async def test_immediate_delivery_preserves_live_shared_credential_transport():
    from gateway.run import GatewayRunner

    primary = Adapter()
    fallback = Adapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_type="dm",
        chat_id="private-chat-123",
        thread_id="topic-7",
        user_id="user-1",
        profile="routed-profile",
    )
    setattr(source, "_transport_adapter_ref", weakref.ref(primary))
    entry = SimpleNamespace(
        session_key="session-key-1",
        session_id="session-1",
        origin=source,
        metadata={},
    )
    store = Store(entry)
    runner = cast(
        Any,
        SimpleNamespace(
            session_store=store,
            _registered_transport_adapter=(
                lambda candidate: (
                    primary
                    if getattr(candidate, "_transport_adapter_ref", lambda: None)()
                    is primary
                    else None
                )
            ),
            _adapter_for_source=lambda candidate: (
                primary
                if getattr(candidate, "_transport_adapter_ref", lambda: None)()
                is primary
                else fallback
            ),
            _thread_metadata_for_target=_target_metadata,
        ),
    )
    destination = GatewayRunner._gateway_delivery_destination(
        runner,
        source=source,
        session_id=entry.session_id,
        session_key=entry.session_key,
    )

    result = await destination(
        "fixture.delivery",
        "private-binding-handle",
        time.time() + 60,
        "final",
        "safe answer",
    )

    assert result.outcome == "DELIVERED"
    assert len(primary.calls) == 1
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_collected_bound_transport_fails_closed_without_fallback():
    from gateway.run import GatewayRunner

    primary = Adapter()
    primary_ref = weakref.ref(primary)
    fallback = Adapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_type="dm",
        chat_id="private-chat-123",
        thread_id="topic-7",
        user_id="user-1",
        profile="routed-profile",
    )
    setattr(source, "_transport_adapter_ref", primary_ref)
    entry = SimpleNamespace(
        session_key="session-key-1",
        session_id="session-1",
        origin=source,
        metadata={},
    )
    store = Store(entry)
    runner = cast(
        Any,
        SimpleNamespace(
            session_store=store,
            _registered_transport_adapter=lambda _candidate: None,
            _adapter_for_source=lambda _candidate: fallback,
            _thread_metadata_for_target=_target_metadata,
        ),
    )
    destination = GatewayRunner._gateway_delivery_destination(
        runner,
        source=source,
        session_id=entry.session_id,
        session_key=entry.session_key,
    )
    del primary
    gc.collect()
    assert primary_ref() is None

    result = await destination(
        "fixture.delivery",
        "private-binding-handle",
        time.time() + 60,
        "final",
        "safe answer",
    )

    assert result.outcome == "FAILED"
    assert result.error_code == "adapter_unavailable"
    assert fallback.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "source_overrides"),
    [
        (
            Platform.DISCORD,
            {"chat_type": "channel", "prospective_thread_id": "future-thread"},
        ),
        (Platform.SLACK, {"chat_type": "channel"}),
        (Platform.RELAY, {"chat_type": "channel"}),
    ],
)
async def test_nontelegram_immediate_delivery_fails_before_reserve_or_adapter(
    _isolated, platform, source_overrides
):
    source = SessionSource(
        platform=platform,
        chat_id="nontelegram-chat",
        user_id="user-1",
        **source_overrides,
    )
    entry = SimpleNamespace(
        session_key="session-key-1",
        session_id="session-1",
        origin=source,
        metadata={},
    )
    store = Store(entry)
    adapter = Adapter()
    adapter_resolutions = []
    runner = SimpleNamespace(
        session_store=store,
        _adapter_for_source=lambda candidate: (
            adapter_resolutions.append(candidate) or adapter
        ),
    )

    result = await deliver_once(
        runner,
        plugin_id="fixture.delivery",
        binding_handle="private-binding-handle",
        binding_expires_at=time.time() + 60,
        session_id=entry.session_id,
        session_key=entry.session_key,
        source=source,
        delivery_key="final",
        content="safe answer",
    )

    assert result.outcome == "FAILED"
    assert result.error_code == "unsupported_platform"
    assert adapter_resolutions == []
    assert adapter.calls == []
    assert store.writes == []
    assert not _isolated.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_outcome", "expected"),
    [
        (SendOnceOutcome.FAILED, "FAILED"),
        (SendOnceOutcome.DELIVERY_UNCERTAIN, "DELIVERY_UNCERTAIN"),
    ],
)
async def test_bound_delivery_maps_terminal_transport_outcomes(
    transport_outcome, expected
):
    bound, _binding, _store, adapter = _delivery_fixture(Adapter(transport_outcome))

    result = await bound.deliver_once(delivery_key="final", content="safe answer")

    assert result.outcome == expected
    assert result.transport_attempted is True
    assert len(adapter.calls) == 1
    assert _delivery_record(result.delivery_id).state.value == expected


@pytest.mark.asyncio
async def test_terminal_ledger_failure_is_uncertain_without_transport_retry(
    monkeypatch,
):
    bound, _binding, _store, adapter = _delivery_fixture()
    monkeypatch.setattr(ledger, "mark_delivered", lambda *_args, **_kwargs: False)

    result = await bound.deliver_once(delivery_key="final", content="safe answer")

    assert result.outcome == "DELIVERY_UNCERTAIN"
    assert result.error_code == "terminal_update_failed"
    assert result.transport_attempted is True
    assert len(adapter.calls) == 1
    assert _delivery_record(result.delivery_id).state.value == "SEND_CLAIMED"


@pytest.mark.asyncio
async def test_revoked_binding_never_reserves_or_calls_transport(_isolated):
    bound, binding, store, adapter = _delivery_fixture()
    revoke_gateway_dispatch_binding(binding)

    with pytest.raises(Exception, match="Unknown gateway dispatch binding"):
        await bound.deliver_once(delivery_key="final", content="safe answer")

    assert adapter.calls == []
    assert store.writes == []
    assert not _isolated.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["chat_id", "platform", "profile"])
async def test_destination_rebinding_never_reserves_or_calls_transport(
    _isolated, changed_field
):
    bound, _binding, store, adapter = _delivery_fixture()
    replacements = {
        "chat_id": "different-chat",
        "platform": Platform.SLACK,
        "profile": "different-profile",
    }
    store.entry.origin = dataclasses.replace(
        store.entry.origin, **{changed_field: replacements[changed_field]}
    )

    result = await bound.deliver_once(delivery_key="final", content="safe answer")

    assert result.outcome == "FAILED"
    assert result.error_code == "session_not_live"
    assert result.transport_attempted is False
    assert adapter.calls == []
    assert store.writes == []
    assert not _isolated.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id", [None, "different-message"])
async def test_stale_message_id_does_not_change_standalone_delivery(message_id):
    bound, _binding, store, adapter = _delivery_fixture()
    store.entry.origin = dataclasses.replace(store.entry.origin, message_id=message_id)

    result = await bound.deliver_once(delivery_key="final", content="safe answer")

    assert result.outcome == "DELIVERED"
    assert result.transport_attempted is True
    assert adapter.calls[0][2] is None


@pytest.mark.asyncio
async def test_resolver_persistence_failure_prevents_reserve_and_transport(
    _isolated, monkeypatch
):
    bound, _binding, store, adapter = _delivery_fixture()
    monkeypatch.setattr(store, "set_session_metadata", lambda *_args: False)

    result = await bound.deliver_once(delivery_key="final", content="safe answer")

    assert result.outcome == "FAILED"
    assert result.error_code == "resolver_persist_failed"
    assert result.transport_attempted is False
    assert adapter.calls == []
    assert not _isolated.exists()


@pytest.mark.asyncio
async def test_simultaneous_delivery_calls_make_one_transport_attempt():
    class BlockingAdapter(Adapter):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def send_once(self, chat_id, content, reply_to=None, metadata=None):
            self.calls.append((chat_id, content, reply_to, metadata))
            self.started.set()
            await self.release.wait()
            return SendOnceResult(SendOnceOutcome.DELIVERED, message_id="receipt")

    adapter = BlockingAdapter()
    bound, _binding, _store, _adapter = _delivery_fixture(adapter)
    first = asyncio.create_task(
        bound.deliver_once(delivery_key="final", content="safe answer")
    )
    await adapter.started.wait()
    second = asyncio.create_task(
        bound.deliver_once(delivery_key="final", content="safe answer")
    )
    await asyncio.sleep(0.05)
    adapter.release.set()

    results = await asyncio.gather(first, second)

    assert len(adapter.calls) == 1
    assert {result.outcome for result in results} <= {"DELIVERED", "IN_PROGRESS"}
    assert any(result.outcome == "DELIVERED" for result in results)


@pytest.mark.asyncio
async def test_cancel_during_reserve_removes_restart_resolver_and_never_sends(
    monkeypatch,
):
    bound, _binding, store, adapter = _delivery_fixture()
    original_reserve = ledger.reserve_delivery
    reserve_started = threading.Event()
    reserve_release = threading.Event()
    reserve_finished = threading.Event()

    def blocked_reserve(**kwargs):
        reserve_started.set()
        reserve_release.wait(timeout=2)
        try:
            return original_reserve(**kwargs)
        finally:
            reserve_finished.set()

    monkeypatch.setattr(ledger, "reserve_delivery", blocked_reserve)
    original_set_metadata = store.set_session_metadata

    def cleanup_fails(session_key, key, value):
        if reserve_started.is_set() and value == {}:
            raise RuntimeError("metadata cleanup unavailable")
        return original_set_metadata(session_key, key, value)

    monkeypatch.setattr(store, "set_session_metadata", cleanup_fails)
    task = asyncio.create_task(
        bound.deliver_once(delivery_key="final", content="safe answer")
    )
    assert await asyncio.to_thread(reserve_started.wait, 2)

    task.cancel()
    reserve_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(reserve_finished.wait, 2)

    assert adapter.calls == []
    assert store.entry.metadata[_METADATA_KEY]
    counts = await recover_pending(_recovery_runner(store, adapter))
    assert counts == {"pending": 0, "delivered": 0, "failed": 0, "uncertain": 0}
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_claim_failure_after_reserve_is_terminal_and_restart_never_sends(
    monkeypatch,
):
    bound, _binding, store, adapter = _delivery_fixture()

    def claim_raises(**_kwargs):
        raise RuntimeError("claim preparation failed")

    monkeypatch.setattr(ledger, "claim_for_send", claim_raises)

    result = await bound.deliver_once(delivery_key="final", content="safe answer")

    assert result.outcome == "FAILED"
    assert result.error_code == "delivery_prepare_failed"
    assert adapter.calls == []
    assert ledger.recover_after_restart() == []
    counts = await recover_pending(_recovery_runner(store, adapter))
    assert counts == {"pending": 0, "delivered": 0, "failed": 0, "uncertain": 0}
    assert adapter.calls == []


def _reserve_restart_pending(
    store,
    adapter,
    *,
    handle="private-binding-handle",
    expires_at=None,
):
    digest = hashlib.sha256(handle.encode()).hexdigest()
    owner_digest = _transport_owner_digest(
        digest, platform=store.entry.origin.platform, adapter=adapter
    )
    context_digest = _recovery_context_digest(
        digest,
        session_id=store.entry.session_id,
        session_key=store.entry.session_key,
        origin=store.entry.origin,
        transport_owner_digest=owner_digest,
    )
    record = ledger.reserve_delivery(
        plugin_id="fixture.delivery",
        binding_handle=handle,
        delivery_key="final",
        sanitized_content="restart answer",
        recovery_context_digest=context_digest,
    )
    store.entry.metadata[_METADATA_KEY] = {
        digest: {
            "expires_at": expires_at or (time.time() + 60),
            "recovery_context_digest": context_digest,
            "transport_owner_digest": owner_digest,
        }
    }
    return record


@pytest.mark.asyncio
@pytest.mark.parametrize("registry_mode", ["exact", "missing", "duplicate"])
async def test_restart_resolves_exact_transport_owner_without_routed_profile_fallback(
    registry_mode,
):
    _bound, _binding, store, ingress = _delivery_fixture()
    setattr(ingress, "_owner_profile", None)
    fallback = Adapter()
    setattr(fallback, "_owner_profile", "routed-profile")
    store.entry.origin = dataclasses.replace(
        store.entry.origin, profile="routed-profile"
    )
    record = _reserve_restart_pending(store, ingress)
    assert "routed-profile" not in repr(store.entry.metadata[_METADATA_KEY])
    store.entry.origin = SessionSource.from_dict(store.entry.origin.to_dict())

    adapters = {}
    profile_adapters = {
        "routed-profile": {Platform.TELEGRAM: fallback},
    }
    duplicate = None
    if registry_mode != "missing":
        adapters[Platform.TELEGRAM] = ingress
    if registry_mode == "duplicate":
        duplicate = Adapter()
        setattr(duplicate, "_owner_profile", None)
        profile_adapters["duplicate"] = {Platform.TELEGRAM: duplicate}

    counts = await recover_pending(
        _recovery_runner(
            store,
            fallback,
            adapters=adapters,
            profile_adapters=profile_adapters,
        )
    )

    expected_delivered = 1 if registry_mode == "exact" else 0
    assert counts == {
        "pending": 1,
        "delivered": expected_delivered,
        "failed": 1 - expected_delivered,
        "uncertain": 0,
    }
    assert len(ingress.calls) == expected_delivered
    assert fallback.calls == []
    if duplicate is not None:
        assert duplicate.calls == []
    assert _delivery_record(record.delivery_id).state.value == (
        "DELIVERED" if registry_mode == "exact" else "FAILED"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.DISCORD, Platform.SLACK, Platform.RELAY])
async def test_nontelegram_recovery_is_terminal_without_transport(platform):
    adapter = Adapter()
    source = SessionSource(
        platform=platform,
        chat_type="channel",
        chat_id="nontelegram-chat",
        user_id="user-1",
    )
    entry = SimpleNamespace(
        session_key="session-key-1",
        session_id="session-1",
        origin=source,
        metadata={},
    )
    store = Store(entry)
    record = _reserve_restart_pending(store, adapter)

    counts = await recover_pending(
        _recovery_runner(
            store,
            adapter,
            adapters={platform: adapter},
        )
    )

    assert counts == {"pending": 1, "delivered": 0, "failed": 1, "uncertain": 0}
    assert adapter.calls == []
    assert _delivery_record(record.delivery_id).state.value == "FAILED"


@pytest.mark.asyncio
async def test_restart_recovers_pending_once_through_session_origin():
    _bound, _binding, store, adapter = _delivery_fixture()
    record = _reserve_restart_pending(store, adapter)

    counts = await recover_pending(_recovery_runner(store, adapter))

    assert counts == {"pending": 1, "delivered": 1, "failed": 0, "uncertain": 0}
    assert adapter.calls == [
        (
            "private-chat-123",
            "restart answer",
            None,
            {
                "thread_id": "topic-7",
                "telegram_dm_topic_reply_fallback": True,
                "direct_messages_topic_id": "topic-7",
            },
        )
    ]
    assert _delivery_record(record.delivery_id).state.value == "DELIVERED"
    assert store.entry.metadata[_METADATA_KEY] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("spoof", ["changed", "missing"])
async def test_restart_ignores_stale_or_missing_message_id(spoof):
    _bound, _binding, store, adapter = _delivery_fixture()
    record = _reserve_restart_pending(store, adapter)
    if spoof == "changed":
        store.entry.origin = dataclasses.replace(
            store.entry.origin, message_id="message-spoof"
        )
    else:
        store.entry.origin = dataclasses.replace(store.entry.origin, message_id=None)

    counts = await recover_pending(_recovery_runner(store, adapter))

    assert counts == {"pending": 1, "delivered": 1, "failed": 0, "uncertain": 0}
    assert adapter.calls[0][2] is None
    assert _delivery_record(record.delivery_id).state.value == "DELIVERED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spoof",
    ["session", "chat", "topic", "user", "profile"],
)
async def test_restart_rejects_resolver_moved_to_wrong_session_context(spoof):
    _bound, _binding, store, adapter = _delivery_fixture()
    record = _reserve_restart_pending(store, adapter)
    origin_changes = {
        "session": {},
        "chat": {"chat_id": "private-chat-spoof"},
        "topic": {"thread_id": "topic-spoof"},
        "user": {"user_id": "user-spoof"},
        "profile": {"profile": "profile-spoof"},
    }
    if spoof == "session":
        moved_metadata = store.entry.metadata.pop(_METADATA_KEY)
        wrong_entry = SimpleNamespace(
            session_key="session-key-spoof",
            session_id="session-spoof",
            origin=store.entry.origin,
            metadata={_METADATA_KEY: moved_metadata},
        )
        store.entries.append(wrong_entry)
    else:
        wrong_entry = store.entry
        wrong_entry.origin = dataclasses.replace(
            wrong_entry.origin, **origin_changes[spoof]
        )

    counts = await recover_pending(_recovery_runner(store, adapter))

    assert counts == {"pending": 1, "delivered": 0, "failed": 1, "uncertain": 0}
    assert adapter.calls == []
    assert _delivery_record(record.delivery_id).state.value == "FAILED"
    assert wrong_entry.metadata[_METADATA_KEY] == {}


@pytest.mark.asyncio
async def test_restart_quarantines_claimed_without_transport():
    _bound, _binding, store, adapter = _delivery_fixture()
    record = _reserve_restart_pending(store, adapter)
    claim = ledger.claim_for_send(
        plugin_id="fixture.delivery",
        binding_handle="private-binding-handle",
        delivery_key="final",
    )
    assert claim is not None

    counts = await recover_pending(_recovery_runner(store, adapter))

    assert counts["pending"] == 0
    assert adapter.calls == []
    assert _delivery_record(record.delivery_id).state.value == "DELIVERY_UNCERTAIN"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unknown", "expired", "ambiguous", "adapter_missing"])
async def test_restart_resolver_failures_are_terminal_without_transport(mode):
    _bound, _binding, store, adapter = _delivery_fixture()
    expires_at = time.time() - 1 if mode == "expired" else None
    record = _reserve_restart_pending(store, adapter, expires_at=expires_at)
    if mode == "unknown":
        store.entry.metadata[_METADATA_KEY] = {}
    elif mode == "ambiguous":
        duplicate = SimpleNamespace(
            session_key=store.entry.session_key,
            session_id=store.entry.session_id,
            origin=store.entry.origin,
            metadata={_METADATA_KEY: dict(store.entry.metadata[_METADATA_KEY])},
        )
        store.entries.append(duplicate)

    counts = await recover_pending(
        _recovery_runner(
            store,
            adapter,
            adapters={} if mode == "adapter_missing" else None,
        )
    )

    assert counts == {"pending": 1, "delivered": 0, "failed": 1, "uncertain": 0}
    assert adapter.calls == []
    assert _delivery_record(record.delivery_id).state.value == "FAILED"


@pytest.mark.asyncio
async def test_transport_exception_is_uncertain_and_logs_no_sensitive_data(caplog):
    class RaisingAdapter(Adapter):
        async def send_once(self, chat_id, content, reply_to=None, metadata=None):
            self.calls.append((chat_id, content, reply_to, metadata))
            raise RuntimeError("sensitive-content private-chat-123 claim-token")

    bound, _binding, _store, adapter = _delivery_fixture(RaisingAdapter())
    with caplog.at_level(logging.INFO, logger="gateway.plugin_delivery"):
        result = await bound.deliver_once(
            delivery_key="final", content="sensitive-content"
        )

    assert result.outcome == "DELIVERY_UNCERTAIN"
    assert len(adapter.calls) == 1
    for secret in ("sensitive-content", "private-chat-123", "claim-token"):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_gateway_startup_recovery_runs_once(monkeypatch):
    from gateway.run import GatewayRunner

    calls = []

    async def recover(_runner):
        calls.append(True)
        return {"pending": 1, "delivered": 1, "failed": 0, "uncertain": 0}

    monkeypatch.setattr("gateway.plugin_delivery_service.recover_pending", recover)
    runner = GatewayRunner.__new__(GatewayRunner)

    first = await runner._recover_gateway_plugin_deliveries_once()
    second = await runner._recover_gateway_plugin_deliveries_once()

    assert first["delivered"] == 1
    assert second == {"pending": 0, "delivered": 0, "failed": 0, "uncertain": 0}
    assert calls == [True]
