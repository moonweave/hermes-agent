from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from agent.route_capability import (
    GatewayDispatchDecision,
    issue_gateway_dispatch_binding,
)
from gateway.config import Platform
from gateway.run import (
    GatewayRunner,
    _gateway_dispatch_eligible,
    _log_gateway_inbound_metadata,
)
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _telegram_source(*, thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
        user_name="tester",
        thread_id=thread_id,
    )


def _task_binding(nonce: str = "a" * 32):
    del nonce
    return issue_gateway_dispatch_binding(
        issuer_plugin_id="fixture-dispatch",
        parent_session_id="session-1",
        parent_turn_id="gateway-turn-1",
        user_message="question",
        platform="telegram",
        message_id="message-1",
        parent=SimpleNamespace(session_id="session-1"),
        schedule=lambda *_args: "unused",
        validity_resolver=lambda: True,
    )


def test_inbound_log_omits_message_and_reply_text(caplog):
    event = SimpleNamespace(
        text="내 계좌 1234와 민감한 질문",
        reply_to_text="비밀 답장 내용",
        reply_to_message_id="reply-1",
    )
    source = _telegram_source()

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _log_gateway_inbound_metadata(event, source)

    assert "내 계좌 1234와 민감한 질문" not in caplog.text
    assert "비밀 답장 내용" not in caplog.text
    assert "message_sha256=" in caplog.text
    assert "message_chars=" in caplog.text


def test_dispatch_eligibility_rejects_slash_and_internal_events():
    source = _telegram_source()
    assert not _gateway_dispatch_eligible(MessageEvent(text="/status", source=source))
    assert not _gateway_dispatch_eligible(
        MessageEvent(text="background wake", source=source, internal=True)
    )
    assert _gateway_dispatch_eligible(
        MessageEvent(text="market question", source=source)
    )


@pytest.mark.asyncio
async def test_dispatch_gets_stable_binding_and_host_owned_scheduler(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    setattr(
        runner,
        "_is_session_run_current",
        lambda session_key, generation: (
            session_key == "telegram:user:chat" and generation == 7
        ),
    )
    captured = {}
    finished = asyncio.Event()
    parent_resolver_calls = 0

    async def work():
        finished.set()

    def invoke(**kwargs):
        from agent.route_capability import activate_gateway_dispatch_binding

        captured.update(kwargs)
        binding = _task_binding()
        task_id = kwargs["schedule"](lambda: work(), "consultation", binding)
        assert task_id.startswith("gateway-plugin-")
        assert kwargs["binding_validity"]() is True
        activate_gateway_dispatch_binding(binding)
        return GatewayDispatchDecision(action="handled", response="accepted")

    monkeypatch.setattr("hermes_cli.plugins.invoke_gateway_pre_agent_dispatch", invoke)
    source = _telegram_source()

    def slow_parent_constructor():
        nonlocal parent_resolver_calls
        parent_resolver_calls += 1
        time.sleep(5)
        return object()

    started = time.monotonic()
    decision = runner._invoke_gateway_dispatch_before_agent(
        message="question",
        source=source,
        session_id="session-1",
        session_key="telegram:user:chat",
        message_id="message-1",
        parent_resolver=slow_parent_constructor,
        loop=asyncio.get_running_loop(),
        binding_validity=lambda: True,
    )
    await asyncio.wait_for(finished.wait(), timeout=1)

    assert time.monotonic() - started < 1
    assert parent_resolver_calls == 0
    assert decision == GatewayDispatchDecision(action="handled", response="accepted")
    assert captured["parent_turn_id"] == "gateway:session-1:message-1"
    assert captured["message"] == "question"


@pytest.mark.asyncio
async def test_real_scheduler_propagates_factory_failure_before_return():
    from agent.route_capability import activate_gateway_dispatch_binding

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    binding = _task_binding()

    runner._schedule_gateway_plugin_task(
        asyncio.get_running_loop(),
        lambda: (_ for _ in ()).throw(RuntimeError("factory failed")),
        "consultation",
        binding,
    )
    with pytest.raises(RuntimeError, match="factory failed"):
        activate_gateway_dispatch_binding(binding)

    assert runner._gateway_plugin_task_status("session-1") == (0, 0)


@pytest.mark.asyncio
async def test_real_scheduler_propagates_task_creation_failure(monkeypatch):
    from agent.route_capability import activate_gateway_dispatch_binding

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    binding = _task_binding()

    async def work():
        return None

    monkeypatch.setattr(
        asyncio,
        "ensure_future",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("task creation failed")
        ),
    )
    runner._schedule_gateway_plugin_task(
        asyncio.get_running_loop(), work, "consultation", binding
    )
    with pytest.raises(RuntimeError, match="task creation failed"):
        activate_gateway_dispatch_binding(binding)

    assert runner._gateway_plugin_task_status("session-1") == (0, 0)


@pytest.mark.asyncio
async def test_cross_thread_scheduler_timeout_abandons_slow_admission():
    from agent.route_capability import activate_gateway_dispatch_binding

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    loop = asyncio.get_running_loop()
    loop_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    work_ran = threading.Event()

    async def work():
        work_ran.set()

    def slow_factory():
        time.sleep(3)
        return work()

    binding = _task_binding()
    runner._schedule_gateway_plugin_task(
        loop,
        slow_factory,
        "consultation",
        binding,
    )
    try:
        with pytest.raises(TimeoutError, match="acknowledgement timed out"):
            await asyncio.to_thread(
                activate_gateway_dispatch_binding,
                binding,
            )
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not work_ran.is_set()
    assert runner._gateway_plugin_task_status("session-1") == (0, 0)
    assert loop_errors == []


@pytest.mark.asyncio
async def test_cross_thread_scheduler_timeout_makes_dispatch_fail_open(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import (
        LoadedPlugin,
        PluginContext,
        PluginManager,
        PluginManifest,
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    loop = asyncio.get_running_loop()
    loop_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    work_ran = threading.Event()
    manifest = PluginManifest(
        name="fixture-dispatch",
        key="fixture-dispatch",
        capabilities=["gateway.message_dispatch"],
    )
    manager = PluginManager()
    manager._plugins[manifest.key] = LoadedPlugin(manifest=manifest, enabled=True)
    plugin_context = PluginContext(manifest, manager)
    monkeypatch.setattr(plugins_mod, "plugin_capability_granted", lambda *_: True)

    async def work():
        work_ran.set()

    def dispatch(*, context):
        def slow_factory():
            time.sleep(3)
            return work()

        plugin_context.gateway_tasks.for_gateway_binding(context.binding).spawn(
            slow_factory,
            name="consultation",
        )
        return {"action": "handled"}

    plugin_context.register_hook("gateway_pre_agent_dispatch", dispatch)
    monkeypatch.setattr(
        plugins_mod,
        "invoke_gateway_pre_agent_dispatch",
        manager.invoke_gateway_pre_agent_dispatch,
    )

    try:
        decision = await asyncio.to_thread(
            runner._invoke_gateway_dispatch_before_agent,
            message="question",
            source=_telegram_source(),
            session_id="session-1",
            session_key="telegram:user:chat",
            message_id="message-1",
            parent_resolver=lambda: SimpleNamespace(session_id="session-1"),
            loop=loop,
            binding_validity=lambda: True,
        )
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert decision == GatewayDispatchDecision()
    assert not work_ran.is_set()
    assert runner._gateway_plugin_task_status("session-1") == (0, 0)
    assert loop_errors == []


@pytest.mark.asyncio
async def test_cross_thread_scheduled_work_waits_for_handled_decision(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import (
        LoadedPlugin,
        PluginContext,
        PluginManager,
        PluginManifest,
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    loop = asyncio.get_running_loop()
    work_ran = threading.Event()
    manifest = PluginManifest(
        name="fixture-dispatch",
        key="fixture-dispatch",
        capabilities=["gateway.message_dispatch"],
    )
    manager = PluginManager()
    manager._plugins[manifest.key] = LoadedPlugin(manifest=manifest, enabled=True)
    plugin_context = PluginContext(manifest, manager)
    monkeypatch.setattr(plugins_mod, "plugin_capability_granted", lambda *_: True)

    async def work():
        work_ran.set()

    def dispatch(*, context):
        plugin_context.gateway_tasks.for_gateway_binding(context.binding).spawn(
            work,
            name="consultation",
        )
        time.sleep(0.2)
        return {"action": "allow"}

    plugin_context.register_hook("gateway_pre_agent_dispatch", dispatch)
    monkeypatch.setattr(
        plugins_mod,
        "invoke_gateway_pre_agent_dispatch",
        manager.invoke_gateway_pre_agent_dispatch,
    )

    decision = await asyncio.to_thread(
        runner._invoke_gateway_dispatch_before_agent,
        message="question",
        source=_telegram_source(),
        session_id="session-1",
        session_key="telegram:user:chat",
        message_id="message-1",
        parent_resolver=lambda: SimpleNamespace(session_id="session-1"),
        loop=loop,
        binding_validity=lambda: True,
    )
    await asyncio.sleep(0)

    assert decision == GatewayDispatchDecision()
    assert not work_ran.is_set()
    assert runner._gateway_plugin_task_status("session-1") == (0, 0)


@pytest.mark.asyncio
async def test_factory_is_staged_until_handled_decision_commit(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import (
        LoadedPlugin,
        PluginContext,
        PluginManager,
        PluginManifest,
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    loop = asyncio.get_running_loop()
    scheduled = threading.Event()
    release_decision = threading.Event()
    work_ran = asyncio.Event()
    factory_calls = 0
    manifest = PluginManifest(
        name="fixture-dispatch",
        key="fixture-dispatch",
        capabilities=["gateway.message_dispatch"],
    )
    manager = PluginManager()
    manager._plugins[manifest.key] = LoadedPlugin(manifest=manifest, enabled=True)
    plugin_context = PluginContext(manifest, manager)
    monkeypatch.setattr(plugins_mod, "plugin_capability_granted", lambda *_: True)

    async def work():
        work_ran.set()

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return work()

    def dispatch(*, context):
        plugin_context.gateway_tasks.for_gateway_binding(context.binding).spawn(
            factory,
            name="consultation",
        )
        scheduled.set()
        assert release_decision.wait(timeout=2)
        return {"action": "handled"}

    plugin_context.register_hook("gateway_pre_agent_dispatch", dispatch)
    monkeypatch.setattr(
        plugins_mod,
        "invoke_gateway_pre_agent_dispatch",
        manager.invoke_gateway_pre_agent_dispatch,
    )
    invocation = asyncio.create_task(
        asyncio.to_thread(
            runner._invoke_gateway_dispatch_before_agent,
            message="question",
            source=_telegram_source(),
            session_id="session-1",
            session_key="telegram:user:chat",
            message_id="message-1",
            parent_resolver=lambda: SimpleNamespace(session_id="session-1"),
            loop=loop,
            binding_validity=lambda: True,
        )
    )

    assert await asyncio.to_thread(scheduled.wait, 1)
    await asyncio.sleep(0)
    assert factory_calls == 0
    assert not work_ran.is_set()

    release_decision.set()
    decision = await asyncio.wait_for(invocation, timeout=1)
    await asyncio.wait_for(work_ran.wait(), timeout=1)

    assert decision == GatewayDispatchDecision(action="handled")
    assert factory_calls == 1


@pytest.mark.asyncio
async def test_real_scheduler_start_failure_makes_dispatch_allow(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import (
        LoadedPlugin,
        PluginContext,
        PluginManager,
        PluginManifest,
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    manifest = PluginManifest(
        name="fixture-dispatch",
        key="fixture-dispatch",
        capabilities=["gateway.message_dispatch"],
    )
    manager = PluginManager()
    manager._plugins[manifest.key] = LoadedPlugin(manifest=manifest, enabled=True)
    context = PluginContext(manifest, manager)
    monkeypatch.setattr(plugins_mod, "plugin_capability_granted", lambda *_: True)

    def dispatch(*, context: object):
        binding = context.binding  # type: ignore[attr-defined]
        tasks = PluginContext(manifest, manager).gateway_tasks.for_gateway_binding(
            binding
        )
        tasks.spawn(
            lambda: (_ for _ in ()).throw(RuntimeError("factory failed")),
            name="consultation",
        )
        return {"action": "handled"}

    context.register_hook("gateway_pre_agent_dispatch", dispatch)
    monkeypatch.setattr(
        plugins_mod,
        "invoke_gateway_pre_agent_dispatch",
        manager.invoke_gateway_pre_agent_dispatch,
    )

    decision = runner._invoke_gateway_dispatch_before_agent(
        message="question",
        source=_telegram_source(),
        session_id="session-1",
        session_key="telegram:user:chat",
        message_id="message-1",
        parent_resolver=lambda: SimpleNamespace(
            session_id="session-1", _delegate_depth=0, _subagent_id=None
        ),
        loop=asyncio.get_running_loop(),
        binding_validity=lambda: True,
    )

    assert decision == GatewayDispatchDecision()
    assert runner._gateway_plugin_task_status("session-1") == (0, 0)


@pytest.mark.parametrize("outcome", ["allow", "malformed", "error"])
@pytest.mark.asyncio
async def test_dispatch_revocation_cancels_work_scheduled_before_fail_open(
    monkeypatch, outcome
):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import (
        LoadedPlugin,
        PluginContext,
        PluginManager,
        PluginManifest,
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    manifest = PluginManifest(
        name="fixture-dispatch",
        key="fixture-dispatch",
        capabilities=["gateway.message_dispatch"],
    )
    manager = PluginManager()
    manager._plugins[manifest.key] = LoadedPlugin(manifest=manifest, enabled=True)
    plugin_context = PluginContext(manifest, manager)
    monkeypatch.setattr(plugins_mod, "plugin_capability_granted", lambda *_: True)
    work_ran = asyncio.Event()
    factory_calls = 0

    async def work():
        work_ran.set()

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return work()

    def dispatch(*, context):
        tasks = plugin_context.gateway_tasks.for_gateway_binding(context.binding)
        tasks.spawn(factory, name="consultation")
        if outcome == "error":
            raise RuntimeError("dispatch failed")
        if outcome == "malformed":
            return {"action": "invented"}
        return {"action": "allow"}

    plugin_context.register_hook("gateway_pre_agent_dispatch", dispatch)
    monkeypatch.setattr(
        plugins_mod,
        "invoke_gateway_pre_agent_dispatch",
        manager.invoke_gateway_pre_agent_dispatch,
    )

    decision = runner._invoke_gateway_dispatch_before_agent(
        message="question",
        source=_telegram_source(),
        session_id="session-1",
        session_key="telegram:user:chat",
        message_id="message-1",
        parent_resolver=lambda: SimpleNamespace(
            session_id="session-1", _delegate_depth=0, _subagent_id=None
        ),
        loop=asyncio.get_running_loop(),
        binding_validity=lambda: True,
    )
    await asyncio.sleep(0)

    assert decision == GatewayDispatchDecision()
    assert factory_calls == 0
    assert not work_ran.is_set()
    assert runner._gateway_plugin_task_status("session-1") == (0, 0)


@pytest.mark.asyncio
async def test_gateway_task_registry_bounds_queues_and_cancels_both_slots():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    release = asyncio.Event()
    pending_started = asyncio.Event()

    async def active():
        await release.wait()

    async def pending():
        pending_started.set()

    loop = asyncio.get_running_loop()
    first = runner._schedule_gateway_plugin_task(
        loop, active, "first", _task_binding("a" * 32)
    )
    second = runner._schedule_gateway_plugin_task(
        loop, pending, "second", _task_binding("b" * 32)
    )
    assert first != second
    assert runner._gateway_plugin_task_status("session-1") == (1, 1)

    with pytest.raises(RuntimeError, match="active and pending"):
        runner._schedule_gateway_plugin_task(
            loop, pending, "third", _task_binding("c" * 32)
        )

    assert runner._cancel_gateway_plugin_tasks("session-1") == (1, 1)
    await asyncio.sleep(0)
    assert not pending_started.is_set()
    assert runner._gateway_plugin_task_status("session-1") == (0, 0)


@pytest.mark.asyncio
async def test_cold_stop_cancels_detached_tasks_before_observer(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    runner._running_agents = {}
    runner.adapters = {}
    setattr(
        runner,
        "_is_user_authorized",
        lambda source, *, allow_adapter_delegation=True: True,
    )
    source = _telegram_source(thread_id="topic-7")
    entry = SimpleNamespace(session_key="session-key-1", session_id="session-1")

    class Store:
        def get_or_create_session(self, _source):
            return entry

    setattr(runner, "session_store", Store())
    release = asyncio.Event()

    async def active():
        await release.wait()

    runner._schedule_gateway_plugin_task(
        asyncio.get_running_loop(), active, "consultation", _task_binding()
    )
    observed = []

    def observer(**payload):
        observed.append((
            payload["outcome"],
            runner._gateway_plugin_task_status("session-1"),
        ))

    monkeypatch.setattr("hermes_cli.plugins.notify_gateway_stop_observers", observer)
    event = MessageEvent(text="/stop", source=source)

    await runner._handle_stop_command(event)
    await asyncio.sleep(0)

    assert observed == [("no_active", (0, 0))]


@pytest.mark.asyncio
async def test_cold_stop_revokes_retained_binding_after_task_completed(monkeypatch):
    from agent.route_capability import GatewayTaskService

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    runner._running_agents = {}
    runner.adapters = {}
    setattr(
        runner,
        "_is_user_authorized",
        lambda source, *, allow_adapter_delegation=True: True,
    )
    source = _telegram_source(thread_id="topic-7")
    entry = SimpleNamespace(session_key="session-key-1", session_id="session-1")

    class Store:
        def get_or_create_session(self, _source):
            return entry

    setattr(runner, "session_store", Store())
    binding = _task_binding()
    completed = asyncio.Event()

    async def work():
        completed.set()

    runner._schedule_gateway_plugin_task(
        asyncio.get_running_loop(), work, "consultation", binding
    )
    from agent.route_capability import activate_gateway_dispatch_binding

    activate_gateway_dispatch_binding(binding)
    await asyncio.wait_for(completed.wait(), timeout=1)
    await asyncio.sleep(0)
    assert runner._gateway_plugin_task_status("session-1") == (0, 0)

    bound_tasks = GatewayTaskService(
        "fixture-dispatch", authorization_resolver=lambda: True
    ).for_gateway_binding(binding)
    monkeypatch.setattr(
        "hermes_cli.plugins.notify_gateway_stop_observers", lambda **_payload: None
    )
    await runner._handle_stop_command(MessageEvent(text="/stop", source=source))

    with pytest.raises(Exception, match="Unknown gateway dispatch binding"):
        bound_tasks.spawn(work, name="after-stop")


@pytest.mark.parametrize("run_state", ["active", "pending"])
@pytest.mark.asyncio
async def test_active_and_pending_stop_revoke_retained_session_binding(
    monkeypatch, run_state
):
    from agent.route_capability import GatewayTaskService
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    runner.adapters = {}
    source = _telegram_source(thread_id="topic-7")
    entry = SimpleNamespace(session_key="session-key-1", session_id="session-1")

    class Store:
        def get_or_create_session(self, _source):
            return entry

    setattr(runner, "session_store", Store())
    runner._running_agents = {
        entry.session_key: (
            object() if run_state == "active" else _AGENT_PENDING_SENTINEL
        )
    }
    runner._interrupt_and_clear_session = AsyncMock()
    binding = _task_binding()
    bound_tasks = GatewayTaskService(
        "fixture-dispatch", authorization_resolver=lambda: True
    ).for_gateway_binding(binding)
    monkeypatch.setattr(
        "hermes_cli.plugins.notify_gateway_stop_observers", lambda **_payload: None
    )

    await runner._handle_stop_command(MessageEvent(text="/stop", source=source))

    with pytest.raises(Exception, match="Unknown gateway dispatch binding"):
        bound_tasks.spawn(lambda: asyncio.sleep(0), name="after-stop")


@pytest.mark.asyncio
async def test_sibling_stop_revokes_retained_sibling_session_binding(monkeypatch):
    from agent.route_capability import (
        GatewayTaskService,
        activate_gateway_dispatch_binding,
        issue_gateway_dispatch_binding,
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    runner.adapters = {}
    runner._running_agents = {}
    setattr(
        runner,
        "_is_user_authorized",
        lambda source, *, allow_adapter_delegation=True: True,
    )
    runner._interrupt_and_clear_session = AsyncMock()
    source = _telegram_source(thread_id="topic-7")
    own_entry = SimpleNamespace(
        session_key="agent:main:telegram:dm:chat-1:topic-7:user-1",
        session_id="session-1",
    )
    sibling_entry = SimpleNamespace(
        session_key="agent:main:telegram:dm:chat-1:topic-7:user-2",
        session_id="session-sibling",
    )

    class Store:
        def get_or_create_session(self, _source):
            return own_entry

        def lookup_by_session_key(self, _session_key):
            raise RuntimeError("detached sibling is not available in the store")

    setattr(runner, "session_store", Store())
    binding = issue_gateway_dispatch_binding(
        issuer_plugin_id="fixture-dispatch",
        parent_session_id="session-sibling",
        parent_turn_id="gateway-turn-sibling",
        user_message="question",
        platform="telegram",
        message_id="message-sibling",
        parent=SimpleNamespace(session_id="session-sibling"),
        schedule=lambda *_args: "unused",
        validity_resolver=lambda: True,
    )
    bound_tasks = GatewayTaskService(
        "fixture-dispatch", authorization_resolver=lambda: True
    ).for_gateway_binding(binding)
    sibling_source = dataclasses.replace(source, user_id="user-2")
    runner._cache_session_source(sibling_entry.session_key, sibling_source)
    runner._register_gateway_detached_session(
        sibling_entry.session_id,
        sibling_entry.session_key,
    )
    release = asyncio.Event()

    async def active():
        await release.wait()

    runner._schedule_gateway_plugin_task(
        asyncio.get_running_loop(), active, "sibling-consultation", binding
    )
    runner._schedule_gateway_plugin_task(
        asyncio.get_running_loop(), active, "sibling-followup", binding
    )
    activate_gateway_dispatch_binding(binding)
    await asyncio.sleep(0)
    task_records = tuple(runner._gateway_plugin_tasks.values())
    active_record = next(record for record in task_records if record.state == "active")
    pending_record = next(
        record for record in task_records if record.state == "pending"
    )
    assert runner._gateway_plugin_task_status("session-sibling") == (1, 1)
    monkeypatch.setattr(
        "hermes_cli.plugins.notify_gateway_stop_observers", lambda **_payload: None
    )

    await runner._handle_stop_command(MessageEvent(text="/stop", source=source))

    await asyncio.sleep(0)
    assert runner._gateway_plugin_task_status("session-sibling") == (0, 0)
    assert all(record.suppress_result for record in task_records)
    assert active_record.task is not None and active_record.task.cancelled()
    assert pending_record.task is None
    with pytest.raises(Exception, match="Unknown gateway dispatch binding"):
        bound_tasks.spawn(lambda: asyncio.sleep(0), name="after-stop")


@pytest.mark.asyncio
async def test_handled_dispatch_bypasses_agent_and_transcript(
    monkeypatch, tmp_path, caplog
):
    from hermes_state import SessionDB
    from tests.gateway.test_first_turn_session_meta_rebaseline import (
        SESSION_ID,
        SESSION_KEY,
        _bootstrap,
        _event,
        _live_count,
    )

    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session(SESSION_ID, source="telegram")
    runner = _bootstrap(monkeypatch, tmp_path, db)
    constructor_calls = 0

    def unexpected_constructor(*_args, **_kwargs):
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError("handled dispatch must not construct AIAgent")

    monkeypatch.setattr("run_agent.AIAgent", unexpected_constructor)
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("handled dispatch must not construct an agent")
    )
    dispatch_call = {}

    def handled_dispatch(**kwargs):
        dispatch_call.update(kwargs)
        return GatewayDispatchDecision(action="handled", response="accepted")

    runner._invoke_gateway_dispatch_before_agent = handled_dispatch
    setattr(runner, "_claim_active_session_slot", pytest.fail)
    setattr(runner, "_persist_active_agents", pytest.fail)
    setattr(runner, "_begin_session_run_generation", pytest.fail)

    event = _event()
    event.text = "민감 계좌 질문 1234"
    event.source = dataclasses.replace(event.source, thread_id="topic-7")
    event.reply_to_message_id = "older-reply-anchor"
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        response = await runner._handle_message(event)

    assert response == "accepted"
    assert constructor_calls == 0
    cast(AsyncMock, runner._run_agent).assert_not_awaited()
    cast(Any, runner.session_store.load_transcript).assert_not_called()
    assert _live_count(db, SESSION_ID) == 0
    cast(AsyncMock, runner.hooks.emit).assert_not_awaited()
    assert dispatch_call["message_id"] == "msg-1"
    assert runner._gateway_detached_session_keys[SESSION_ID] == SESSION_KEY
    assert "민감 계좌 질문 1234" not in caplog.text
