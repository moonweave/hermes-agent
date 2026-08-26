from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from agent.route_capability import GatewayDispatchDecision
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import (
    LoadedPlugin,
    PluginContext,
    PluginManager,
    PluginManifest,
)


def _loaded_manager(plugin_id: str = "fixture-dispatch"):
    manifest = PluginManifest(
        name=plugin_id,
        key=plugin_id,
        capabilities=["gateway.message_dispatch", "gateway.control_observer"],
    )
    manager = PluginManager()
    manager._plugins[plugin_id] = LoadedPlugin(manifest=manifest, enabled=True)
    return manager, PluginContext(manifest, manager)


def _invoke(manager: PluginManager):
    parent = SimpleNamespace(
        session_id="session-1", _delegate_depth=0, _subagent_id=None
    )
    return manager.invoke_gateway_pre_agent_dispatch(
        message="sensitive portfolio question",
        platform="telegram",
        session_id="session-1",
        parent_turn_id="gateway-turn-1",
        message_id="message-1",
        parent=parent,
        schedule=lambda _factory, _name, _binding: "task-1",
        binding_validity=lambda: True,
    )


def test_dispatch_is_single_owner_and_fail_open(monkeypatch):
    manager, context = _loaded_manager()
    monkeypatch.setattr(plugins_mod, "plugin_capability_granted", lambda *_: True)
    seen = []

    context.register_hook(
        "gateway_pre_agent_dispatch",
        lambda *, context: (
            seen.append(context) or {"action": "handled", "response": "accepted"}
        ),
    )

    decision = _invoke(manager)

    assert decision == GatewayDispatchDecision(action="handled", response="accepted")
    assert len(seen) == 1
    assert seen[0].message == "sensitive portfolio question"
    assert "sensitive portfolio question" not in repr(seen[0])

    other_manifest = PluginManifest(
        name="other", key="other", capabilities=["gateway.message_dispatch"]
    )
    manager._plugins["other"] = LoadedPlugin(manifest=other_manifest, enabled=True)
    other = PluginContext(other_manifest, manager)
    with pytest.raises(ValueError, match="different owner"):
        other.register_hook("gateway_pre_agent_dispatch", lambda **_: None)


def test_gateway_hooks_require_declared_and_live_grants(monkeypatch):
    manager, context = _loaded_manager()
    monkeypatch.setattr(plugins_mod, "plugin_capability_granted", lambda *_: False)
    with pytest.raises(PermissionError, match="live grant"):
        context.register_hook("gateway_pre_agent_dispatch", lambda **_: None)

    undeclared = PluginContext(
        PluginManifest(name="plain", key="plain", capabilities=[]), manager
    )
    with pytest.raises(PermissionError, match="must declare"):
        undeclared.register_hook("gateway_status_contributor", lambda **_: "x")


def test_dispatch_error_and_malformed_result_allow(monkeypatch, caplog):
    monkeypatch.setattr(plugins_mod, "plugin_capability_granted", lambda *_: True)

    manager, context = _loaded_manager()
    context.register_hook(
        "gateway_pre_agent_dispatch",
        lambda **_: (_ for _ in ()).throw(RuntimeError("sensitive portfolio question")),
    )
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        assert _invoke(manager) == GatewayDispatchDecision()
    assert "sensitive portfolio question" not in caplog.text

    manager, context = _loaded_manager()
    context.register_hook(
        "gateway_pre_agent_dispatch", lambda **_: {"action": "invented"}
    )
    assert _invoke(manager) == GatewayDispatchDecision()


def test_gateway_task_schedule_failure_allows_normal_agent(monkeypatch):
    monkeypatch.setattr(plugins_mod, "plugin_capability_granted", lambda *_: True)
    manager, plugin_context = _loaded_manager()

    def dispatch(*, context):
        tasks = plugin_context.gateway_tasks.for_gateway_binding(context.binding)
        tasks.spawn(lambda: None, name="consultation")  # type: ignore[arg-type]
        return {"action": "handled"}

    plugin_context.register_hook("gateway_pre_agent_dispatch", dispatch)
    parent = SimpleNamespace(
        session_id="session-1", _delegate_depth=0, _subagent_id=None
    )
    decision = manager.invoke_gateway_pre_agent_dispatch(
        message="sensitive portfolio question",
        platform="telegram",
        session_id="session-1",
        parent_turn_id="gateway-turn-1",
        message_id="message-1",
        parent=parent,
        schedule=lambda *_: (_ for _ in ()).throw(RuntimeError("loop closed")),
        binding_validity=lambda: True,
    )

    assert decision == GatewayDispatchDecision()


def test_dispatch_and_control_observers_honor_live_revocation(monkeypatch):
    manager, context = _loaded_manager()
    granted = {"gateway.message_dispatch", "gateway.control_observer"}
    monkeypatch.setattr(
        plugins_mod,
        "plugin_capability_granted",
        lambda _plugin, capability: capability in granted,
    )
    calls = []
    context.register_hook(
        "gateway_pre_agent_dispatch",
        lambda **_: calls.append("dispatch") or {"action": "handled"},
    )
    context.register_hook(
        "gateway_status_contributor",
        lambda **_: calls.append("status") or "team: ready",
    )
    context.register_hook(
        "gateway_stop_observer", lambda **_: calls.append("stop") or "ignored"
    )

    assert _invoke(manager).action == "handled"
    assert manager.gateway_status_contributions(session_id="session-1") == [
        "team: ready"
    ]
    manager.notify_gateway_stop_observers(session_id="session-1", outcome="stopped")

    granted.clear()
    assert _invoke(manager) == GatewayDispatchDecision()
    assert manager.gateway_status_contributions(session_id="session-1") == []
    manager.notify_gateway_stop_observers(session_id="session-1", outcome="stopped")
    assert calls == ["dispatch", "status", "stop"]


def test_module_helpers_lazy_discover_and_delegate(monkeypatch):
    manager, _context = _loaded_manager()
    monkeypatch.setattr(plugins_mod, "_delivery_manager", lambda: manager)
    monkeypatch.setattr(
        manager,
        "gateway_status_contributions",
        lambda **payload: [payload["platform"]],
    )

    assert plugins_mod.gateway_status_contributions(platform="telegram") == ["telegram"]
