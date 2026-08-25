"""Contract tests for the public plugin subagent lifecycle API."""

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from agent.subagent_lifecycle import (
    SubagentHandle,
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentResult,
    SubagentState,
    SubagentStatus,
    bind_subagent_parent,
    get_active_subagent_parent,
)


class FakeChild:
    def __init__(self, ident="sa-test"):
        self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "test"
        self.model = "test-model"
        self.interrupted = False
        self.interrupt_kind = None
        self.interrupt_message = None
        self.tool_reason = None
        self.session_id = ""
        self.enabled_toolsets: Any = None
        self.valid_tool_names = set()
        self.tools = []

    def interrupt(self, _reason):
        self.interrupted = True
        self.interrupt_kind = "soft"

    def hard_interrupt(self, reason, *, tool_reason=None):
        self.interrupted = True
        self.interrupt_kind = "hard"
        self.interrupt_message = reason
        self.tool_reason = tool_reason


@pytest.fixture
def lifecycle(monkeypatch):
    parent = SimpleNamespace(session_id="parent-1", enabled_toolsets=["file"])
    counter = iter(range(1000))

    def build(**_kwargs):
        return FakeChild(f"sa-{next(counter)}")

    def run(_index, _goal, child, _parent):
        for _ in range(20):
            if child.interrupted:
                return {
                    "status": "interrupted",
                    "summary": None,
                    "api_calls": 0,
                    "duration_seconds": 0,
                }
            time.sleep(0.002)
        return {
            "status": "completed",
            "summary": "safe summary",
            "api_calls": 1,
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    return SubagentLifecycleService(lambda: parent)






def test_cancel_is_cooperative_and_forged_handle_is_unknown(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    assert lifecycle.cancel(handle, reason="test").accepted
    terminal = lifecycle.wait(handle, timeout_seconds=1)
    assert terminal.state is SubagentState.CANCELLED
    forged = handle.__class__(**{**handle.to_dict(), "capability": "forged"})
    assert lifecycle.status(forged).state is SubagentState.UNKNOWN
    assert lifecycle.result(forged).error_classification == "UNKNOWN_HANDLE"
    other_parent = SimpleNamespace(session_id="different-parent")
    other_service = SubagentLifecycleService(lambda: other_parent)
    assert other_service.status(handle).state is SubagentState.UNKNOWN


def test_cancel_uses_explicit_hard_interrupt(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    record = lifecycle._record(handle)
    assert record is not None and record.agent is not None

    assert lifecycle.cancel(handle, reason="explicit user cancel").accepted

    assert record.agent.interrupt_kind == "hard"
    assert "explicit user cancel" in record.agent.interrupt_message
    assert record.agent.tool_reason == "subagent cancellation requested"
    lifecycle.wait(handle, timeout_seconds=1)








def test_public_lifecycle_runs_host_aggregation(monkeypatch):
    memory = Mock()
    parent = SimpleNamespace(
        session_id="parent-aggregate",
        enabled_toolsets=["file"],
        _memory_manager=memory,
        _current_turn_id="turn-1",
        session_estimated_cost_usd=1.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )
    child = FakeChild("sa-aggregate")
    child.session_id = "child-session"
    hook = Mock()

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **_kwargs: child)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "task_index": 0,
            "status": "completed",
            "summary": "aggregated",
            "api_calls": 1,
            "duration_seconds": 0.25,
            "_child_role": "leaf",
            "_child_cost_usd": 2.5,
        },
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="aggregate me"))
    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED

    memory.on_delegation.assert_called_once_with(
        task="aggregate me", result="aggregated", child_session_id="child-session"
    )
    hook.assert_called_once_with(
        "subagent_stop",
        parent_session_id="parent-aggregate",
        parent_turn_id="turn-1",
        child_session_id="child-session",
        child_role="leaf",
        child_summary="aggregated",
        child_status="completed",
        # Redacted tool history rides the shared finalization pipeline
        # (#62011/#72403); empty here because the fabricated result carries
        # no tool_trace.
        tool_call_history=[],
        duration_ms=250,
    )
    assert parent.session_estimated_cost_usd == 3.5
    assert parent.session_cost_source == "subagent"
    assert parent.session_cost_status == "estimated"




def test_agent_turn_binds_and_clears_lifecycle_parent(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    observed = []

    def run_conversation(parent, *_args, **_kwargs):
        observed.append(get_active_subagent_parent())
        return {"final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", run_conversation)

    assert agent.run_conversation("hello") == {"final_response": "ok"}
    assert observed == [agent]
    assert get_active_subagent_parent() is None


def test_empty_allowed_toolsets_constructs_a_strictly_toolless_child(monkeypatch):
    parent = SimpleNamespace(
        session_id="parent-toolless",
        enabled_toolsets=["file", "mcp-market"],
        valid_tool_names={"read_file", "mcp_market_snapshot"},
    )
    captured = {}
    child = FakeChild("sa-toolless")
    child.enabled_toolsets = []
    child.valid_tool_names = set()
    child.tools = []

    def build(**kwargs):
        captured.update(kwargs)
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    service = SubagentLifecycleService(lambda: parent)
    monkeypatch.setattr(service, "_submit", lambda *_args, **_kwargs: None)

    service.launch(SubagentLaunchRequest(goal="tool-less", allowed_toolsets=()))

    assert captured["toolsets"] == []


def test_unsafe_toolless_child_is_closed_before_error(monkeypatch):
    parent = SimpleNamespace(session_id="parent-unsafe", enabled_toolsets=["file"])

    class UnsafeChild(FakeChild):
        def __init__(self):
            super().__init__("sa-unsafe")
            self.enabled_toolsets = ["file"]
            self.valid_tool_names = {"read_file"}
            self.tools = [{"function": {"name": "read_file"}}]
            self.closed = False

        def close(self):
            self.closed = True

    child = UnsafeChild()
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", lambda **_kwargs: child
    )
    service = SubagentLifecycleService(lambda: parent)

    with pytest.raises(SubagentLifecycleError, match="strictly tool-less"):
        service.launch(SubagentLaunchRequest(goal="tool-less", allowed_toolsets=()))

    assert child.closed is True


def test_toolless_child_without_identity_is_closed_before_error(monkeypatch):
    parent = SimpleNamespace(session_id="parent-no-identity", enabled_toolsets=[])

    class IdentitylessChild(FakeChild):
        def __init__(self):
            super().__init__("")
            self.enabled_toolsets = []
            self.valid_tool_names = set()
            self.tools = []
            self.closed = False

        def close(self):
            self.closed = True

    child = IdentitylessChild()
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", lambda **_kwargs: child
    )
    service = SubagentLifecycleService(lambda: parent)

    with pytest.raises(SubagentLifecycleError, match="child identity"):
        service.launch(SubagentLaunchRequest(goal="tool-less", allowed_toolsets=()))

    assert child.closed is True


def test_submit_failure_rolls_back_exact_record_and_closes_child(monkeypatch):
    parent = SimpleNamespace(session_id="parent-submit-failure", enabled_toolsets=[])
    children = []

    class CloseableChild(FakeChild):
        def __init__(self, ident):
            super().__init__(ident)
            self.closed = False

        def close(self):
            self.closed = True

    def build(**_kwargs):
        child = CloseableChild(f"sa-submit-{len(children)}")
        children.append(child)
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    service = SubagentLifecycleService(lambda: parent)
    attempts = iter((RuntimeError("submit failed"), None))

    def submit(*_args, **_kwargs):
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(service, "_submit", submit)
    request = SubagentLaunchRequest(goal="retry", correlation_id="same")

    with pytest.raises(RuntimeError, match="submit failed"):
        service.launch(request)

    assert children[0].closed is True
    handle = service.launch(request)
    assert handle.subagent_id == "sa-submit-1"
    assert children[1].closed is False


def test_subagent_handle_repr_omits_capability():
    handle = SubagentHandle(
        contract_version=1,
        subagent_id="sa-repr",
        parent_session_id="parent",
        correlation_id=None,
        created_at=1.0,
        provider=None,
        model=None,
        role="leaf",
        depth=1,
        capability="secret-capability",
    )

    assert "secret-capability" not in repr(handle)
    assert "secret-capability" not in repr(
        SubagentStatus(handle, SubagentState.PENDING, 1.0)
    )
    assert "secret-capability" not in repr(
        SubagentResult(handle, SubagentState.PENDING, False)
    )
    assert handle.to_dict()["capability"] == "secret-capability"


def test_cached_plugin_lifecycle_revocation_blocks_every_operation():
    authorized = [True]
    parent = SimpleNamespace(session_id="parent-revoked", enabled_toolsets=[])
    service = SubagentLifecycleService(
        lambda: parent, authorization_resolver=lambda: authorized[0]
    )
    handle = SubagentHandle(
        contract_version=1,
        subagent_id="sa-revoked",
        parent_session_id="parent-revoked",
        correlation_id=None,
        created_at=1.0,
        provider=None,
        model=None,
        role="leaf",
        depth=1,
        capability="opaque",
    )
    authorized[0] = False
    operations = (
        lambda: service.launch(SubagentLaunchRequest(goal="blocked")),
        lambda: service.status(handle),
        lambda: service.wait(handle, timeout_seconds=0),
        lambda: service.result(handle),
        lambda: service.cancel(handle, reason="stop"),
        lambda: service.reconnect(handle),
    )

    for operation in operations:
        with pytest.raises(SubagentLifecycleError, match="authorized"):
            operation()
