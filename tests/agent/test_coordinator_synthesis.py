"""RED contracts for the additive top-level coordinator synthesis lane."""

from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import pytest

import agent.route_capability as route_capability
from agent.delegation_context import delegated_child_context
from agent.subagent_lifecycle import SubagentLifecycleError, SubagentState


@pytest.fixture(autouse=True)
def _clean_registry():
    route_capability.reset_coordinator_registry_for_tests()
    yield
    route_capability.reset_coordinator_registry_for_tests()


def _synthesis_api() -> type[Any]:
    request = getattr(route_capability, "CoordinatorSynthesisRequest", None)
    assert request is not None, "planned CoordinatorSynthesisRequest API is missing"
    return request


@pytest.fixture
def coordinator(monkeypatch: pytest.MonkeyPatch):
    parent = SimpleNamespace(
        session_id="session-1",
        _current_turn_id="turn-1",
        _delegate_depth=0,
        _subagent_id=None,
        enabled_toolsets=["terminal", "file", "mcp"],
        valid_tool_names={"terminal", "read_file", "mcp_market"},
    )
    built: list[tuple[dict[str, Any], Any]] = []

    class Child:
        def __init__(self, child_id: str) -> None:
            self._subagent_id = child_id
            self._delegate_role = "leaf"
            self._delegate_depth = 1
            self.provider = "test"
            self.model = "test-model"
            self.enabled_toolsets: list[str] = []
            self.valid_tool_names: set[str] = set()
            self.tools: list[Any] = []
            self.closed = False
            self.interrupt_reasons: list[str | None] = []

        def close(self) -> None:
            self.closed = True

        def hard_interrupt(
            self, _reason: str, *, tool_reason: str | None = None
        ) -> None:
            self.interrupt_reason = tool_reason
            self.interrupt_reasons.append(tool_reason)

    def build(**kwargs: Any) -> Child:
        child = Child(f"sa-{len(built) + 1}")
        built.append((kwargs, child))
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService._submit",
        lambda *_args, **_kwargs: None,
    )
    service = route_capability.CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: ("investment.team", "role_followup"),
        authorization_resolver=lambda: True,
        clock=lambda: 1_800_000_000.0,
    )
    capability = service.issue_route_capability(
        user_message="오늘 시장 어때?",
        coordinator_route="investment.team",
        consultation_id="T123",
        account_scope=route_capability.AccountScope.OMITTED,
    )
    handle = service.reserve_consultation(
        capability=capability,
        user_message="오늘 시장 어때?",
        coordinator_route="investment.team",
        consultation_id="T123",
        account_scope=route_capability.AccountScope.OMITTED,
    )
    return parent, service, handle, built


def _request(goal: str = "검증된 역할 결과만 종합하세요", digest: str = "a" * 64):
    request_type = _synthesis_api()
    return request_type(goal=goal, input_digest=f"sha256:{digest}")


def test_synthesis_request_has_no_model_tool_or_profile_override_surface() -> None:
    request_type = _synthesis_api()

    assert [field.name for field in dataclasses.fields(request_type)] == [
        "goal",
        "input_digest",
    ]
    for override in ("model", "toolsets", "tools", "profile"):
        with pytest.raises(TypeError, match=override):
            request_type(
                goal="종합",
                input_digest="sha256:" + "a" * 64,
                **{override: "forbidden"},
            )


def test_lifecycle_max_iterations_override_is_bounded_and_default_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.subagent_lifecycle import (
        SubagentLaunchRequest,
        SubagentLifecycleService,
    )

    parent = SimpleNamespace(session_id="parent-iterations", enabled_toolsets=[])
    captured: list[int] = []

    class Child:
        _delegate_role = "leaf"
        _delegate_depth = 1
        provider = "test"
        model = "test-model"
        enabled_toolsets: list[str] = []
        valid_tool_names: set[str] = set()
        tools: list[Any] = []

        def __init__(self, child_id: str) -> None:
            self._subagent_id = child_id

    def build(**kwargs: Any) -> Child:
        captured.append(kwargs["max_iterations"])
        return Child(f"sa-iterations-{len(captured)}")

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService._submit",
        lambda *_args, **_kwargs: None,
    )
    service = SubagentLifecycleService(lambda: parent)

    service.launch(SubagentLaunchRequest(goal="default iterations"))
    service.launch(SubagentLaunchRequest(goal="one iteration", max_iterations=1))

    assert captured == [250, 1]
    for invalid in (True, 0, -1, 251, 1.5, "1"):
        with pytest.raises(SubagentLifecycleError, match="max_iterations"):
            service.launch(
                SubagentLaunchRequest(
                    goal="invalid iterations",
                    max_iterations=cast(Any, invalid),
                )
            )


def test_one_synthesis_is_idempotent_by_digest_and_rejects_digest_change(
    coordinator,
) -> None:
    _parent, service, handle, built = coordinator

    first = service.launch_synthesis(handle, _request())
    second = service.launch_synthesis(handle, _request(goal="무시되는 재시도 문구"))

    assert second == first
    assert len(built) == 1
    with pytest.raises(route_capability.CoordinatorRouteError, match="digest"):
        service.launch_synthesis(handle, _request(digest="b" * 64))
    assert len(built) == 1


def test_concurrent_duplicate_synthesis_returns_one_handle(coordinator) -> None:
    _synthesis_api()
    parent, first_service, handle, built = coordinator
    second_service = route_capability.CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: ("investment.team", "role_followup"),
        authorization_resolver=lambda: True,
        clock=lambda: 1_800_000_000.0,
    )
    start = threading.Barrier(2)

    def launch(service: Any):
        start.wait(timeout=2)
        return service.launch_synthesis(handle, _request())

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(launch, first_service),
            executor.submit(launch, second_service),
        )
        handles = tuple(future.result(timeout=3) for future in futures)

    assert handles[0] == handles[1]
    assert len(built) == 1


def test_synthesis_is_top_level_bound_and_not_exposed_as_a_role(coordinator) -> None:
    parent, service, handle, built = coordinator
    request = _request()

    with delegated_child_context("nested"):
        with pytest.raises(route_capability.CoordinatorRouteError, match="top-level"):
            service.launch_synthesis(handle, request)
    parent._current_turn_id = "turn-2"
    with pytest.raises(route_capability.CoordinatorRouteError, match="binding"):
        service.launch_synthesis(handle, request)
    assert built == []

    assert not hasattr(route_capability.CoordinatorRole, "SYNTHESIS")
    assert "TEAM_LEAD" not in route_capability.CoordinatorRole.__members__
    parent._current_turn_id = "turn-1"
    with pytest.raises(route_capability.CoordinatorRouteError, match="Malformed"):
        service.launch_role(
            handle,
            route_capability.CoordinatorRoleRequest(
                role=cast(route_capability.CoordinatorRole, "TEAM_LEAD"),
                goal="forbidden role API",
            ),
        )
    parent._current_turn_id = "turn-2"
    followup_capability = service.issue_route_capability(
        user_message="Risk 의견만 다시",
        coordinator_route="role_followup",
        consultation_id="T124",
        account_scope=route_capability.AccountScope.OMITTED,
    )
    followup = service.reserve_consultation(
        capability=followup_capability,
        user_message="Risk 의견만 다시",
        coordinator_route="role_followup",
        consultation_id="T124",
        account_scope=route_capability.AccountScope.OMITTED,
    )
    followup_synthesis = service.launch_synthesis(followup, request)

    assert followup_synthesis.coordinator_id == followup.coordinator_id
    assert len(built) == 1


def test_synthesis_child_is_depth_one_leaf_with_no_tools(coordinator) -> None:
    _parent, service, handle, built = coordinator

    service.launch_synthesis(handle, _request())

    kwargs, child = built[0]
    assert kwargs["role"] == "leaf"
    assert kwargs["toolsets"] == []
    assert kwargs["model"] is None
    assert kwargs["max_iterations"] == 1
    assert child._delegate_depth == 1
    assert child.enabled_toolsets == []
    assert child.valid_tool_names == set()
    assert child.tools == []


@pytest.mark.parametrize("unsafe_kind", ["tools", "depth"])
def test_unsafe_synthesis_child_is_rejected_closed_and_unlinked(
    coordinator, monkeypatch: pytest.MonkeyPatch, unsafe_kind: str
) -> None:
    _synthesis_api()
    parent, service, handle, _built = coordinator
    parent._active_children = []
    parent._active_children_lock = threading.RLock()

    class UnsafeChild:
        _subagent_id = "unsafe-child"
        _delegate_role = "leaf"
        _delegate_depth = 2 if unsafe_kind == "depth" else 1
        provider = "test"
        model = "test-model"
        enabled_toolsets = ["terminal"] if unsafe_kind == "tools" else []
        valid_tool_names = {"terminal"} if unsafe_kind == "tools" else set()
        tools = [object()] if unsafe_kind == "tools" else []
        closed = False

        def close(self) -> None:
            self.closed = True

    child = UnsafeChild()

    def build(**_kwargs: Any) -> UnsafeChild:
        with parent._active_children_lock:
            parent._active_children.append(child)
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )

    expected_message = "tool|unsafe" if unsafe_kind == "tools" else "depth|unsafe"
    with pytest.raises(
        (route_capability.CoordinatorRouteError, SubagentLifecycleError),
        match=expected_message,
    ):
        service.launch_synthesis(handle, _request())

    assert child.closed is True
    assert parent._active_children == []


def test_synthesis_result_proves_one_model_call_and_zero_tool_execution(
    coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, service, handle, built = coordinator
    synthesis = service.launch_synthesis(handle, _request())

    payload = {
        "status": "completed",
        "answer": "관찰이 우선입니다.",
        "key_points": [],
        "dissent": None,
        "data_note": None,
    }
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.result",
        lambda _self, child_handle: SimpleNamespace(
            handle=child_handle,
            terminal_state=SubagentState.SUCCEEDED,
            ready=True,
            summary="unsafe summary sentinel",
            structured_payload=payload,
            error_classification=None,
            error_message=None,
            result_hash="sha256:" + "c" * 64,
            usage_metadata={"api_calls": 1},
            tool_execution_summary={"tool_calls": 0, "tool_turns": 0},
        ),
    )

    result = service.synthesis_result(handle)

    assert result.synthesis_handle == synthesis
    assert result.state == "SUCCEEDED"
    assert result.ready is True
    assert result.summary == "unsafe summary sentinel"
    assert result.structured_payload == payload
    assert result.error_classification is None
    assert result.error_message is None
    assert result.result_hash == "sha256:" + "c" * 64
    assert result.usage_metadata == {"api_calls": 1}
    assert result.tool_execution_summary == {"tool_calls": 0, "tool_turns": 0}
    assert len(built) == 1


def test_real_lifecycle_result_retains_bounded_toolless_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.subagent_lifecycle import (
        SubagentLaunchRequest,
        SubagentLifecycleService,
    )

    parent = SimpleNamespace(
        session_id="session-real-result",
        _current_turn_id="turn-real-result",
        _delegate_depth=0,
        _subagent_id=None,
        enabled_toolsets=[],
    )
    built = 0

    class Child:
        _delegate_role = "leaf"
        _delegate_depth = 1
        provider = "test"
        model = "test-model"
        enabled_toolsets: list[str] = []
        valid_tool_names: set[str] = set()
        tools: list[Any] = []

        def __init__(self, child_id: str) -> None:
            self._subagent_id = child_id

    def build(**_kwargs: Any) -> Child:
        nonlocal built
        built += 1
        return Child(f"sa-real-result-{built}")

    raw_summary = (
        '{"status":"completed","answer":"관찰이 우선입니다.",'
        '"key_points":[],"dissent":null,"data_note":null}'
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "summary": raw_summary,
            "api_calls": 1,
            "duration_seconds": 0.01,
            "tool_trace": [],
        },
    )
    service = route_capability.CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: ("investment.team",),
        authorization_resolver=lambda: True,
    )
    capability = service.issue_route_capability(
        user_message="팀 결과를 종합해줘",
        coordinator_route="investment.team",
        consultation_id="T-REAL-RESULT",
        account_scope=route_capability.AccountScope.OMITTED,
    )
    handle = service.reserve_consultation(
        capability=capability,
        user_message="팀 결과를 종합해줘",
        coordinator_route="investment.team",
        consultation_id="T-REAL-RESULT",
        account_scope=route_capability.AccountScope.OMITTED,
    )

    synthesis = service.launch_synthesis(handle, _request(digest="9" * 64))
    terminal = service.wait_synthesis(handle, timeout_seconds=1)
    result = service.synthesis_result(handle)

    assert terminal.synthesis_handle == synthesis
    assert terminal.state == "SUCCEEDED"
    assert terminal.completed is True
    assert result.synthesis_handle == synthesis
    assert result.state == "SUCCEEDED"
    assert result.ready is True
    assert result.summary == raw_summary
    assert result.structured_payload == {
        "status": "completed",
        "answer": "관찰이 우선입니다.",
        "key_points": [],
        "dissent": None,
        "data_note": None,
    }
    assert result.usage_metadata == {"api_calls": 1}
    assert result.tool_execution_summary == {
        "duration_seconds": 0.01,
        "tool_calls": 0,
        "tool_turns": 0,
    }
    assert isinstance(result.result_hash, str) and len(result.result_hash) == 64

    ordinary = SubagentLifecycleService(lambda: parent)
    ordinary_handle = ordinary.launch(SubagentLaunchRequest(goal="ordinary child"))
    assert (
        ordinary.wait(ordinary_handle, timeout_seconds=1).state
        is SubagentState.SUCCEEDED
    )
    ordinary_result = ordinary.result(ordinary_handle)
    assert ordinary_result.structured_payload is None
    assert ordinary_result.tool_execution_summary == {"duration_seconds": 0.01}


def test_synthesis_lifecycle_cancel_does_not_change_existing_role_api(
    coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, service, handle, built = coordinator
    synthesis = service.launch_synthesis(handle, _request())

    terminal = service.wait_synthesis(handle, timeout_seconds=0)
    cancelled = service.cancel_synthesis(handle, reason="deadline")

    assert terminal.synthesis_handle == synthesis
    assert cancelled.synthesis_handle == synthesis
    assert cancelled.accepted is True

    role = route_capability.CoordinatorRole.RISK_PORTFOLIO
    role_handle = service.launch_role(
        handle,
        route_capability.CoordinatorRoleRequest(role=role, goal="risk"),
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.result",
        lambda _self, child_handle: SimpleNamespace(
            handle=child_handle,
            terminal_state=SubagentState.SUCCEEDED,
            ready=True,
            summary="risk summary",
            structured_payload=None,
            error_classification=None,
            error_message=None,
            result_hash="sha256:" + "e" * 64,
        ),
    )
    assert service.role_status(handle, role).role_handle == role_handle
    assert service.wait_role(handle, role, timeout_seconds=0).role_handle == role_handle
    assert service.role_result(handle, role).role_handle == role_handle
    assert service.cancel_role(handle, role, reason="stop").role_handle == role_handle
    assert len(built) == 2


def test_synthesis_cancel_is_idempotent_without_claiming_false_terminal_state(
    coordinator,
) -> None:
    _parent, service, handle, built = coordinator
    synthesis = service.launch_synthesis(handle, _request())
    child = built[0][1]

    first = service.cancel_synthesis(handle, reason="deadline")
    terminal = service.wait_synthesis(handle, timeout_seconds=0)
    second = service.cancel_synthesis(handle, reason="duplicate")

    assert first.synthesis_handle == synthesis
    assert first.accepted is True
    assert child.interrupt_reason == "subagent cancellation requested"
    assert terminal.synthesis_handle == synthesis
    assert terminal.state == "CANCEL_REQUESTED"
    assert terminal.completed is False
    assert second.synthesis_handle == synthesis
    assert second.accepted is False
    assert second.already_terminal is False
    assert second.state == "CANCEL_REQUESTED"
    assert child.interrupt_reasons == ["subagent cancellation requested"]


def test_synthesis_cancel_does_not_hide_a_later_lifecycle_terminal_state(
    coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, service, handle, _built = coordinator
    synthesis = service.launch_synthesis(handle, _request())
    assert service.cancel_synthesis(handle, reason="deadline").accepted is True

    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.status",
        lambda _self, child_handle: SimpleNamespace(
            handle=child_handle,
            state=SubagentState.SUCCEEDED,
            updated_at=1.0,
            diagnostic=None,
        ),
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.wait",
        lambda _self, child_handle, *, timeout_seconds=None: SimpleNamespace(
            handle=child_handle,
            state=SubagentState.SUCCEEDED,
            completed=True,
            timed_out=False,
            diagnostic=None,
        ),
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.result",
        lambda _self, child_handle: SimpleNamespace(
            handle=child_handle,
            terminal_state=SubagentState.SUCCEEDED,
            ready=True,
            summary="late success",
            structured_payload={"status": "completed"},
            started_at=1.0,
            completed_at=2.0,
            error_classification=None,
            error_message=None,
            usage_metadata={"api_calls": 1},
            tool_execution_summary={"tool_calls": 0, "tool_turns": 0},
            result_hash="sha256:" + "7" * 64,
        ),
    )

    terminal = service.wait_synthesis(handle, timeout_seconds=0)
    result = service.synthesis_result(handle)
    repeated_cancel = service.cancel_synthesis(handle, reason="duplicate")

    assert terminal.synthesis_handle == synthesis
    assert terminal.state == "SUCCEEDED"
    assert terminal.completed is True
    assert result.state == "SUCCEEDED"
    assert result.summary == "late success"
    assert repeated_cancel.accepted is False
    assert repeated_cancel.already_terminal is True
    assert repeated_cancel.state == "SUCCEEDED"


def test_gateway_bound_service_forwards_synthesis_and_rechecks_binding(
    coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, _service, _handle, built = coordinator
    binding = route_capability.issue_gateway_dispatch_binding(
        issuer_plugin_id="kospi-team",
        parent=parent,
        parent_session_id="session-1",
        parent_turn_id="turn-1",
        user_message="오늘 시장 어때?",
        platform="telegram",
        message_id="message-1",
        schedule=lambda _factory, _name, _binding: "task-1",
        validity_resolver=lambda: True,
    )
    bound = route_capability.CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: None,
        allowed_routes_resolver=lambda: ("investment.team",),
        authorization_resolver=lambda: True,
    ).for_gateway_binding(binding)
    bound_api = cast(Any, bound)
    handle = bound.reserve_consultation(
        user_message="오늘 시장 어때?",
        coordinator_route="investment.team",
        consultation_id="T-GATEWAY",
        account_scope=route_capability.AccountScope.OMITTED,
    )

    synthesis = bound_api.launch_synthesis(handle, _request(digest="d" * 64))

    assert synthesis.coordinator_id == handle.coordinator_id
    assert built[-1][0]["parent_agent"] is parent
    assert bound_api.launch_synthesis(handle, _request(digest="d" * 64)) == synthesis
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.wait",
        lambda _self, child_handle, *, timeout_seconds=None: SimpleNamespace(
            handle=child_handle,
            state=SubagentState.SUCCEEDED,
            completed=True,
            timed_out=False,
            diagnostic=None,
        ),
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.result",
        lambda _self, child_handle: SimpleNamespace(
            handle=child_handle,
            terminal_state=SubagentState.SUCCEEDED,
            ready=True,
            summary="summary",
            structured_payload={"status": "completed"},
            error_classification=None,
            error_message=None,
            result_hash="sha256:" + "f" * 64,
            usage_metadata={"api_calls": 1},
            tool_execution_summary={"tool_calls": 0, "tool_turns": 0},
        ),
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.cancel",
        lambda _self, child_handle, *, reason: SimpleNamespace(
            handle=child_handle,
            accepted=True,
            already_terminal=False,
            unsupported=False,
            state=SubagentState.CANCEL_REQUESTED,
        ),
    )

    assert (
        bound_api.wait_synthesis(handle, timeout_seconds=0).synthesis_handle
        == synthesis
    )
    assert bound_api.synthesis_result(handle).synthesis_handle == synthesis
    assert (
        bound_api.cancel_synthesis(handle, reason="valid").synthesis_handle == synthesis
    )
    route_capability.revoke_gateway_dispatch_binding(binding)
    for method, kwargs in (
        (bound_api.wait_synthesis, {"timeout_seconds": 0}),
        (bound_api.synthesis_result, {}),
        (bound_api.cancel_synthesis, {"reason": "revoked"}),
    ):
        with pytest.raises(route_capability.CoordinatorRouteError, match="Unknown"):
            method(handle, **kwargs)


def test_synthesis_timeout_returns_before_release_and_discards_late_success(
    coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, service, handle, _built = coordinator
    parent._active_children = []
    parent._active_children_lock = threading.RLock()
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    late_finished = threading.Event()
    submit_calls: list[str] = []
    build_count = 0

    class Child:
        _delegate_role = "leaf"
        _delegate_depth = 1
        provider = "test"
        model = "test-model"
        enabled_toolsets: list[str] = []
        valid_tool_names: set[str] = set()
        tools: list[Any] = []

        def __init__(self, child_id: str) -> None:
            self._subagent_id = child_id

        def close(self) -> None:
            closed.set()

    def build(**_kwargs: Any) -> Child:
        nonlocal build_count
        build_count += 1
        child = Child(f"sa-preemptive-{build_count}")
        with parent._active_children_lock:
            parent._active_children.append(child)
        if build_count == 1:
            started.set()
            assert release.wait(timeout=2)
            late_finished.set()
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr("agent.subagent_lifecycle.time.monotonic", lambda: 100.0)
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService._submit",
        lambda _self, _record, goal, _parent: submit_calls.append(goal),
    )
    captured: list[BaseException] = []

    def launch() -> None:
        try:
            service.launch_synthesis(handle, _request(), timeout_seconds=0.02)
        except BaseException as exc:
            captured.append(exc)

    thread = threading.Thread(target=launch)
    thread.start()
    assert started.wait(timeout=1)
    thread.join(timeout=0.5)

    assert thread.is_alive() is False
    assert len(captured) == 1
    assert isinstance(captured[0], route_capability.CoordinatorSynthesisTimeoutError)
    assert submit_calls == []
    assert closed.is_set() is False

    release.set()
    assert late_finished.wait(timeout=1)
    assert closed.wait(timeout=1)
    deadline = time.monotonic() + 1
    while parent._active_children and time.monotonic() < deadline:
        time.sleep(0.005)
    assert parent._active_children == []
    assert submit_calls == []
    with pytest.raises(
        route_capability.CoordinatorRouteError, match="not been launched"
    ):
        service.synthesis_result(handle)

    retry = service.launch_synthesis(handle, _request(), timeout_seconds=0.5)

    assert retry.coordinator_id == handle.coordinator_id
    assert build_count == 2
    assert submit_calls == ["검증된 역할 결과만 종합하세요"]


def test_synthesis_deadline_is_rechecked_after_coordinator_lock_contention(
    coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, service, handle, _built = coordinator
    parent._active_children = []
    parent._active_children_lock = threading.RLock()
    build_started = threading.Event()
    release_build = threading.Event()
    prepared = threading.Event()
    lock_held = threading.Event()
    release_lock = threading.Event()
    deadline_elapsed = threading.Event()
    closed = threading.Event()
    submit_calls: list[str] = []

    class Child:
        _subagent_id = "sa-lock-contention"
        _delegate_role = "leaf"
        _delegate_depth = 1
        provider = "test"
        model = "test-model"
        enabled_toolsets: list[str] = []
        valid_tool_names: set[str] = set()
        tools: list[Any] = []

        def close(self) -> None:
            closed.set()

    child = Child()

    def build(**_kwargs: Any) -> Child:
        with parent._active_children_lock:
            parent._active_children.append(child)
        build_started.set()
        assert release_build.wait(timeout=2)
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    from agent.subagent_lifecycle import SubagentLifecycleService

    original_prepare = SubagentLifecycleService._prepare_strictly_toolless_launch

    def prepare(self: Any, request: Any) -> Any:
        child_handle = original_prepare(self, request)
        prepared.set()
        return child_handle

    monkeypatch.setattr(
        SubagentLifecycleService, "_prepare_strictly_toolless_launch", prepare
    )
    monkeypatch.setattr(
        route_capability.time,
        "monotonic",
        lambda: 101.0 if deadline_elapsed.is_set() else 100.0,
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService._submit",
        lambda _self, _record, goal, _parent: submit_calls.append(goal),
    )
    captured: list[BaseException] = []

    def launch() -> None:
        try:
            service.launch_synthesis(handle, _request(), timeout_seconds=0.5)
        except BaseException as exc:
            captured.append(exc)

    def hold_registry_lock() -> None:
        with route_capability._REGISTRY.lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    launch_thread = threading.Thread(target=launch)
    launch_thread.start()
    assert build_started.wait(timeout=1)
    lock_thread = threading.Thread(target=hold_registry_lock)
    lock_thread.start()
    assert lock_held.wait(timeout=1)
    release_build.set()
    assert prepared.wait(timeout=1)
    deadline_elapsed.set()
    release_lock.set()
    lock_thread.join(timeout=1)
    launch_thread.join(timeout=1)

    assert lock_thread.is_alive() is False
    assert launch_thread.is_alive() is False
    assert len(captured) == 1
    assert isinstance(captured[0], route_capability.CoordinatorSynthesisTimeoutError)
    assert submit_calls == []
    assert closed.wait(timeout=1)
    assert parent._active_children == []
    with pytest.raises(
        route_capability.CoordinatorRouteError, match="not been launched"
    ):
        service.synthesis_result(handle)


def test_synthesis_abandon_cleans_completion_that_won_callback_race() -> None:
    child_handle = object()
    abandoned = threading.Event()
    discarded: list[Any] = []
    lifecycle = SimpleNamespace(
        _discard_prepared=lambda handle: discarded.append(handle)
    )
    future: Future[Any] = Future()
    future.add_done_callback(
        lambda completed: route_capability._discard_abandoned_synthesis_launch(
            completed, lifecycle, abandoned
        )
    )
    future.set_result(child_handle)

    assert discarded == []

    route_capability._abandon_synthesis_launch(future, lifecycle, abandoned)

    assert discarded == [child_handle]


def test_synthesis_start_failure_discards_attempt_and_allows_fresh_retry(
    coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, service, handle, built = coordinator
    submit_calls = 0

    def submit(*_args: Any, **_kwargs: Any) -> None:
        nonlocal submit_calls
        submit_calls += 1
        if submit_calls == 1:
            raise RuntimeError("submit failed")

    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService._submit", submit
    )

    with pytest.raises(RuntimeError, match="submit failed"):
        service.launch_synthesis(handle, _request())

    assert len(built) == 1
    assert built[0][1].closed is True
    with pytest.raises(
        route_capability.CoordinatorRouteError, match="not been launched"
    ):
        service.synthesis_result(handle)

    retry = service.launch_synthesis(handle, _request())

    assert retry.coordinator_id == handle.coordinator_id
    assert len(built) == 2
    assert submit_calls == 2
