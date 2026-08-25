"""Fail-closed host coordinator service contract tests."""

from __future__ import annotations

import dataclasses
import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.delegation_context import delegated_child_context
from agent.secret_scope import (
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from agent.route_capability import (
    AccountScope,
    CoordinatorJobState,
    CoordinatorRole,
    CoordinatorRoleCancelResult,
    CoordinatorRoleResult,
    CoordinatorRoleRequest,
    CoordinatorRoleStatus,
    CoordinatorRoleTerminalState,
    CoordinatorRouteError,
    CoordinatorService,
    TeamMcpBindingToken,
    reset_coordinator_registry_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_coordinator_registry_for_tests()
    yield
    reset_coordinator_registry_for_tests()


def _parent(*, session_id: str = "session-1", turn_id: str = "turn-1"):
    return SimpleNamespace(
        session_id=session_id,
        _current_turn_id=turn_id,
        _delegate_depth=0,
        enabled_toolsets=["terminal", "file", "mcp-market"],
        valid_tool_names={"terminal", "read_file", "mcp_market_snapshot"},
    )


@pytest.fixture
def service_setup(monkeypatch):
    parent = _parent()
    now = [1_800_000_000.0]
    service = CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: ("investment.team",),
        authorization_resolver=lambda: True,
        clock=lambda: now[0],
    )
    built = []

    class Child:
        def __init__(self):
            self._subagent_id = "sa-role"
            self._delegate_role = "leaf"
            self._delegate_depth = 1
            self.provider = "test"
            self.model = "test-model"
            self.enabled_toolsets = []
            self.valid_tool_names = set()
            self.tools = []
            self.closed = False

        def close(self):
            self.closed = True

        def hard_interrupt(self, _reason, *, tool_reason=None):
            self.interrupt_reason = tool_reason

    def build(**kwargs):
        child = Child()
        built.append((kwargs, child))
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService._submit",
        lambda *_a, **_k: None,
    )
    return parent, now, service, built


def _capability(service, **overrides):
    values = {
        "user_message": "오늘 시장 어때?",
        "coordinator_route": "investment.team",
        "consultation_id": "T123",
        "account_scope": AccountScope.OMITTED,
        "ttl_seconds": 60.0,
    }
    values.update(overrides)
    return service.issue_route_capability(**values)


def _reserve(service, capability, **overrides):
    values = {
        "capability": capability,
        "user_message": "오늘 시장 어때?",
        "coordinator_route": "investment.team",
        "consultation_id": "T123",
        "account_scope": AccountScope.OMITTED,
    }
    values.update(overrides)
    return service.reserve_consultation(**values)


def _decode_binding_payload(token):
    _version, encoded_payload, _signature = token.reveal_for_transport().split(".")
    padding = "=" * (-len(encoded_payload) % 4)
    return json.loads(base64.urlsafe_b64decode(encoded_payload + padding))


def test_reserve_creates_host_job_handle_without_constructing_aiagent(service_setup):
    _parent_agent, _now, service, built = service_setup
    capability = _capability(service)

    handle = _reserve(service, capability)

    assert built == []
    assert handle.issuer_plugin_id == "kospi-team"
    assert handle.parent_session_id == "session-1"
    assert handle.parent_turn_id == "turn-1"
    assert handle.consultation_id == "T123"
    assert handle.account_scope is AccountScope.OMITTED
    assert service.status(handle).state is CoordinatorJobState.RESERVED


def test_account_scope_is_closed_enum(service_setup):
    _parent_agent, _now, service, built = service_setup

    with pytest.raises(CoordinatorRouteError, match="AccountScope"):
        service.issue_route_capability(
            user_message="내 보유분 팔까?",
            coordinator_route="investment.team",
            consultation_id="T123",
            account_scope="PORTFOLIO",
        )

    assert built == []


def test_capability_and_handle_repr_omit_secrets(service_setup):
    _parent_agent, _now, service, _built = service_setup
    capability = _capability(service)
    handle = _reserve(service, capability)

    assert capability.signature not in repr(capability)
    assert capability.nonce not in repr(capability)
    assert handle.capability not in repr(handle)
    assert handle.capability not in repr(service.status(handle))


def test_team_mcp_binding_token_uses_profile_scoped_secret_and_exact_payload(
    service_setup, monkeypatch, caplog
):
    _parent_agent, now, service, _built = service_setup
    handle = _reserve(service, _capability(service))
    monkeypatch.setattr(
        "agent.route_capability.secrets.token_hex", lambda size: "ab" * size
    )
    encoded_key = base64.b64encode(bytes(range(32))).decode("ascii")
    scope_token = set_secret_scope({"KOSPI_TEAM_COORDINATOR_HMAC_KEY_B64": encoded_key})
    try:
        token = service.issue_team_mcp_binding_token(
            handle, personal_portfolio=False, ttl_seconds=60
        )
    finally:
        reset_secret_scope(scope_token)

    wire_token = token.reveal_for_transport()
    assert not isinstance(token, str)
    assert wire_token.startswith("v1.")
    assert wire_token not in str(token)
    assert wire_token not in f"{token}"
    assert wire_token not in "%s" % token
    assert wire_token not in repr(token)
    assert wire_token not in repr({"token": token})
    with pytest.raises(TypeError) as json_error:
        json.dumps({"token": token})
    assert wire_token not in str(json_error.value)
    assert wire_token not in caplog.text
    assert encoded_key not in caplog.text
    assert _decode_binding_payload(token) == {
        "version": 1,
        "key_id": "kospi-team-v2",
        "type": "coordinator_binding",
        "issuer_plugin_id": "kospi-team",
        "coordinator_id": handle.coordinator_id,
        "consultation_id": "T123",
        "parent_session_id": "session-1",
        "parent_turn_id": "turn-1",
        "user_message_sha256": handle.user_message_sha256,
        "coordinator_route": "investment.team",
        "account_scope": "OMITTED",
        "personal_portfolio": False,
        "issued_at": int(now[0]),
        "expires_at": int(now[0]) + 60,
        "nonce": "ab" * 16,
    }


def test_team_mcp_binding_token_matches_phase2_golden_vector():
    from agent.route_capability import _build_team_mcp_binding_token

    payload = {
        "account_scope": "PORTFOLIO",
        "consultation_id": "T123",
        "coordinator_id": "coord-0123456789abcdef",
        "coordinator_route": "portfolio_consultation",
        "expires_at": 1787652300,
        "issued_at": 1787652000,
        "issuer_plugin_id": "kospi-investment-team",
        "key_id": "kospi-team-v2",
        "nonce": "0123456789abcdef0123456789abcdef",
        "parent_session_id": "session-abc",
        "parent_turn_id": "turn-7",
        "personal_portfolio": True,
        "type": "coordinator_binding",
        "user_message_sha256": (
            "beaa718106b953f6761305f586ad2dce5318b7b60cfa0b0ba91d04d28d6299ca"
        ),
        "version": 1,
    }
    expected = (
        "v1.eyJhY2NvdW50X3Njb3BlIjoiUE9SVEZPTElPIiwiY29uc3VsdGF0aW9uX2lkIjoi"
        "VDEyMyIsImNvb3JkaW5hdG9yX2lkIjoiY29vcmQtMDEyMzQ1Njc4OWFiY2RlZiIsImNv"
        "b3JkaW5hdG9yX3JvdXRlIjoicG9ydGZvbGlvX2NvbnN1bHRhdGlvbiIsImV4cGlyZXNf"
        "YXQiOjE3ODc2NTIzMDAsImlzc3VlZF9hdCI6MTc4NzY1MjAwMCwiaXNzdWVyX3BsdWdp"
        "bl9pZCI6Imtvc3BpLWludmVzdG1lbnQtdGVhbSIsImtleV9pZCI6Imtvc3BpLXRlYW0t"
        "djIiLCJub25jZSI6IjAxMjM0NTY3ODlhYmNkZWYwMTIzNDU2Nzg5YWJjZGVmIiwicGFy"
        "ZW50X3Nlc3Npb25faWQiOiJzZXNzaW9uLWFiYyIsInBhcmVudF90dXJuX2lkIjoidHVy"
        "bi03IiwicGVyc29uYWxfcG9ydGZvbGlvIjp0cnVlLCJ0eXBlIjoiY29vcmRpbmF0b3Jf"
        "YmluZGluZyIsInVzZXJfbWVzc2FnZV9zaGEyNTYiOiJiZWFhNzE4MTA2Yjk1M2Y2NzYx"
        "MzA1ZjU4NmFkMmRjZTUzMThiN2I2MGNmYTBiMGJhOTFkMDRkMjhkNjI5OWNhIiwidmVy"
        "c2lvbiI6MX0.V6SXcPxZAU0wbtyPd88hsukI3wI4NUe4UC9TASFmK60"
    )

    token = TeamMcpBindingToken(
        _build_team_mcp_binding_token(payload, bytes(range(32)))
    )
    assert token.reveal_for_transport() == expected


@pytest.mark.parametrize(
    "encoded_key",
    [None, "%%%not-base64%%%", base64.b64encode(b"short").decode("ascii")],
)
def test_team_mcp_binding_token_missing_or_malformed_key_fails_closed(
    service_setup, monkeypatch, encoded_key
):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda _name: encoded_key)

    with pytest.raises(CoordinatorRouteError) as exc_info:
        service.issue_team_mcp_binding_token(
            handle, personal_portfolio=False, ttl_seconds=60
        )

    assert str(encoded_key) not in str(exc_info.value)
    assert built == []


def test_team_mcp_binding_token_never_falls_back_to_process_env_in_multiplex(
    service_setup, monkeypatch
):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))
    monkeypatch.setenv(
        "KOSPI_TEAM_COORDINATOR_HMAC_KEY_B64",
        base64.b64encode(bytes(range(32))).decode("ascii"),
    )
    previous = is_multiplex_active()
    scope_token = set_secret_scope(None)
    set_multiplex_active(True)
    try:
        with pytest.raises(CoordinatorRouteError, match="unavailable"):
            service.issue_team_mcp_binding_token(
                handle, personal_portfolio=False, ttl_seconds=60
            )
    finally:
        set_multiplex_active(previous)
        reset_secret_scope(scope_token)

    assert built == []


@pytest.mark.parametrize("ttl", [True, 0, -1, 301, float("inf"), float("nan")])
def test_team_mcp_binding_token_rejects_invalid_ttl(service_setup, ttl):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))

    with pytest.raises(CoordinatorRouteError, match="ttl_seconds"):
        service.issue_team_mcp_binding_token(
            handle, personal_portfolio=False, ttl_seconds=ttl
        )

    assert built == []


@pytest.mark.parametrize(
    ("account_scope", "personal_portfolio"),
    [(AccountScope.OMITTED, True), (AccountScope.PORTFOLIO, False)],
)
def test_team_mcp_binding_token_enforces_portfolio_scope_invariant(
    service_setup, account_scope, personal_portfolio
):
    _parent_agent, _now, service, built = service_setup
    capability = _capability(service, account_scope=account_scope)
    handle = _reserve(service, capability, account_scope=account_scope)

    with pytest.raises(CoordinatorRouteError, match="account scope"):
        service.issue_team_mcp_binding_token(
            handle, personal_portfolio=personal_portfolio, ttl_seconds=60
        )

    assert built == []


@pytest.mark.parametrize("personal_portfolio", [None, 0, 1, "false"])
def test_team_mcp_binding_token_requires_closed_boolean(
    service_setup, personal_portfolio
):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))

    with pytest.raises(CoordinatorRouteError, match="bool"):
        service.issue_team_mcp_binding_token(
            handle,
            personal_portfolio=cast(Any, personal_portfolio),
            ttl_seconds=60,
        )

    assert built == []


def test_team_mcp_binding_token_rechecks_grant_route_and_top_level(
    service_setup, monkeypatch
):
    parent, now, _service, built = service_setup
    authorized = [True]
    routes = [["investment.team"]]
    service = CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: tuple(routes[0]),
        authorization_resolver=lambda: authorized[0],
        clock=lambda: now[0],
    )
    handle = _reserve(service, _capability(service))
    monkeypatch.setattr(
        "agent.secret_scope.get_secret",
        lambda _name: base64.b64encode(bytes(range(32))).decode("ascii"),
    )

    authorized[0] = False
    with pytest.raises(CoordinatorRouteError, match="authorized"):
        service.issue_team_mcp_binding_token(handle, personal_portfolio=False)
    authorized[0] = True
    routes[0] = []
    with pytest.raises(CoordinatorRouteError, match="allowlisted"):
        service.issue_team_mcp_binding_token(handle, personal_portfolio=False)
    routes[0] = ["investment.team"]
    with delegated_child_context("child"):
        with pytest.raises(CoordinatorRouteError, match="top-level"):
            service.issue_team_mcp_binding_token(handle, personal_portfolio=False)

    assert built == []


def test_team_mcp_binding_token_never_enters_role_construction_input(
    service_setup, monkeypatch
):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))
    monkeypatch.setattr(
        "agent.secret_scope.get_secret",
        lambda _name: base64.b64encode(bytes(range(32))).decode("ascii"),
    )
    token = service.issue_team_mcp_binding_token(
        handle, personal_portfolio=False, ttl_seconds=60
    )

    service.launch_role(
        handle,
        CoordinatorRoleRequest(role=CoordinatorRole.MARKET_MACRO, goal="market"),
    )

    assert len(built) == 1
    assert token.reveal_for_transport() not in repr(built[0][0])


@pytest.mark.parametrize("nonce", ["a" * 31, "a" * 33, "A" * 32, "g" * 32])
def test_team_mcp_binding_token_requires_exact_lowercase_hex_nonce(
    service_setup, monkeypatch, nonce
):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))
    monkeypatch.setattr("agent.route_capability.secrets.token_hex", lambda _size: nonce)
    monkeypatch.setattr(
        "agent.secret_scope.get_secret",
        lambda _name: base64.b64encode(bytes(range(32))).decode("ascii"),
    )

    with pytest.raises(CoordinatorRouteError, match="nonce"):
        service.issue_team_mcp_binding_token(
            handle, personal_portfolio=False, ttl_seconds=60
        )

    assert built == []


def test_role_request_rejects_plugin_selected_model():
    request_type = cast(Any, CoordinatorRoleRequest)
    with pytest.raises(TypeError, match="model"):
        request_type(
            role=CoordinatorRole.RISK_PORTFOLIO,
            goal="risk",
            model="plugin/arbitrary-model",
        )


@pytest.mark.parametrize(
    ("reserve_override", "capability_override"),
    [
        ({}, {"signature": "0" * 64}),
        ({"user_message": "다른 질문"}, {}),
        ({"coordinator_route": "investment.other"}, {}),
        ({"consultation_id": "T999"}, {}),
        ({"account_scope": AccountScope.PORTFOLIO}, {}),
        ({}, {"issuer_plugin_id": "other-plugin"}),
        ({}, {"parent_session_id": "other-session"}),
        ({}, {"parent_turn_id": "other-turn"}),
    ],
)
def test_forged_or_cross_bound_capability_rejected_without_child(
    service_setup, reserve_override, capability_override
):
    _parent_agent, _now, service, built = service_setup
    capability = dataclasses.replace(_capability(service), **capability_override)

    with pytest.raises(CoordinatorRouteError):
        _reserve(service, capability, **reserve_override)
    assert built == []


def test_expired_capability_rejected_without_child(service_setup):
    _parent_agent, now, service, built = service_setup
    capability = _capability(service, ttl_seconds=1.0)
    now[0] += 2.0

    with pytest.raises(CoordinatorRouteError, match="expired"):
        _reserve(service, capability)
    assert built == []


def test_module_global_reservation_survives_plugin_context_reload(service_setup):
    parent, now, service, built = service_setup
    first = _reserve(service, _capability(service))
    reloaded = CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: ("investment.team",),
        authorization_resolver=lambda: True,
        clock=lambda: now[0],
    )

    second = _reserve(reloaded, _capability(reloaded))

    assert second == first
    assert built == []


def test_concurrent_reserve_across_service_instances_is_globally_atomic(service_setup):
    parent, now, first_service, built = service_setup
    second_service = CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: ("investment.team",),
        authorization_resolver=lambda: True,
        clock=lambda: now[0],
    )
    first_capability = _capability(first_service)
    second_capability = _capability(second_service)
    barrier = threading.Barrier(2)

    def reserve(service, capability):
        barrier.wait(timeout=2)
        return _reserve(service, capability)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(reserve, first_service, first_capability),
            executor.submit(reserve, second_service, second_capability),
        )
        handles = tuple(future.result(timeout=3) for future in futures)

    assert handles[0] == handles[1]
    assert built == []


def test_same_global_key_with_different_route_fails_instead_of_aliasing(service_setup):
    _parent_agent, _now, service, built = service_setup
    first = _reserve(service, _capability(service))
    service._allowed_routes_resolver = lambda: ("investment.team", "investment.other")
    other = _capability(service, coordinator_route="investment.other")

    with pytest.raises(CoordinatorRouteError, match="already reserved"):
        _reserve(service, other, coordinator_route="investment.other")

    assert service.status(first).state is CoordinatorJobState.RESERVED
    assert built == []


@pytest.mark.parametrize(
    "route_policy",
    [lambda: (), lambda: ("investment.other",), lambda: cast(Any, "corrupt")],
)
def test_cached_service_route_policy_drift_blocks_role_construction(
    service_setup, route_policy
):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))
    service._allowed_routes_resolver = route_policy

    with pytest.raises(CoordinatorRouteError, match="allowlisted"):
        service.launch_role(
            handle,
            CoordinatorRoleRequest(role=CoordinatorRole.MARKET_MACRO, goal="market"),
        )

    assert built == []


def test_internal_or_nested_child_cannot_issue_reserve_or_launch(service_setup):
    parent, _now, service, built = service_setup
    capability = _capability(service)
    handle = _reserve(service, capability)

    with delegated_child_context("child"):
        with pytest.raises(CoordinatorRouteError, match="top-level"):
            _capability(service, consultation_id="T124")
        with pytest.raises(CoordinatorRouteError, match="top-level"):
            _reserve(service, capability)
        with pytest.raises(CoordinatorRouteError, match="top-level"):
            service.launch_role(
                handle,
                CoordinatorRoleRequest(role=CoordinatorRole.RISK_PORTFOLIO, goal="x"),
            )

    parent._delegate_depth = 1
    parent._subagent_id = "sa-parent"
    with pytest.raises(CoordinatorRouteError, match="top-level"):
        _capability(service, consultation_id="T125")
    assert built == []


def test_role_launch_is_the_only_child_path_and_is_strict_leaf_toolless(service_setup):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))

    role_handle = service.launch_role(
        handle,
        CoordinatorRoleRequest(
            role=CoordinatorRole.RISK_PORTFOLIO, goal="Review portfolio risk"
        ),
    )

    assert role_handle.role is CoordinatorRole.RISK_PORTFOLIO
    assert len(built) == 1
    kwargs, child = built[0]
    assert kwargs["toolsets"] == []
    assert kwargs["role"] == "leaf"
    assert child.closed is False
    assert service.status(handle).state is CoordinatorJobState.ROLES_RUNNING


def test_role_lifecycle_is_scoped_through_coordinator_wrappers(service_setup):
    _parent_agent, _now, service, _built = service_setup
    handle = _reserve(service, _capability(service))
    request = CoordinatorRoleRequest(
        role=CoordinatorRole.RISK_PORTFOLIO, goal="Review portfolio risk"
    )
    launched = service.launch_role(handle, request)

    status = service.role_status(handle, CoordinatorRole.RISK_PORTFOLIO)
    terminal = service.wait_role(
        handle, CoordinatorRole.RISK_PORTFOLIO, timeout_seconds=0
    )
    result = service.role_result(handle, CoordinatorRole.RISK_PORTFOLIO)
    cancelled = service.cancel_role(
        handle, CoordinatorRole.RISK_PORTFOLIO, reason="stop"
    )

    assert status == CoordinatorRoleStatus(
        role_handle=launched,
        state="PENDING",
        updated_at=status.updated_at,
        diagnostic=None,
    )
    assert isinstance(terminal, CoordinatorRoleTerminalState)
    assert terminal.role_handle == launched
    assert terminal.completed is False
    assert isinstance(result, CoordinatorRoleResult)
    assert result.role_handle == launched
    assert result.ready is False
    assert isinstance(cancelled, CoordinatorRoleCancelResult)
    assert cancelled.role_handle == launched
    assert cancelled.accepted is True


def test_role_lifecycle_wrappers_reject_unknown_role_and_nested_context(service_setup):
    _parent_agent, _now, service, _built = service_setup
    handle = _reserve(service, _capability(service))
    role = CoordinatorRole.RISK_PORTFOLIO
    service.launch_role(
        handle, CoordinatorRoleRequest(role=role, goal="Review portfolio risk")
    )
    invalid_role = cast(Any, "PORTFOLIO_MANAGER")
    operations = (
        lambda selected: service.role_status(handle, selected),
        lambda selected: service.wait_role(handle, selected, timeout_seconds=0),
        lambda selected: service.role_result(handle, selected),
        lambda selected: service.cancel_role(handle, selected, reason="stop"),
    )

    for operation in operations:
        with pytest.raises(CoordinatorRouteError, match="CoordinatorRole"):
            operation(invalid_role)

    with delegated_child_context("child"):
        for operation in operations:
            with pytest.raises(CoordinatorRouteError, match="top-level"):
                operation(role)


@pytest.mark.parametrize("reason", [None, "", "x" * 501])
def test_cancel_role_rejects_invalid_reason(service_setup, reason):
    _parent_agent, _now, service, _built = service_setup
    handle = _reserve(service, _capability(service))
    role = CoordinatorRole.RISK_PORTFOLIO
    service.launch_role(handle, CoordinatorRoleRequest(role=role, goal="risk"))

    with pytest.raises(CoordinatorRouteError, match="reason"):
        service.cancel_role(handle, role, reason=cast(Any, reason))


def test_concurrent_duplicate_role_launch_constructs_one_child(service_setup):
    parent, now, first_service, built = service_setup
    second_service = CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: ("investment.team",),
        authorization_resolver=lambda: True,
        clock=lambda: now[0],
    )
    handle = _reserve(first_service, _capability(first_service))
    barrier = threading.Barrier(2)
    request = CoordinatorRoleRequest(
        role=CoordinatorRole.FUNDAMENTAL, goal="Analyze fundamentals"
    )

    def launch(service):
        barrier.wait(timeout=2)
        return service.launch_role(handle, request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(launch, first_service),
            executor.submit(launch, second_service),
        )
        role_handles = tuple(future.result(timeout=3) for future in futures)

    assert role_handles[0] == role_handles[1]
    assert len(built) == 1


def test_unsafe_partially_built_role_is_closed_and_retry_is_duplicate_free(
    service_setup, monkeypatch
):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))

    def unsafe_build(**kwargs):
        class UnsafeChild:
            _subagent_id = "unsafe"
            _delegate_role = "orchestrator"
            _delegate_depth = 1
            provider = "test"
            model = "test"
            enabled_toolsets = ["delegation"]
            valid_tool_names = {"delegate_task"}
            tools = [{"function": {"name": "delegate_task"}}]
            closed = False

            def close(self):
                self.closed = True

        child = UnsafeChild()
        built.append((kwargs, child))
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", unsafe_build
    )
    request = CoordinatorRoleRequest(
        role=CoordinatorRole.RISK_PORTFOLIO, goal="Review risk"
    )
    with pytest.raises(Exception, match="strictly tool-less"):
        service.launch_role(handle, request)

    assert built[0][1].closed is True
    assert service.status(handle).state is CoordinatorJobState.ROLE_FAILED

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **kwargs: type(
            "SafeChild",
            (),
            {
                "_subagent_id": "safe",
                "_delegate_role": "leaf",
                "_delegate_depth": 1,
                "provider": "test",
                "model": "test",
                "enabled_toolsets": [],
                "valid_tool_names": set(),
                "tools": [],
            },
        )(),
    )
    first = service.launch_role(handle, request)
    second = service.launch_role(handle, request)
    assert second == first
    assert service.status(handle).state is CoordinatorJobState.ROLES_RUNNING


def test_unknown_role_is_rejected_without_child(service_setup):
    _parent_agent, _now, service, built = service_setup
    handle = _reserve(service, _capability(service))

    with pytest.raises(CoordinatorRouteError, match="role request"):
        service.launch_role(
            handle,
            CoordinatorRoleRequest(
                role=cast(Any, "PORTFOLIO_MANAGER"), goal="escalate"
            ),
        )

    assert built == []
    assert service.status(handle).state is CoordinatorJobState.RESERVED


def test_service_authorization_revocation_fails_closed(service_setup):
    parent, now, _service, built = service_setup
    service = CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: ("investment.team",),
        authorization_resolver=lambda: False,
        clock=lambda: now[0],
    )

    with pytest.raises(CoordinatorRouteError, match="authorized"):
        _capability(service)
    assert built == []


def test_cached_service_revocation_blocks_every_coordinator_operation(service_setup):
    parent, now, _service, built = service_setup
    authorized = [True]
    service = CoordinatorService(
        issuer_plugin_id="kospi-team",
        parent_agent_resolver=lambda: parent,
        allowed_routes_resolver=lambda: ("investment.team",),
        authorization_resolver=lambda: authorized[0],
        clock=lambda: now[0],
    )
    handle = _reserve(service, _capability(service))
    role = CoordinatorRole.RISK_PORTFOLIO
    request = CoordinatorRoleRequest(role=role, goal="risk")
    service.launch_role(handle, request)
    reserve_capability = _capability(service, consultation_id="T124")
    authorized[0] = False

    operations = (
        lambda: _reserve(
            service,
            reserve_capability,
            consultation_id="T124",
        ),
        lambda: service.launch_role(handle, request),
        lambda: service.status(handle),
        lambda: service.role_status(handle, role),
        lambda: service.wait_role(handle, role, timeout_seconds=0),
        lambda: service.role_result(handle, role),
        lambda: service.cancel_role(handle, role, reason="stop"),
    )
    for operation in operations:
        with pytest.raises(CoordinatorRouteError, match="authorized"):
            operation()

    assert len(built) == 1


@pytest.mark.parametrize(
    ("declared", "granted"),
    [(False, False), (True, False), (False, True)],
)
def test_plugin_context_default_denies_without_declaration_and_grant(
    monkeypatch, declared, granted
):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    capabilities = ["delegation.coordinator"] if declared else []
    ctx = PluginContext(
        PluginManifest(
            name="kospi", key="kospi-team", source="user", capabilities=capabilities
        ),
        PluginManager(),
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.plugin_capability_granted",
        lambda plugin_id, capability: granted,
    )

    with pytest.raises(PermissionError, match="delegation.coordinator"):
        ctx.coordinator_service

    assert not hasattr(ctx, "route_authority")


def test_plugin_context_exposes_single_authorized_host_service(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    ctx = PluginContext(
        PluginManifest(
            name="kospi",
            key="kospi-team",
            source="user",
            capabilities=["delegation.coordinator"],
        ),
        PluginManager(),
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.plugin_capability_granted",
        lambda plugin_id, capability: True,
    )
    monkeypatch.setattr(
        ctx,
        "get_config",
        lambda key, default=None: (
            ["investment.team"] if key == "coordinator_routes" else default
        ),
    )

    service = ctx.coordinator_service
    assert service is ctx.coordinator_service
    assert isinstance(service, CoordinatorService)
    assert not hasattr(ctx, "route_authority")
    assert not hasattr(ctx, "coordinator_lifecycle")

    with pytest.raises(PermissionError, match="delegation.subagents"):
        ctx.subagent_lifecycle


@pytest.mark.parametrize(
    ("declared", "granted"),
    [(False, False), (True, False), (False, True)],
)
def test_generic_subagent_surface_requires_declaration_and_grant(
    monkeypatch, declared, granted
):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    ctx = PluginContext(
        PluginManifest(
            name="generic",
            key="generic",
            source="user",
            capabilities=["delegation.subagents"] if declared else [],
        ),
        PluginManager(),
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.plugin_capability_granted",
        lambda plugin_id, capability: granted,
    )

    with pytest.raises(PermissionError, match="delegation.subagents"):
        ctx.subagent_lifecycle


def test_cached_plugin_context_subagent_service_rechecks_revoked_grant(monkeypatch):
    from agent.subagent_lifecycle import (
        SubagentHandle,
        SubagentLaunchRequest,
        SubagentLifecycleError,
    )
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    granted = [True]
    ctx = PluginContext(
        PluginManifest(
            name="generic",
            key="generic",
            source="user",
            capabilities=["delegation.subagents"],
        ),
        PluginManager(),
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.plugin_capability_granted",
        lambda plugin_id, capability: granted[0],
    )
    service = ctx.subagent_lifecycle
    handle = SubagentHandle(
        contract_version=1,
        subagent_id="sa-cached-revoked",
        parent_session_id="parent",
        correlation_id=None,
        created_at=1.0,
        provider=None,
        model=None,
        role="leaf",
        depth=1,
        capability="opaque",
    )
    granted[0] = False
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
