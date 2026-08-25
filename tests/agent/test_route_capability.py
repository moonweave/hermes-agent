"""Fail-closed host coordinator service contract tests."""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.delegation_context import delegated_child_context
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
