from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from agent.route_capability import (
    AccountScope,
    CoordinatorRouteError,
    CoordinatorRole,
    CoordinatorRoleRequest,
    CoordinatorService,
    GatewayDeliveryResult,
    GatewayDeliveryService,
    GatewayTaskService,
    activate_gateway_dispatch_binding,
    issue_gateway_dispatch_binding,
    reset_coordinator_registry_for_tests,
    revoke_gateway_dispatch_binding,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_coordinator_registry_for_tests()
    yield
    reset_coordinator_registry_for_tests()


def _binding(
    *, parent=None, parent_resolver=None, schedule=None, delivery=None, validity=None
):
    if parent is None and parent_resolver is None:
        parent = SimpleNamespace(
            session_id="session-1", _delegate_depth=0, _subagent_id=None
        )
    return issue_gateway_dispatch_binding(
        issuer_plugin_id="fixture",
        parent=parent,
        parent_resolver=parent_resolver,
        parent_session_id="session-1",
        parent_turn_id="turn-1",
        user_message="question",
        platform="telegram",
        message_id="message-1",
        schedule=schedule or (lambda _factory, _name, _binding: "task-1"),
        validity_resolver=validity or (lambda: True),
        delivery=delivery,
    )


def test_binding_is_opaque_signed_and_bound_to_exact_message():
    binding = _binding()
    assert binding.nonce not in repr(binding)
    assert binding.signature not in repr(binding)

    service = CoordinatorService(
        issuer_plugin_id="fixture",
        parent_agent_resolver=lambda: None,
        allowed_routes_resolver=lambda: ("route-a",),
        authorization_resolver=lambda: True,
    ).for_gateway_binding(binding)

    with pytest.raises(CoordinatorRouteError, match="message binding"):
        service.reserve_consultation(
            user_message="other",
            coordinator_route="route-a",
            consultation_id="T1",
            account_scope=AccountScope.OMITTED,
        )

    handle = service.reserve_consultation(
        user_message="question",
        coordinator_route="route-a",
        consultation_id="T1",
        account_scope=AccountScope.OMITTED,
    )
    assert handle.parent_session_id == "session-1"
    assert handle.parent_turn_id == "turn-1"

    assert (
        service.reserve_consultation(
            user_message="question",
            coordinator_route="route-a",
            consultation_id="T1",
            account_scope=AccountScope.OMITTED,
        )
        == handle
    )
    for changed in (
        {
            "coordinator_route": "route-b",
            "consultation_id": "T1",
            "account_scope": AccountScope.OMITTED,
        },
        {
            "coordinator_route": "route-a",
            "consultation_id": "T2",
            "account_scope": AccountScope.OMITTED,
        },
        {
            "coordinator_route": "route-a",
            "consultation_id": "T1",
            "account_scope": AccountScope.PORTFOLIO,
        },
    ):
        with pytest.raises(CoordinatorRouteError, match="already bound"):
            service.reserve_consultation(user_message="question", **changed)

    forged = dataclasses.replace(binding, parent_session_id="session-2")
    with pytest.raises(CoordinatorRouteError, match="verification failed"):
        CoordinatorService(
            issuer_plugin_id="fixture",
            parent_agent_resolver=lambda: None,
            allowed_routes_resolver=lambda: ("route-a",),
            authorization_resolver=lambda: True,
        ).for_gateway_binding(forged)


def test_bound_task_and_coordinator_fail_after_live_revocation():
    authorized = [True]
    scheduled = []
    binding = _binding(
        schedule=lambda factory, name, _binding: (
            scheduled.append((factory, name)) or "task-1"
        )
    )
    tasks = GatewayTaskService(
        "fixture", authorization_resolver=lambda: authorized[0]
    ).for_gateway_binding(binding)
    coordinator = CoordinatorService(
        issuer_plugin_id="fixture",
        parent_agent_resolver=lambda: None,
        allowed_routes_resolver=lambda: ("route-a",),
        authorization_resolver=lambda: authorized[0],
    ).for_gateway_binding(binding)

    authorized[0] = False
    with pytest.raises(PermissionError):
        tasks.spawn(lambda: None, name="consultation")  # type: ignore[arg-type]
    with pytest.raises(CoordinatorRouteError, match="not authorized"):
        coordinator.reserve_consultation(
            user_message="question",
            coordinator_route="route-a",
            consultation_id="T1",
            account_scope=AccountScope.OMITTED,
        )
    assert scheduled == []

    revoke_gateway_dispatch_binding(binding)
    authorized[0] = True
    with pytest.raises(CoordinatorRouteError, match="Unknown"):
        tasks.spawn(lambda: None, name="consultation")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_bound_delivery_rechecks_grant_issuer_and_binding_revocation():
    authorized = [True]
    calls = []

    async def deliver(plugin_id, binding_handle, _expires_at, delivery_key, content):
        calls.append((plugin_id, binding_handle, delivery_key, content))
        return GatewayDeliveryResult("DELIVERED", True, "delivery-1")

    binding = _binding(delivery=deliver)
    service = GatewayDeliveryService(
        "fixture", authorization_resolver=lambda: authorized[0]
    )
    bound = service.for_gateway_binding(binding)
    assert not any("adapter" in name for name in vars(bound))

    with pytest.raises(CoordinatorRouteError, match="verification failed"):
        GatewayDeliveryService(
            "other", authorization_resolver=lambda: True
        ).for_gateway_binding(binding)

    with pytest.raises(CoordinatorRouteError, match="not active"):
        await bound.deliver_once(delivery_key="final", content="answer")
    assert calls == []
    activate_gateway_dispatch_binding(binding)

    authorized[0] = False
    with pytest.raises(PermissionError, match="not granted"):
        await bound.deliver_once(delivery_key="final", content="answer")
    assert calls == []

    authorized[0] = True
    revoke_gateway_dispatch_binding(binding)
    with pytest.raises(CoordinatorRouteError, match="Unknown"):
        await bound.deliver_once(delivery_key="final", content="answer")
    assert calls == []


@pytest.mark.asyncio
async def test_bound_delivery_rechecks_session_liveness_before_callback():
    live = [True]
    calls = []

    async def deliver(*args):
        calls.append(args)
        return GatewayDeliveryResult("DELIVERED", True)

    binding = _binding(delivery=deliver, validity=lambda: live[0])
    bound = GatewayDeliveryService(
        "fixture", authorization_resolver=lambda: True
    ).for_gateway_binding(binding)
    live[0] = False

    with pytest.raises(CoordinatorRouteError, match="no longer live"):
        await bound.deliver_once(delivery_key="final", content="answer")
    assert calls == []


def test_gateway_bound_coordinator_launches_strict_toolless_leaf(monkeypatch):
    parent = SimpleNamespace(
        session_id="session-1",
        _delegate_depth=0,
        _subagent_id=None,
        enabled_toolsets=["terminal", "mcp"],
        valid_tool_names={"terminal", "mcp_market"},
    )
    parent_resolution_count = 0

    def resolve_parent():
        nonlocal parent_resolution_count
        parent_resolution_count += 1
        return parent

    binding = _binding(parent_resolver=resolve_parent)
    service = CoordinatorService(
        issuer_plugin_id="fixture",
        parent_agent_resolver=lambda: None,
        allowed_routes_resolver=lambda: ("route-a",),
        authorization_resolver=lambda: True,
    ).for_gateway_binding(binding)
    handle = service.reserve_consultation(
        user_message="question",
        coordinator_route="route-a",
        consultation_id="T1",
        account_scope=AccountScope.OMITTED,
    )
    assert parent_resolution_count == 0
    captured = {}

    class Child:
        _subagent_id = "child-1"
        _delegate_role = "leaf"
        _delegate_depth = 1
        provider = "test"
        model = "test-model"
        enabled_toolsets = []
        valid_tool_names = set()
        tools = []

    def build(**kwargs):
        captured.update(kwargs)
        return Child()

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService._submit",
        lambda *_args, **_kwargs: None,
    )

    role = service.launch_role(
        handle,
        CoordinatorRoleRequest(
            role=CoordinatorRole.RISK_PORTFOLIO,
            goal="Review risk",
            context="projection",
        ),
    )

    assert role.role is CoordinatorRole.RISK_PORTFOLIO
    assert captured["parent_agent"] is parent
    assert parent_resolution_count == 1
    assert captured["toolsets"] == []
    assert captured["role"] == "leaf"

    status = service.role_status(handle, CoordinatorRole.RISK_PORTFOLIO)
    terminal = service.wait_role(
        handle, CoordinatorRole.RISK_PORTFOLIO, timeout_seconds=0
    )
    result = service.role_result(handle, CoordinatorRole.RISK_PORTFOLIO)
    cancelled = service.cancel_role(
        handle, CoordinatorRole.RISK_PORTFOLIO, reason="test cancellation"
    )

    assert status.diagnostic is None
    assert terminal.diagnostic is None
    assert result.error_classification == "NOT_READY"
    assert cancelled.state == "CANCEL_REQUESTED"
    assert parent_resolution_count == 5
