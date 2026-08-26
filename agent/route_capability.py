"""Capability-gated host coordinator jobs for trusted plugins.

Reserving a consultation creates only a host-owned record.  LLM agents are
created separately for named roles and are always first-level, leaf, and
strictly tool-less.
"""

from __future__ import annotations

import dataclasses
import base64
import binascii
import enum
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from typing import Any, Awaitable, Callable, Mapping, Optional


_TEAM_MCP_HMAC_ENV = "KOSPI_TEAM_COORDINATOR_HMAC_KEY_B64"


class CoordinatorRouteError(ValueError):
    """A coordinator operation cannot be safely accepted."""


@dataclasses.dataclass(frozen=True)
class GatewayDispatchBinding:
    """Opaque, process-local proof of one authorized gateway message."""

    version: int
    issuer_plugin_id: str
    parent_session_id: str
    parent_turn_id: str
    user_message_sha256: str
    platform: str
    message_id: str
    issued_at: float
    expires_at: float
    nonce: str = dataclasses.field(repr=False)
    signature: str = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class GatewayDispatchContext:
    """Frozen message projection delivered to a dispatch owner."""

    binding: GatewayDispatchBinding
    message: str = dataclasses.field(repr=False)
    platform: str
    session_id: str
    message_id: str


@dataclasses.dataclass(frozen=True)
class GatewayDispatchDecision:
    """Validated result of the single-owner gateway dispatch hook."""

    action: str = "allow"
    response: Optional[str] = None


@dataclasses.dataclass
class _GatewayDispatchRecord:
    binding: GatewayDispatchBinding
    parent_resolver: Callable[[], Any]
    schedule: Callable[[Callable[[], Awaitable[Any]], str, GatewayDispatchBinding], str]
    validity_resolver: Callable[[], bool]
    consultation_binding: Optional[tuple[str, str, AccountScope]] = None
    revocation_callbacks: list[Callable[[], None]] = dataclasses.field(
        default_factory=list
    )
    activation_callbacks: list[Callable[[], None]] = dataclasses.field(
        default_factory=list
    )
    activated: bool = False


class TeamMcpBindingToken:
    """Opaque bearer value requiring explicit transport disclosure."""

    __slots__ = ("__wire_value",)

    def __init__(self, wire_value: str) -> None:
        object.__setattr__(self, "_TeamMcpBindingToken__wire_value", wire_value)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("TeamMcpBindingToken is immutable.")

    def __str__(self) -> str:
        return "TeamMcpBindingToken(<redacted>)"

    def __repr__(self) -> str:
        return "TeamMcpBindingToken(<redacted>)"

    def reveal_for_transport(self) -> str:
        """Return wire bytes only at the explicit team-MCP bridge boundary."""
        return self.__wire_value


class AccountScope(str, enum.Enum):
    OMITTED = "OMITTED"
    PORTFOLIO = "PORTFOLIO"


class CoordinatorJobState(str, enum.Enum):
    RESERVED = "RESERVED"
    ROLES_RUNNING = "ROLES_RUNNING"
    ROLE_FAILED = "ROLE_FAILED"


class CoordinatorRole(str, enum.Enum):
    MARKET_MACRO = "MARKET_MACRO"
    FLOW_TECHNICAL = "FLOW_TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    RISK_PORTFOLIO = "RISK_PORTFOLIO"
    RED_TEAM = "RED_TEAM"


@dataclasses.dataclass(frozen=True)
class RouteCapability:
    issuer_plugin_id: str
    parent_session_id: str
    parent_turn_id: str
    user_message_sha256: str
    coordinator_route: str
    consultation_id: str
    account_scope: AccountScope
    issued_at: float
    expires_at: float
    nonce: str = dataclasses.field(repr=False)
    signature: str = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class CoordinatorHandle:
    coordinator_id: str
    issuer_plugin_id: str
    parent_session_id: str
    parent_turn_id: str
    user_message_sha256: str
    coordinator_route: str
    consultation_id: str
    account_scope: AccountScope
    created_at: float
    capability: str = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class CoordinatorRoleRequest:
    role: CoordinatorRole
    goal: str
    context: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class CoordinatorRoleHandle:
    coordinator_id: str
    role: CoordinatorRole
    role_run_id: str


@dataclasses.dataclass(frozen=True)
class CoordinatorRoleStatus:
    role_handle: CoordinatorRoleHandle
    state: str
    updated_at: float
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class CoordinatorRoleTerminalState:
    role_handle: CoordinatorRoleHandle
    state: str
    completed: bool
    timed_out: bool = False
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class CoordinatorRoleResult:
    role_handle: CoordinatorRoleHandle
    state: str
    ready: bool
    summary: Optional[str] = None
    structured_payload: Optional[Mapping[str, Any]] = None
    error_classification: Optional[str] = None
    error_message: Optional[str] = None
    result_hash: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class CoordinatorRoleCancelResult:
    role_handle: CoordinatorRoleHandle
    accepted: bool
    already_terminal: bool = False
    unsupported: bool = False
    state: str = "UNKNOWN"


@dataclasses.dataclass(frozen=True)
class CoordinatorStatus:
    handle: CoordinatorHandle
    state: CoordinatorJobState
    launched_roles: tuple[str, ...]


@dataclasses.dataclass
class _CoordinatorRecord:
    handle: CoordinatorHandle
    state: CoordinatorJobState = CoordinatorJobState.RESERVED
    role_handles: dict[str, Any] = dataclasses.field(default_factory=dict)


class _CoordinatorRegistry:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.by_key: dict[tuple[str, str, str, str, str], _CoordinatorRecord] = {}
        self.by_id: dict[str, _CoordinatorRecord] = {}
        self.used_nonces: set[str] = set()


_REGISTRY = _CoordinatorRegistry()
_ROUTE_SECRET = secrets.token_bytes(32)
_HANDLE_SECRET = secrets.token_bytes(32)
_GATEWAY_BINDING_SECRET = secrets.token_bytes(32)
_GATEWAY_BINDINGS_LOCK = threading.RLock()
_GATEWAY_BINDINGS: dict[str, _GatewayDispatchRecord] = {}


def issue_gateway_dispatch_binding(
    *,
    issuer_plugin_id: str,
    parent: Any = None,
    parent_resolver: Optional[Callable[[], Any]] = None,
    parent_session_id: str,
    parent_turn_id: str,
    user_message: str,
    platform: str,
    message_id: str,
    schedule: Callable[
        [Callable[[], Awaitable[Any]], str, GatewayDispatchBinding], str
    ],
    validity_resolver: Callable[[], bool],
    clock: Callable[[], float] = time.time,
    ttl_seconds: float = 300.0,
) -> GatewayDispatchBinding:
    """Create a host-owned binding after gateway authorization succeeds."""
    if (
        not _valid_text(issuer_plugin_id)
        or not _valid_text(parent_session_id)
        or not _valid_text(parent_turn_id)
        or not _valid_text(platform)
        or not isinstance(message_id, str)
        or len(message_id) > 512
        or not isinstance(user_message, str)
        or not callable(schedule)
        or not callable(validity_resolver)
        or (parent is None and not callable(parent_resolver))
        or (parent is not None and parent_resolver is not None)
    ):
        raise CoordinatorRouteError("Malformed gateway dispatch binding request.")
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, (int, float))
        or not math.isfinite(float(ttl_seconds))
    ):
        raise CoordinatorRouteError("Invalid gateway dispatch binding lifetime.")
    now = float(clock())
    if not math.isfinite(now) or ttl_seconds <= 0 or ttl_seconds > 300:
        raise CoordinatorRouteError("Invalid gateway dispatch binding lifetime.")
    unsigned = GatewayDispatchBinding(
        version=1,
        issuer_plugin_id=issuer_plugin_id,
        parent_session_id=parent_session_id,
        parent_turn_id=parent_turn_id,
        user_message_sha256=_message_hash(user_message),
        platform=platform,
        message_id=message_id,
        issued_at=now,
        expires_at=now + ttl_seconds,
        nonce=secrets.token_hex(16),
        signature="",
    )
    binding = dataclasses.replace(unsigned, signature=_sign_gateway_binding(unsigned))
    with _GATEWAY_BINDINGS_LOCK:
        _cleanup_gateway_bindings_locked(now)
        _GATEWAY_BINDINGS[binding.nonce] = _GatewayDispatchRecord(
            binding=binding,
            parent_resolver=(
                parent_resolver if parent_resolver is not None else lambda: parent
            ),
            schedule=schedule,
            validity_resolver=validity_resolver,
        )
    return binding


def revoke_gateway_dispatch_binding(binding: GatewayDispatchBinding) -> None:
    """Invalidate a dispatch binding that fell through to the normal agent."""
    callbacks: tuple[Callable[[], None], ...] = ()
    if isinstance(binding, GatewayDispatchBinding):
        with _GATEWAY_BINDINGS_LOCK:
            record = _GATEWAY_BINDINGS.get(binding.nonce)
            if record is not None and record.binding == binding:
                _GATEWAY_BINDINGS.pop(binding.nonce, None)
                callbacks = tuple(record.revocation_callbacks)
    for callback in callbacks:
        try:
            callback()
        except Exception:
            continue


def register_gateway_dispatch_revocation_callback(
    binding: GatewayDispatchBinding,
    callback: Callable[[], None],
) -> None:
    """Attach host cleanup that runs synchronously when a binding is revoked."""
    if not callable(callback):
        raise CoordinatorRouteError("Gateway revocation callback is invalid.")
    record = _gateway_dispatch_record(
        binding,
        issuer_plugin_id=binding.issuer_plugin_id,
    )
    with _GATEWAY_BINDINGS_LOCK:
        current = _GATEWAY_BINDINGS.get(binding.nonce)
        if current is None or current is not record or current.binding != binding:
            raise CoordinatorRouteError("Unknown gateway dispatch binding.")
        current.revocation_callbacks.append(callback)


def register_gateway_dispatch_activation_callback(
    binding: GatewayDispatchBinding,
    callback: Callable[[], None],
) -> None:
    """Attach host work that is released only after a handled decision."""
    if not callable(callback):
        raise CoordinatorRouteError("Gateway activation callback is invalid.")
    record = _gateway_dispatch_record(
        binding,
        issuer_plugin_id=binding.issuer_plugin_id,
    )
    invoke_now = False
    with _GATEWAY_BINDINGS_LOCK:
        current = _GATEWAY_BINDINGS.get(binding.nonce)
        if current is None or current is not record or current.binding != binding:
            raise CoordinatorRouteError("Unknown gateway dispatch binding.")
        if current.activated:
            invoke_now = True
        else:
            current.activation_callbacks.append(callback)
    if invoke_now:
        callback()


def activate_gateway_dispatch_binding(binding: GatewayDispatchBinding) -> None:
    """Release tasks admitted by the owner after a valid handled decision."""
    if not isinstance(binding, GatewayDispatchBinding):
        raise CoordinatorRouteError("Malformed gateway dispatch binding.")
    callbacks: tuple[Callable[[], None], ...] = ()
    with _GATEWAY_BINDINGS_LOCK:
        record = _GATEWAY_BINDINGS.get(binding.nonce)
        if record is None or record.binding != binding:
            raise CoordinatorRouteError("Unknown gateway dispatch binding.")
        if not record.activated:
            record.activated = True
            callbacks = tuple(record.activation_callbacks)
            record.activation_callbacks.clear()
    for callback in callbacks:
        callback()


def revoke_gateway_dispatch_bindings_for_session(session_id: str) -> int:
    """Revoke every live binding owned by one authorized gateway session."""
    if not _valid_text(session_id):
        return 0
    callbacks: list[Callable[[], None]] = []
    with _GATEWAY_BINDINGS_LOCK:
        nonces = [
            nonce
            for nonce, record in _GATEWAY_BINDINGS.items()
            if record.binding.parent_session_id == session_id
        ]
        for nonce in nonces:
            record = _GATEWAY_BINDINGS.pop(nonce)
            callbacks.extend(record.revocation_callbacks)
    for callback in callbacks:
        try:
            callback()
        except Exception:
            continue
    return len(nonces)


def gateway_dispatch_session_has_live_bindings(session_id: str) -> bool:
    """Return whether an authorized gateway session still owns a live binding."""
    if not _valid_text(session_id):
        return False
    with _GATEWAY_BINDINGS_LOCK:
        _cleanup_gateway_bindings_locked(time.time())
        return any(
            record.binding.parent_session_id == session_id
            for record in _GATEWAY_BINDINGS.values()
        )


def _gateway_dispatch_record(
    binding: GatewayDispatchBinding,
    *,
    issuer_plugin_id: str,
    clock: Callable[[], float] = time.time,
) -> _GatewayDispatchRecord:
    if not isinstance(binding, GatewayDispatchBinding):
        raise CoordinatorRouteError("Malformed gateway dispatch binding.")
    if (
        binding.version != 1
        or not _valid_text(binding.issuer_plugin_id)
        or not _valid_text(binding.parent_session_id)
        or not _valid_text(binding.parent_turn_id)
        or not _valid_text(binding.platform)
        or not isinstance(binding.message_id, str)
        or len(binding.message_id) > 512
        or not isinstance(binding.user_message_sha256, str)
        or len(binding.user_message_sha256) != 64
        or not isinstance(binding.nonce, str)
        or len(binding.nonce) != 32
        or not isinstance(binding.signature, str)
        or len(binding.signature) != 64
        or not isinstance(binding.issued_at, (int, float))
        or not isinstance(binding.expires_at, (int, float))
        or isinstance(binding.issued_at, bool)
        or isinstance(binding.expires_at, bool)
        or not math.isfinite(float(binding.issued_at))
        or not math.isfinite(float(binding.expires_at))
    ):
        raise CoordinatorRouteError("Malformed gateway dispatch binding.")
    expected = _sign_gateway_binding(dataclasses.replace(binding, signature=""))
    if binding.issuer_plugin_id != issuer_plugin_id or not hmac.compare_digest(
        binding.signature, expected
    ):
        raise CoordinatorRouteError("Gateway dispatch binding verification failed.")
    now = float(clock())
    if binding.issued_at > now or binding.expires_at <= now:
        raise CoordinatorRouteError(
            "Gateway dispatch binding is expired or not yet valid."
        )
    with _GATEWAY_BINDINGS_LOCK:
        _cleanup_gateway_bindings_locked(now)
        record = _GATEWAY_BINDINGS.get(binding.nonce)
        if record is None or record.binding != binding:
            raise CoordinatorRouteError("Unknown gateway dispatch binding.")
        try:
            still_valid = record.validity_resolver()
        except Exception as exc:
            raise CoordinatorRouteError(
                "Gateway dispatch binding liveness check failed."
            ) from exc
        if still_valid is not True:
            raise CoordinatorRouteError("Gateway dispatch binding is no longer live.")
        return record


def _gateway_dispatch_parent(record: _GatewayDispatchRecord) -> Any:
    try:
        parent = record.parent_resolver()
    except Exception as exc:
        raise CoordinatorRouteError(
            "Gateway detached parent resolution failed."
        ) from exc
    if parent is None:
        raise CoordinatorRouteError("Gateway detached parent is unavailable.")
    if str(getattr(parent, "session_id", "") or "") != record.binding.parent_session_id:
        raise CoordinatorRouteError("Gateway parent session binding changed.")
    return parent


def _bind_gateway_consultation(
    binding: GatewayDispatchBinding,
    *,
    issuer_plugin_id: str,
    coordinator_route: str,
    consultation_id: str,
    account_scope: AccountScope,
) -> None:
    """Atomically bind one gateway message to its first consultation tuple."""
    record = _gateway_dispatch_record(binding, issuer_plugin_id=issuer_plugin_id)
    requested = (coordinator_route, consultation_id, account_scope)
    with _GATEWAY_BINDINGS_LOCK:
        if record.consultation_binding is None:
            record.consultation_binding = requested
        elif record.consultation_binding != requested:
            raise CoordinatorRouteError(
                "Gateway dispatch binding is already bound to another consultation."
            )


def _cleanup_gateway_bindings_locked(now: float) -> None:
    for nonce in [
        nonce
        for nonce, record in _GATEWAY_BINDINGS.items()
        if record.binding.expires_at <= now
    ]:
        _GATEWAY_BINDINGS.pop(nonce, None)


class GatewayBoundTaskService:
    """Host-owned scheduler bound to one signed gateway message."""

    def __init__(
        self,
        *,
        issuer_plugin_id: str,
        binding: GatewayDispatchBinding,
        authorization_resolver: Callable[[], bool],
    ) -> None:
        self._issuer_plugin_id = issuer_plugin_id
        self._binding = binding
        self._authorization_resolver = authorization_resolver

    def spawn(
        self, factory: Callable[[], Awaitable[Any]], *, name: str = "plugin-task"
    ) -> str:
        if self._authorization_resolver() is not True:
            raise PermissionError("Gateway task capability is not granted.")
        if not callable(factory) or not _valid_text(name, 128):
            raise ValueError("Gateway background task request is malformed.")
        record = _gateway_dispatch_record(
            self._binding, issuer_plugin_id=self._issuer_plugin_id
        )
        return record.schedule(factory, name, self._binding)


class GatewayTaskService:
    """Unbound plugin facade; a signed dispatch binding is always required."""

    def __init__(
        self, issuer_plugin_id: str, authorization_resolver: Callable[[], bool]
    ) -> None:
        self._issuer_plugin_id = issuer_plugin_id
        self._authorization_resolver = authorization_resolver

    def for_gateway_binding(
        self, binding: GatewayDispatchBinding
    ) -> GatewayBoundTaskService:
        if self._authorization_resolver() is not True:
            raise PermissionError("Gateway task capability is not granted.")
        _gateway_dispatch_record(binding, issuer_plugin_id=self._issuer_plugin_id)
        return GatewayBoundTaskService(
            issuer_plugin_id=self._issuer_plugin_id,
            binding=binding,
            authorization_resolver=self._authorization_resolver,
        )


class CoordinatorService:
    """The single host surface for coordinator reservation and role launch."""

    def __init__(
        self,
        *,
        issuer_plugin_id: str,
        parent_agent_resolver: Callable[[], Any],
        allowed_routes_resolver: Callable[[], tuple[str, ...]],
        authorization_resolver: Callable[[], bool],
        clock: Callable[[], float] = time.time,
        gateway_binding: Optional[GatewayDispatchBinding] = None,
    ) -> None:
        if not _valid_text(issuer_plugin_id):
            raise CoordinatorRouteError("issuer_plugin_id is invalid.")
        self.issuer_plugin_id = issuer_plugin_id
        self._parent_agent_resolver = parent_agent_resolver
        self._allowed_routes_resolver = allowed_routes_resolver
        self._authorization_resolver = authorization_resolver
        self._clock = clock
        self._gateway_binding = gateway_binding

    def for_gateway_binding(
        self, binding: GatewayDispatchBinding
    ) -> "GatewayBoundCoordinatorService":
        """Bind coordinator authority to one authorized gateway message."""
        self._require_authorized()
        record = _gateway_dispatch_record(
            binding, issuer_plugin_id=self.issuer_plugin_id, clock=self._clock
        )
        bound = CoordinatorService(
            issuer_plugin_id=self.issuer_plugin_id,
            parent_agent_resolver=lambda: _gateway_dispatch_parent(record),
            allowed_routes_resolver=self._allowed_routes_resolver,
            authorization_resolver=self._authorization_resolver,
            clock=self._clock,
            gateway_binding=binding,
        )
        return GatewayBoundCoordinatorService(bound, binding)

    def issue_route_capability(
        self,
        *,
        user_message: str,
        coordinator_route: str,
        consultation_id: str,
        account_scope: AccountScope,
        ttl_seconds: float = 300.0,
    ) -> RouteCapability:
        self._require_authorized()
        _parent, session_id, turn_id = self._current_binding()
        self._validate_route_request(
            user_message, coordinator_route, consultation_id, account_scope
        )
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
            or ttl_seconds > 300
        ):
            raise CoordinatorRouteError(
                "ttl_seconds must be greater than 0 and at most 300."
            )
        issued_at = float(self._clock())
        expires_at = issued_at + float(ttl_seconds)
        if not math.isfinite(issued_at) or not math.isfinite(expires_at):
            raise CoordinatorRouteError("trusted clock returned an invalid time.")
        unsigned = RouteCapability(
            issuer_plugin_id=self.issuer_plugin_id,
            parent_session_id=session_id,
            parent_turn_id=turn_id,
            user_message_sha256=_message_hash(user_message),
            coordinator_route=coordinator_route,
            consultation_id=consultation_id,
            account_scope=account_scope,
            issued_at=issued_at,
            expires_at=expires_at,
            nonce=secrets.token_hex(16),
            signature="",
        )
        return dataclasses.replace(unsigned, signature=_sign_route(unsigned))

    def reserve_consultation(
        self,
        *,
        capability: RouteCapability,
        user_message: str,
        coordinator_route: str,
        consultation_id: str,
        account_scope: AccountScope,
    ) -> CoordinatorHandle:
        self._require_authorized()
        _parent, session_id, turn_id = self._current_binding()
        self._validate_route_request(
            user_message, coordinator_route, consultation_id, account_scope
        )
        self._verify_route_capability(
            capability,
            session_id=session_id,
            turn_id=turn_id,
            user_message=user_message,
            coordinator_route=coordinator_route,
            consultation_id=consultation_id,
            account_scope=account_scope,
        )
        message_hash = _message_hash(user_message)
        key = (
            self.issuer_plugin_id,
            session_id,
            turn_id,
            message_hash,
            consultation_id,
        )
        with _REGISTRY.lock:
            existing = _REGISTRY.by_key.get(key)
            if existing is not None:
                handle = existing.handle
                if (
                    handle.coordinator_route != coordinator_route
                    or handle.account_scope is not account_scope
                ):
                    raise CoordinatorRouteError(
                        "Consultation key is already reserved with different bindings."
                    )
                return handle
            if capability.nonce in _REGISTRY.used_nonces:
                raise CoordinatorRouteError("Route capability has already been used.")
            _REGISTRY.used_nonces.add(capability.nonce)
            created_at = float(self._clock())
            unsigned_handle = CoordinatorHandle(
                coordinator_id=f"coord-{secrets.token_hex(8)}",
                issuer_plugin_id=self.issuer_plugin_id,
                parent_session_id=session_id,
                parent_turn_id=turn_id,
                user_message_sha256=message_hash,
                coordinator_route=coordinator_route,
                consultation_id=consultation_id,
                account_scope=account_scope,
                created_at=created_at,
                capability="",
            )
            handle = dataclasses.replace(
                unsigned_handle, capability=_sign_handle(unsigned_handle)
            )
            record = _CoordinatorRecord(handle=handle)
            _REGISTRY.by_key[key] = record
            _REGISTRY.by_id[handle.coordinator_id] = record
            return handle

    def launch_role(self, handle: CoordinatorHandle, request: CoordinatorRoleRequest):
        self._require_authorized()
        parent, session_id, turn_id = self._current_binding(require_parent=True)
        record = self._record_for_handle(handle, session_id=session_id, turn_id=turn_id)
        self._require_route_allowed(handle.coordinator_route)
        if (
            not isinstance(request, CoordinatorRoleRequest)
            or not isinstance(request.role, CoordinatorRole)
            or not isinstance(request.goal, str)
            or not request.goal.strip()
        ):
            raise CoordinatorRouteError("Malformed coordinator role request.")

        with _REGISTRY.lock:
            role_name = request.role.value
            existing = record.role_handles.get(role_name)
            if existing is not None:
                return self._public_role_handle(handle, request.role, existing)
            from agent.subagent_lifecycle import (
                SubagentLaunchRequest,
                SubagentLifecycleService,
            )

            lifecycle = SubagentLifecycleService(
                lambda: parent, authorization_resolver=self._authorization_resolver
            )
            try:
                role_handle = lifecycle.launch(
                    SubagentLaunchRequest(
                        goal=request.goal,
                        context=request.context,
                        role="leaf",
                        model=None,
                        allowed_toolsets=(),
                        parent_session_id=session_id,
                        correlation_id=(f"{handle.coordinator_id}:{role_name}"),
                        metadata={"coordinator_id": handle.coordinator_id},
                    )
                )
            except Exception:
                record.state = CoordinatorJobState.ROLE_FAILED
                raise
            record.role_handles[role_name] = role_handle
            record.state = CoordinatorJobState.ROLES_RUNNING
            return self._public_role_handle(handle, request.role, role_handle)

    def issue_team_mcp_binding_token(
        self,
        handle: CoordinatorHandle,
        *,
        personal_portfolio: bool,
        ttl_seconds: float = 300.0,
    ) -> TeamMcpBindingToken:
        """Issue a short-lived profile-scoped binding for the team MCP."""
        self._require_authorized()
        _parent, session_id, turn_id = self._current_binding()
        self._record_for_handle(handle, session_id=session_id, turn_id=turn_id)
        self._require_route_allowed(handle.coordinator_route)
        if not isinstance(personal_portfolio, bool):
            raise CoordinatorRouteError("personal_portfolio must be a bool.")
        if (handle.account_scope is AccountScope.OMITTED and personal_portfolio) or (
            handle.account_scope is AccountScope.PORTFOLIO and not personal_portfolio
        ):
            raise CoordinatorRouteError(
                "personal_portfolio does not match the coordinator account scope."
            )
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
            or ttl_seconds > 300
        ):
            raise CoordinatorRouteError(
                "ttl_seconds must be greater than 0 and at most 300."
            )
        now = float(self._clock())
        if not math.isfinite(now) or now < 0:
            raise CoordinatorRouteError("trusted clock returned an invalid time.")
        issued_at = int(now)
        expires_at = issued_at + max(1, math.ceil(float(ttl_seconds)))
        nonce = secrets.token_hex(16)
        if len(nonce) != 32 or any(char not in "0123456789abcdef" for char in nonce):
            raise CoordinatorRouteError(
                "trusted nonce generator returned invalid data."
            )
        signing_key = _team_mcp_signing_key()
        payload = {
            "version": 1,
            "key_id": "kospi-team-v2",
            "type": "coordinator_binding",
            "issuer_plugin_id": handle.issuer_plugin_id,
            "coordinator_id": handle.coordinator_id,
            "consultation_id": handle.consultation_id,
            "parent_session_id": handle.parent_session_id,
            "parent_turn_id": handle.parent_turn_id,
            "user_message_sha256": handle.user_message_sha256,
            "coordinator_route": handle.coordinator_route,
            "account_scope": handle.account_scope.value,
            "personal_portfolio": personal_portfolio,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "nonce": nonce,
        }
        return TeamMcpBindingToken(_build_team_mcp_binding_token(payload, signing_key))

    def role_status(
        self, handle: CoordinatorHandle, role: CoordinatorRole
    ) -> CoordinatorRoleStatus:
        lifecycle, child_handle, role_handle = self._role_binding(handle, role)
        status = lifecycle.status(child_handle)
        return CoordinatorRoleStatus(
            role_handle=role_handle,
            state=status.state.value,
            updated_at=status.updated_at,
            diagnostic=status.diagnostic,
        )

    def wait_role(
        self,
        handle: CoordinatorHandle,
        role: CoordinatorRole,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> CoordinatorRoleTerminalState:
        lifecycle, child_handle, role_handle = self._role_binding(handle, role)
        terminal = lifecycle.wait(child_handle, timeout_seconds=timeout_seconds)
        return CoordinatorRoleTerminalState(
            role_handle=role_handle,
            state=terminal.state.value,
            completed=terminal.completed,
            timed_out=terminal.timed_out,
            diagnostic=terminal.diagnostic,
        )

    def role_result(
        self, handle: CoordinatorHandle, role: CoordinatorRole
    ) -> CoordinatorRoleResult:
        lifecycle, child_handle, role_handle = self._role_binding(handle, role)
        result = lifecycle.result(child_handle)
        return CoordinatorRoleResult(
            role_handle=role_handle,
            state=result.terminal_state.value,
            ready=result.ready,
            summary=result.summary,
            structured_payload=result.structured_payload,
            error_classification=result.error_classification,
            error_message=result.error_message,
            result_hash=result.result_hash,
        )

    def cancel_role(
        self,
        handle: CoordinatorHandle,
        role: CoordinatorRole,
        *,
        reason: str,
    ) -> CoordinatorRoleCancelResult:
        lifecycle, child_handle, role_handle = self._role_binding(handle, role)
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise CoordinatorRouteError(
                "cancel reason must be a non-empty string of at most 500 characters."
            )
        result = lifecycle.cancel(child_handle, reason=reason)
        return CoordinatorRoleCancelResult(
            role_handle=role_handle,
            accepted=result.accepted,
            already_terminal=result.already_terminal,
            unsupported=result.unsupported,
            state=result.state.value,
        )

    def status(self, handle: CoordinatorHandle) -> CoordinatorStatus:
        self._require_authorized()
        _parent, session_id, turn_id = self._current_binding()
        record = self._record_for_handle(handle, session_id=session_id, turn_id=turn_id)
        with _REGISTRY.lock:
            return CoordinatorStatus(
                handle=record.handle,
                state=record.state,
                launched_roles=tuple(sorted(record.role_handles)),
            )

    def _role_binding(self, handle: CoordinatorHandle, role: CoordinatorRole):
        self._require_authorized()
        parent, session_id, turn_id = self._current_binding(require_parent=True)
        if not isinstance(role, CoordinatorRole):
            raise CoordinatorRouteError("role must be a CoordinatorRole value.")
        record = self._record_for_handle(handle, session_id=session_id, turn_id=turn_id)
        with _REGISTRY.lock:
            child_handle = record.role_handles.get(role.value)
        if child_handle is None:
            raise CoordinatorRouteError("Coordinator role has not been launched.")
        from agent.subagent_lifecycle import SubagentLifecycleService

        return (
            SubagentLifecycleService(
                lambda: parent, authorization_resolver=self._authorization_resolver
            ),
            child_handle,
            self._public_role_handle(handle, role, child_handle),
        )

    @staticmethod
    def _public_role_handle(
        handle: CoordinatorHandle, role: CoordinatorRole, child_handle: Any
    ) -> CoordinatorRoleHandle:
        return CoordinatorRoleHandle(
            coordinator_id=handle.coordinator_id,
            role=role,
            role_run_id=str(child_handle.subagent_id),
        )

    def _record_for_handle(
        self, handle: CoordinatorHandle, *, session_id: str, turn_id: str
    ) -> _CoordinatorRecord:
        if not isinstance(handle, CoordinatorHandle):
            raise CoordinatorRouteError("Malformed coordinator handle.")
        expected = _sign_handle(dataclasses.replace(handle, capability=""))
        if not hmac.compare_digest(handle.capability, expected):
            raise CoordinatorRouteError("Coordinator handle verification failed.")
        if (
            handle.issuer_plugin_id != self.issuer_plugin_id
            or handle.parent_session_id != session_id
            or handle.parent_turn_id != turn_id
        ):
            raise CoordinatorRouteError("Coordinator handle binding does not match.")
        with _REGISTRY.lock:
            record = _REGISTRY.by_id.get(handle.coordinator_id)
            if record is None or record.handle != handle:
                raise CoordinatorRouteError("Unknown coordinator handle.")
            return record

    def _verify_route_capability(
        self,
        capability: RouteCapability,
        *,
        session_id: str,
        turn_id: str,
        user_message: str,
        coordinator_route: str,
        consultation_id: str,
        account_scope: AccountScope,
    ) -> None:
        if not isinstance(capability, RouteCapability):
            raise CoordinatorRouteError("Malformed route capability.")
        _validate_capability_shape(capability)
        expected_signature = _sign_route(dataclasses.replace(capability, signature=""))
        if not hmac.compare_digest(capability.signature, expected_signature):
            raise CoordinatorRouteError("Route capability verification failed.")
        expected = (
            self.issuer_plugin_id,
            session_id,
            turn_id,
            _message_hash(user_message),
            coordinator_route,
            consultation_id,
            account_scope,
        )
        actual = (
            capability.issuer_plugin_id,
            capability.parent_session_id,
            capability.parent_turn_id,
            capability.user_message_sha256,
            capability.coordinator_route,
            capability.consultation_id,
            capability.account_scope,
        )
        if actual != expected:
            raise CoordinatorRouteError("Route capability binding does not match.")
        now = float(self._clock())
        if capability.issued_at > now or capability.expires_at <= now:
            raise CoordinatorRouteError("Route capability is expired or not yet valid.")

    def _validate_route_request(
        self,
        user_message: str,
        coordinator_route: str,
        consultation_id: str,
        account_scope: AccountScope,
    ) -> None:
        if not isinstance(account_scope, AccountScope):
            raise CoordinatorRouteError("account_scope must be an AccountScope value.")
        if (
            not isinstance(user_message, str)
            or not user_message
            or len(user_message) > 64_000
        ):
            raise CoordinatorRouteError("user_message is invalid.")
        if not _valid_text(coordinator_route) or not _valid_text(consultation_id):
            raise CoordinatorRouteError("route or consultation id is invalid.")
        self._require_route_allowed(coordinator_route)

    def _require_route_allowed(self, coordinator_route: str) -> None:
        try:
            routes = self._allowed_routes_resolver()
        except Exception as exc:
            raise CoordinatorRouteError(
                "Coordinator route policy is unavailable."
            ) from exc
        if (
            not isinstance(routes, (tuple, list))
            or any(not _valid_text(route) for route in routes)
            or coordinator_route not in routes
        ):
            raise CoordinatorRouteError("Coordinator route is not allowlisted.")

    def _require_authorized(self) -> None:
        try:
            authorized = self._authorization_resolver()
        except Exception as exc:
            raise CoordinatorRouteError(
                "Coordinator service authorization failed."
            ) from exc
        if authorized is not True:
            raise CoordinatorRouteError(
                "Plugin is not authorized for coordinator service."
            )

    def _require_top_level_parent(self) -> Any:
        from agent.delegation_context import (
            is_delegated_child_context,
            is_delegated_child_process_context,
        )

        parent = self._parent_agent_resolver()
        depth = getattr(parent, "_delegate_depth", 0) if parent is not None else None
        if (
            parent is None
            or is_delegated_child_context()
            or is_delegated_child_process_context()
            or isinstance(depth, bool)
            or not isinstance(depth, int)
            or depth != 0
            or bool(getattr(parent, "_subagent_id", None))
        ):
            raise CoordinatorRouteError(
                "Coordinator service is available only from a top-level parent turn."
            )
        return parent

    def _current_binding(self, *, require_parent: bool = False) -> tuple[Any, str, str]:
        if self._gateway_binding is None:
            parent = self._require_top_level_parent()
            session_id, turn_id = _active_binding(parent)
            return parent, session_id, turn_id
        record = _gateway_dispatch_record(
            self._gateway_binding,
            issuer_plugin_id=self.issuer_plugin_id,
            clock=self._clock,
        )
        parent = _gateway_dispatch_parent(record) if require_parent else None
        if parent is not None:
            _gateway_dispatch_record(
                self._gateway_binding,
                issuer_plugin_id=self.issuer_plugin_id,
                clock=self._clock,
            )
            depth = getattr(parent, "_delegate_depth", 0)
            if (
                isinstance(depth, bool)
                or not isinstance(depth, int)
                or depth != 0
                or bool(getattr(parent, "_subagent_id", None))
            ):
                raise CoordinatorRouteError(
                    "Gateway coordinator parent is not a top-level Hermes agent."
                )
        return (
            parent,
            self._gateway_binding.parent_session_id,
            self._gateway_binding.parent_turn_id,
        )


class GatewayBoundCoordinatorService:
    """Coordinator facade that cannot be rebound to another gateway message."""

    def __init__(
        self, service: CoordinatorService, binding: GatewayDispatchBinding
    ) -> None:
        self._service = service
        self._binding = binding

    def reserve_consultation(
        self,
        *,
        user_message: str,
        coordinator_route: str,
        consultation_id: str,
        account_scope: AccountScope,
    ) -> CoordinatorHandle:
        if _message_hash(user_message) != self._binding.user_message_sha256:
            raise CoordinatorRouteError("Gateway user message binding does not match.")
        record = _gateway_dispatch_record(
            self._binding, issuer_plugin_id=self._service.issuer_plugin_id
        )
        with _GATEWAY_BINDINGS_LOCK:
            existing = record.consultation_binding
        requested = (coordinator_route, consultation_id, account_scope)
        if existing is not None and existing != requested:
            raise CoordinatorRouteError(
                "Gateway dispatch binding is already bound to another consultation."
            )
        capability = self._service.issue_route_capability(
            user_message=user_message,
            coordinator_route=coordinator_route,
            consultation_id=consultation_id,
            account_scope=account_scope,
        )
        _bind_gateway_consultation(
            self._binding,
            issuer_plugin_id=self._service.issuer_plugin_id,
            coordinator_route=coordinator_route,
            consultation_id=consultation_id,
            account_scope=account_scope,
        )
        return self._service.reserve_consultation(
            capability=capability,
            user_message=user_message,
            coordinator_route=coordinator_route,
            consultation_id=consultation_id,
            account_scope=account_scope,
        )

    def issue_team_mcp_binding_token(self, *args: Any, **kwargs: Any):
        return self._service.issue_team_mcp_binding_token(*args, **kwargs)

    def launch_role(self, *args: Any, **kwargs: Any):
        return self._service.launch_role(*args, **kwargs)

    def role_status(self, *args: Any, **kwargs: Any):
        return self._service.role_status(*args, **kwargs)

    def wait_role(self, *args: Any, **kwargs: Any):
        return self._service.wait_role(*args, **kwargs)

    def role_result(self, *args: Any, **kwargs: Any):
        return self._service.role_result(*args, **kwargs)

    def cancel_role(self, *args: Any, **kwargs: Any):
        return self._service.cancel_role(*args, **kwargs)

    def status(self, *args: Any, **kwargs: Any):
        return self._service.status(*args, **kwargs)


def reset_coordinator_registry_for_tests() -> None:
    """Clear process-local coordinator records for hermetic contract tests."""
    with _REGISTRY.lock:
        _REGISTRY.by_key.clear()
        _REGISTRY.by_id.clear()
        _REGISTRY.used_nonces.clear()
    with _GATEWAY_BINDINGS_LOCK:
        _GATEWAY_BINDINGS.clear()


def _active_binding(parent: Any) -> tuple[str, str]:
    session_id = str(getattr(parent, "session_id", "") or "")
    turn_id = str(getattr(parent, "_current_turn_id", "") or "")
    if not session_id or not turn_id:
        raise CoordinatorRouteError("An active parent session and turn are required.")
    return session_id, turn_id


def _message_hash(message: str) -> str:
    try:
        return hashlib.sha256(message.encode("utf-8")).hexdigest()
    except UnicodeEncodeError as exc:
        raise CoordinatorRouteError("user_message is not valid UTF-8.") from exc


def _valid_text(value: Any, max_chars: int = 256) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= max_chars


def _canonical_payload(value: Any, excluded: str) -> bytes:
    payload = {
        field.name: (
            getattr(value, field.name).value
            if isinstance(getattr(value, field.name), enum.Enum)
            else getattr(value, field.name)
        )
        for field in dataclasses.fields(value)
        if field.name != excluded
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sign_route(capability: RouteCapability) -> str:
    return hmac.new(
        _ROUTE_SECRET, _canonical_payload(capability, "signature"), hashlib.sha256
    ).hexdigest()


def _sign_handle(handle: CoordinatorHandle) -> str:
    return hmac.new(
        _HANDLE_SECRET, _canonical_payload(handle, "capability"), hashlib.sha256
    ).hexdigest()


def _sign_gateway_binding(binding: GatewayDispatchBinding) -> str:
    return hmac.new(
        _GATEWAY_BINDING_SECRET,
        _canonical_payload(binding, "signature"),
        hashlib.sha256,
    ).hexdigest()


def _team_mcp_signing_key() -> bytes:
    from agent.secret_scope import UnscopedSecretError, get_secret

    try:
        encoded = get_secret(_TEAM_MCP_HMAC_ENV)
    except UnscopedSecretError as exc:
        raise CoordinatorRouteError("Team MCP signing key is unavailable.") from exc
    if not isinstance(encoded, str) or not encoded:
        raise CoordinatorRouteError("Team MCP signing key is unavailable.")
    try:
        signing_key = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CoordinatorRouteError("Team MCP signing key is malformed.") from exc
    if (
        len(signing_key) < 32
        or base64.b64encode(signing_key).decode("ascii") != encoded
    ):
        raise CoordinatorRouteError("Team MCP signing key is malformed.")
    return signing_key


def _build_team_mcp_binding_token(
    payload: Mapping[str, Any], signing_key: bytes
) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    signature = hmac.new(signing_key, canonical, hashlib.sha256).digest()
    encoded_payload = base64.urlsafe_b64encode(canonical).rstrip(b"=").decode("ascii")
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"v1.{encoded_payload}.{encoded_signature}"


def _validate_capability_shape(capability: RouteCapability) -> None:
    if (
        not _valid_text(capability.issuer_plugin_id)
        or not _valid_text(capability.parent_session_id)
        or not _valid_text(capability.parent_turn_id)
        or not _valid_text(capability.coordinator_route)
        or not _valid_text(capability.consultation_id)
        or not isinstance(capability.account_scope, AccountScope)
        or not isinstance(capability.user_message_sha256, str)
        or len(capability.user_message_sha256) != 64
        or not isinstance(capability.nonce, str)
        or len(capability.nonce) < 16
        or not isinstance(capability.signature, str)
        or len(capability.signature) != 64
    ):
        raise CoordinatorRouteError("Malformed route capability.")
    for value in (capability.issued_at, capability.expires_at):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise CoordinatorRouteError("Malformed route capability.")
    if capability.expires_at <= capability.issued_at:
        raise CoordinatorRouteError("Malformed route capability.")
