"""Capability-gated host coordinator jobs for trusted plugins.

Reserving a consultation creates only a host-owned record.  LLM agents are
created separately for named roles and are always first-level, leaf, and
strictly tool-less.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from typing import Any, Callable, Mapping, Optional


class CoordinatorRouteError(ValueError):
    """A coordinator operation cannot be safely accepted."""


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
    ) -> None:
        if not _valid_text(issuer_plugin_id):
            raise CoordinatorRouteError("issuer_plugin_id is invalid.")
        self.issuer_plugin_id = issuer_plugin_id
        self._parent_agent_resolver = parent_agent_resolver
        self._allowed_routes_resolver = allowed_routes_resolver
        self._authorization_resolver = authorization_resolver
        self._clock = clock

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
        parent = self._require_top_level_parent()
        session_id, turn_id = _active_binding(parent)
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
        parent = self._require_top_level_parent()
        session_id, turn_id = _active_binding(parent)
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
        parent = self._require_top_level_parent()
        session_id, turn_id = _active_binding(parent)
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

            lifecycle = SubagentLifecycleService(lambda: parent)
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
        parent = self._require_top_level_parent()
        session_id, turn_id = _active_binding(parent)
        record = self._record_for_handle(handle, session_id=session_id, turn_id=turn_id)
        with _REGISTRY.lock:
            return CoordinatorStatus(
                handle=record.handle,
                state=record.state,
                launched_roles=tuple(sorted(record.role_handles)),
            )

    def _role_binding(self, handle: CoordinatorHandle, role: CoordinatorRole):
        self._require_authorized()
        parent = self._require_top_level_parent()
        session_id, turn_id = _active_binding(parent)
        if not isinstance(role, CoordinatorRole):
            raise CoordinatorRouteError("role must be a CoordinatorRole value.")
        record = self._record_for_handle(handle, session_id=session_id, turn_id=turn_id)
        with _REGISTRY.lock:
            child_handle = record.role_handles.get(role.value)
        if child_handle is None:
            raise CoordinatorRouteError("Coordinator role has not been launched.")
        from agent.subagent_lifecycle import SubagentLifecycleService

        return (
            SubagentLifecycleService(lambda: parent),
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


def reset_coordinator_registry_for_tests() -> None:
    """Clear process-local coordinator records for hermetic contract tests."""
    with _REGISTRY.lock:
        _REGISTRY.by_key.clear()
        _REGISTRY.by_id.clear()
        _REGISTRY.used_nonces.clear()


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
