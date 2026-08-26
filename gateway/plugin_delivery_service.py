"""Host-only at-most-once delivery for capability-bound gateway plugins.

The public plugin facade never receives this module's destination objects,
private binding handle, ledger claim token, adapter, or chat identifier.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from agent.route_capability import GatewayDeliveryResult
from gateway.platforms.base import SendOnceOutcome, SendOnceResult
from gateway.session import Platform

logger = logging.getLogger("gateway.plugin_delivery")

_METADATA_KEY = "_gateway_plugin_delivery_bindings_v1"
_MAX_BINDINGS_PER_SESSION = 32
_METADATA_LOCK = threading.RLock()
_RESOLVER_FIELDS = frozenset({
    "expires_at",
    "recovery_context_digest",
    "transport_owner_digest",
})


class _TransportRaised:
    pass


_TRANSPORT_RAISED = _TransportRaised()
_CONTEXT_FIELDS = (
    "platform",
    "profile",
    "scope_id",
    "guild_id",
    "chat_type",
    "chat_id",
    "chat_id_alt",
    "parent_chat_id",
    "thread_id",
    "user_id",
    "user_id_alt",
)


def _binding_digest(binding_handle: str) -> str:
    return hashlib.sha256(binding_handle.encode("utf-8", "strict")).hexdigest()


def _same_origin(left: Any, right: Any) -> bool:
    return all(
        getattr(left, field, None) == getattr(right, field, None)
        for field in _CONTEXT_FIELDS
    )


def _adapter_for_delivery_source(runner: Any, source: Any) -> Any:
    """Resolve live bound transport provenance without credential fallback."""
    transport_ref = getattr(source, "_transport_adapter_ref", None)
    if transport_ref is not None:
        if not callable(transport_ref) or transport_ref() is None:
            return None
        resolver = getattr(runner, "_registered_transport_adapter", None)
        if not callable(resolver):
            return None
        registered = resolver(source)
        if registered is None or registered is not transport_ref():
            return None
        resolved = runner._adapter_for_source(source)
        return resolved if resolved is registered else None
    return runner._adapter_for_source(source)


def _normalized_owner_profile(adapter: Any) -> str:
    raw = getattr(adapter, "_owner_profile", None)
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise ValueError("invalid transport owner profile")
    normalized = raw.strip()
    if not normalized or normalized == "default":
        return "default"
    if len(normalized) > 128:
        raise ValueError("invalid transport owner profile")
    return normalized


def _transport_owner_digest(binding_digest: str, *, platform: Any, adapter: Any) -> str:
    """Bind recovery to a stable adapter owner without persisting its name."""
    try:
        key = bytes.fromhex(binding_digest)
    except ValueError as exc:
        raise ValueError("invalid binding digest") from exc
    if len(key) != 32:
        raise ValueError("invalid binding digest")
    payload = json.dumps(
        {
            "version": 1,
            "platform": _context_value(platform),
            "owner_profile": _normalized_owner_profile(adapter),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8", "strict")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _recovery_adapter(
    runner: Any,
    *,
    platform: Any,
    binding_digest: str,
    transport_owner_digest: str,
) -> Any:
    matches: list[Any] = []
    seen: set[int] = set()
    registries = [getattr(runner, "adapters", None) or {}]
    registries.extend((getattr(runner, "_profile_adapters", None) or {}).values())
    for registry in registries:
        for candidate_platform, adapter in registry.items():
            if _context_value(candidate_platform) != _context_value(platform):
                continue
            identity = id(adapter)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                candidate_digest = _transport_owner_digest(
                    binding_digest,
                    platform=candidate_platform,
                    adapter=adapter,
                )
            except (TypeError, ValueError, UnicodeError):
                continue
            if hmac.compare_digest(candidate_digest, transport_owner_digest):
                matches.append(adapter)
    return matches[0] if len(matches) == 1 else None


def _context_value(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _recovery_context_digest(
    binding_digest: str,
    *,
    session_id: str,
    session_key: str,
    origin: Any,
    transport_owner_digest: str,
) -> str:
    """Bind one ledger row to its immutable, non-secret routing context."""
    if len(binding_digest) != 64:
        raise ValueError("invalid binding digest")
    try:
        key = bytes.fromhex(binding_digest)
    except ValueError as exc:
        raise ValueError("invalid binding digest") from exc
    if len(transport_owner_digest) != 64:
        raise ValueError("invalid transport owner digest")
    payload = {
        "version": 1,
        "session_id": session_id,
        "session_key": session_key,
        "transport_owner_digest": transport_owner_digest,
        "origin": {
            field: _context_value(getattr(origin, field, None))
            for field in _CONTEXT_FIELDS
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", "strict")
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class _ResolverMatch:
    entry: Any
    transport_owner_digest: str


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_live_resolver(value: Any, *, now: float) -> bool:
    if not isinstance(value, dict) or set(value) != _RESOLVER_FIELDS:
        return False
    expires_at = value.get("expires_at")
    if (
        isinstance(expires_at, bool)
        or not isinstance(expires_at, (int, float))
        or not math.isfinite(expires_at)
        or expires_at <= now
    ):
        return False
    return _is_sha256_digest(
        value.get("recovery_context_digest")
    ) and _is_sha256_digest(value.get("transport_owner_digest"))


def _persist_resolver(
    session_store: Any,
    *,
    session_key: str,
    binding_digest: str,
    recovery_context_digest: str,
    transport_owner_digest: str,
    expires_at: float,
) -> bool:
    """Durably add one digest-only restart resolver before ledger reserve."""
    if (
        not _is_sha256_digest(binding_digest)
        or not _is_sha256_digest(recovery_context_digest)
        or not _is_sha256_digest(transport_owner_digest)
        or isinstance(expires_at, bool)
        or not isinstance(expires_at, (int, float))
        or not math.isfinite(expires_at)
    ):
        return False
    with _METADATA_LOCK:
        now = time.time()
        if expires_at <= now:
            return False
        current = session_store.get_session_metadata(session_key, _METADATA_KEY, {})
        if current is None:
            current = {}
        if not isinstance(current, dict):
            return False
        mapping = {
            digest: value
            for digest, value in current.items()
            if _is_sha256_digest(digest) and _is_live_resolver(value, now=now)
        }
        if binding_digest not in mapping and len(mapping) >= _MAX_BINDINGS_PER_SESSION:
            return False
        mapping[binding_digest] = {
            "expires_at": expires_at,
            "recovery_context_digest": recovery_context_digest,
            "transport_owner_digest": transport_owner_digest,
        }
        return (
            session_store.set_session_metadata(session_key, _METADATA_KEY, mapping)
            is True
        )


def _cleanup_resolver(
    session_store: Any, *, session_key: str, binding_digest: str
) -> None:
    """Best-effort terminal cleanup; ledger state remains the send authority."""
    try:
        with _METADATA_LOCK:
            current = session_store.get_session_metadata(session_key, _METADATA_KEY, {})
            if not isinstance(current, dict) or binding_digest not in current:
                return
            mapping = dict(current)
            mapping.pop(binding_digest, None)
            session_store.set_session_metadata(session_key, _METADATA_KEY, mapping)
    except Exception:
        logger.warning("Gateway delivery resolver cleanup failed")


def _resolver_entries(
    runner: Any, pending: Any
) -> tuple[list[_ResolverMatch], list[_ResolverMatch], list[Any]]:
    """Return valid, expired, and all seen entries for one resolver digest."""
    valid: list[_ResolverMatch] = []
    expired: list[_ResolverMatch] = []
    seen: list[Any] = []
    now = time.time()
    for entry in runner.session_store.list_sessions():
        metadata = getattr(entry, "metadata", {})
        mapping = metadata.get(_METADATA_KEY) if isinstance(metadata, dict) else None
        value = (
            mapping.get(pending.binding_digest) if isinstance(mapping, dict) else None
        )
        if value is not None:
            seen.append(entry)
        if not isinstance(value, dict) or set(value) != {
            "expires_at",
            "recovery_context_digest",
            "transport_owner_digest",
        }:
            continue
        stored_context_digest = value.get("recovery_context_digest")
        if stored_context_digest != pending.recovery_context_digest:
            continue
        transport_owner_digest = value.get("transport_owner_digest")
        if not isinstance(transport_owner_digest, str):
            continue
        try:
            current_context_digest = _recovery_context_digest(
                pending.binding_digest,
                session_id=str(getattr(entry, "session_id", "") or ""),
                session_key=str(getattr(entry, "session_key", "") or ""),
                origin=getattr(entry, "origin", None),
                transport_owner_digest=transport_owner_digest,
            )
        except (TypeError, ValueError, UnicodeError):
            continue
        if current_context_digest != pending.recovery_context_digest:
            continue
        expires_at = value.get("expires_at")
        if (
            isinstance(expires_at, bool)
            or not isinstance(expires_at, (int, float))
            or not math.isfinite(float(expires_at))
        ):
            continue
        match = _ResolverMatch(
            entry=entry,
            transport_owner_digest=transport_owner_digest,
        )
        (valid if float(expires_at) > now else expired).append(match)
    return valid, expired, seen


async def _claim_recovered(ledger: Any, pending: Any) -> tuple[Any, Any]:
    record = await asyncio.to_thread(ledger.get_delivery, pending.delivery_id)
    if record is None:
        return None, None
    claim = await asyncio.to_thread(
        ledger.claim_pending_delivery,
        delivery_id=pending.delivery_id,
        plugin_id=record.plugin_id,
        binding_digest=pending.binding_digest,
        recovery_context_digest=pending.recovery_context_digest,
    )
    return record, claim


async def recover_pending(runner: Any) -> dict[str, int]:
    """Recover never-attempted plugin deliveries exactly once after startup.

    ``SEND_CLAIMED`` rows are quarantined by the ledger and never appear here.
    Unknown, expired, ambiguous, or adapter-less resolvers are terminal FAILED;
    no automatic transport is attempted for them.
    """
    from gateway import plugin_delivery_ledger as ledger

    pending_rows = await asyncio.to_thread(ledger.recover_after_restart)
    counts = {
        "pending": len(pending_rows),
        "delivered": 0,
        "failed": 0,
        "uncertain": 0,
    }
    for pending in pending_rows:
        valid, expired, seen = _resolver_entries(runner, pending)
        try:
            _record, claim = await _claim_recovered(ledger, pending)
        except Exception:
            counts["failed"] += 1
            continue
        if claim is None:
            continue

        resolution_error = None
        match = valid[0] if len(valid) == 1 else None
        entry = None if match is None else match.entry
        transport_owner_digest = None if match is None else match.transport_owner_digest
        if len(valid) > 1:
            resolution_error = "resolver_ambiguous"
        elif entry is None:
            resolution_error = "resolver_expired" if expired else "resolver_unknown"
        elif getattr(entry, "origin", None) is None:
            resolution_error = "resolver_origin_missing"
        elif getattr(entry.origin, "platform", None) != Platform.TELEGRAM:
            resolution_error = "unsupported_platform"

        adapter = None
        if resolution_error is None:
            assert entry is not None
            assert transport_owner_digest is not None
            adapter = _recovery_adapter(
                runner,
                platform=entry.origin.platform,
                binding_digest=pending.binding_digest,
                transport_owner_digest=transport_owner_digest,
            )
            if adapter is None:
                resolution_error = "adapter_unavailable"

        if resolution_error is not None:
            await asyncio.to_thread(
                ledger.mark_failed,
                claim.delivery_id,
                claim.claim_token,
                error=resolution_error,
            )
            counts["failed"] += 1
        else:
            assert entry is not None
            assert adapter is not None
            transport: Any
            try:
                metadata = runner._thread_metadata_for_target(
                    entry.origin.platform,
                    entry.origin.chat_id,
                    entry.origin.thread_id,
                    chat_type=entry.origin.chat_type,
                    reply_to_message_id=None,
                    adapter=adapter,
                )
                transport = await adapter.send_once(
                    str(entry.origin.chat_id),
                    claim.sanitized_content,
                    reply_to=None,
                    metadata=metadata,
                )
            except Exception:
                transport = _TRANSPORT_RAISED
            if transport is _TRANSPORT_RAISED or (
                isinstance(transport, SendOnceResult)
                and transport.outcome is SendOnceOutcome.DELIVERY_UNCERTAIN
            ):
                await asyncio.to_thread(
                    ledger.mark_uncertain,
                    claim.delivery_id,
                    claim.claim_token,
                    error="restart_transport_uncertain",
                )
                counts["uncertain"] += 1
            elif (
                isinstance(transport, SendOnceResult)
                and transport.outcome is SendOnceOutcome.DELIVERED
            ):
                await asyncio.to_thread(
                    ledger.mark_delivered,
                    claim.delivery_id,
                    claim.claim_token,
                    receipt_id=transport.message_id,
                )
                counts["delivered"] += 1
            else:
                await asyncio.to_thread(
                    ledger.mark_failed,
                    claim.delivery_id,
                    claim.claim_token,
                    error="restart_transport_failed",
                )
                counts["failed"] += 1

        for resolved in seen:
            _cleanup_resolver(
                runner.session_store,
                session_key=resolved.session_key,
                binding_digest=pending.binding_digest,
            )
    logger.info(
        "Gateway delivery recovery pending=%d delivered=%d failed=%d uncertain=%d",
        counts["pending"],
        counts["delivered"],
        counts["failed"],
        counts["uncertain"],
    )
    return counts


def _existing_result(record: Any) -> GatewayDeliveryResult:
    state = str(getattr(getattr(record, "state", None), "value", "FAILED"))
    if state == "DELIVERED":
        outcome = "DELIVERED"
    elif state == "DELIVERY_UNCERTAIN":
        outcome = "DELIVERY_UNCERTAIN"
    elif state == "SEND_CLAIMED":
        outcome = "IN_PROGRESS"
    else:
        outcome = "FAILED"
    return GatewayDeliveryResult(
        outcome=outcome,
        transport_attempted=False,
        delivery_id=str(getattr(record, "delivery_id", "") or "") or None,
        error_code="duplicate_delivery",
    )


async def _cancel_pending_record(ledger: Any, record: Any, *, reason: str) -> bool:
    """Make an exact pre-transport reservation terminal before resolver cleanup."""
    operation = asyncio.create_task(
        asyncio.to_thread(
            ledger.cancel_pending_delivery,
            delivery_id=record.delivery_id,
            plugin_id=record.plugin_id,
            binding_digest=record.binding_digest,
            recovery_context_digest=record.recovery_context_digest,
            reason=reason,
        )
    )
    try:
        return await asyncio.shield(operation)
    except Exception:
        logger.warning("Gateway pending delivery cancellation failed")
        return False


async def _fail_claimed_record(ledger: Any, claim: Any, *, reason: str) -> bool:
    """Terminalize a claim won before adapter transport began."""
    operation = asyncio.create_task(
        asyncio.to_thread(
            ledger.mark_failed,
            claim.delivery_id,
            claim.claim_token,
            error=reason,
        )
    )
    try:
        return await asyncio.shield(operation)
    except Exception:
        logger.warning("Gateway claimed delivery cancellation failed")
        return False


async def deliver_once(
    runner: Any,
    *,
    plugin_id: str,
    binding_handle: str,
    binding_expires_at: float,
    session_id: str,
    session_key: str,
    source: Any,
    delivery_key: str,
    content: str,
) -> GatewayDeliveryResult:
    """Reserve, atomically claim, and attempt exactly one adapter transport."""
    from gateway import plugin_delivery_ledger as ledger

    if getattr(source, "platform", None) != Platform.TELEGRAM:
        return GatewayDeliveryResult("FAILED", False, error_code="unsupported_platform")

    transport: Any
    digest: str | None = None
    record: Any = None
    claim: Any = None
    try:
        entry = runner.session_store.lookup_by_session_key(session_key)
        if (
            entry is None
            or str(getattr(entry, "session_id", "") or "") != session_id
            or getattr(entry, "origin", None) is None
            or not _same_origin(entry.origin, source)
        ):
            return GatewayDeliveryResult("FAILED", False, error_code="session_not_live")
        adapter = _adapter_for_delivery_source(runner, source)
        if adapter is None:
            return GatewayDeliveryResult(
                "FAILED", False, error_code="adapter_unavailable"
            )
        sanitized = ledger.sanitize_delivery_content(content).text
        digest = _binding_digest(binding_handle)
        transport_owner_digest = _transport_owner_digest(
            digest,
            platform=source.platform,
            adapter=adapter,
        )
        recovery_context_digest = _recovery_context_digest(
            digest,
            session_id=session_id,
            session_key=session_key,
            origin=entry.origin,
            transport_owner_digest=transport_owner_digest,
        )
        if not _persist_resolver(
            runner.session_store,
            session_key=session_key,
            binding_digest=digest,
            recovery_context_digest=recovery_context_digest,
            transport_owner_digest=transport_owner_digest,
            expires_at=binding_expires_at,
        ):
            return GatewayDeliveryResult(
                "FAILED", False, error_code="resolver_persist_failed"
            )
        reserve_task = asyncio.create_task(
            asyncio.to_thread(
                ledger.reserve_delivery,
                plugin_id=plugin_id,
                binding_handle=binding_handle,
                delivery_key=delivery_key,
                sanitized_content=sanitized,
                recovery_context_digest=recovery_context_digest,
            )
        )
        try:
            record = await asyncio.shield(reserve_task)
        except asyncio.CancelledError:
            record = await asyncio.shield(reserve_task)
            await _cancel_pending_record(ledger, record, reason="delivery_cancelled")
            raise
        claim_task = asyncio.create_task(
            asyncio.to_thread(
                ledger.claim_for_send,
                plugin_id=plugin_id,
                binding_handle=binding_handle,
                delivery_key=delivery_key,
            )
        )
        try:
            claim = await asyncio.shield(claim_task)
        except asyncio.CancelledError:
            claim = await asyncio.shield(claim_task)
            if claim is None:
                await _cancel_pending_record(
                    ledger, record, reason="delivery_cancelled"
                )
            else:
                await _fail_claimed_record(ledger, claim, reason="delivery_cancelled")
            raise
    except asyncio.CancelledError:
        if digest is not None:
            _cleanup_resolver(
                runner.session_store,
                session_key=session_key,
                binding_digest=digest,
            )
        raise
    except Exception:
        if record is not None and claim is None:
            await _cancel_pending_record(
                ledger, record, reason="delivery_prepare_failed"
            )
        if digest is not None:
            _cleanup_resolver(
                runner.session_store,
                session_key=session_key,
                binding_digest=digest,
            )
        logger.warning("Gateway delivery preparation failed")
        return GatewayDeliveryResult(
            "FAILED", False, error_code="delivery_prepare_failed"
        )

    if claim is None:
        current = await asyncio.to_thread(ledger.get_delivery, record.delivery_id)
        result = _existing_result(current or record)
        if result.outcome in {"DELIVERED", "FAILED", "DELIVERY_UNCERTAIN"}:
            _cleanup_resolver(
                runner.session_store,
                session_key=session_key,
                binding_digest=digest,
            )
        return result

    try:
        current_adapter = _adapter_for_delivery_source(runner, source)
        if current_adapter is not adapter:
            transport = None
        else:
            metadata = runner._thread_metadata_for_target(
                source.platform,
                source.chat_id,
                source.thread_id,
                chat_type=source.chat_type,
                reply_to_message_id=None,
                adapter=adapter,
            )
            transport = await adapter.send_once(
                str(source.chat_id),
                claim.sanitized_content,
                reply_to=None,
                metadata=metadata,
            )
    except asyncio.CancelledError:
        _cleanup_resolver(
            runner.session_store,
            session_key=session_key,
            binding_digest=digest,
        )
        raise
    except Exception:
        transport = _TRANSPORT_RAISED

    outcome = "FAILED"
    error_code = "transport_failed"
    try:
        if transport is _TRANSPORT_RAISED:
            outcome = "DELIVERY_UNCERTAIN"
            error_code = "transport_exception"
            marked = await asyncio.to_thread(
                ledger.mark_uncertain,
                claim.delivery_id,
                claim.claim_token,
                error=error_code,
            )
        elif transport is None:
            marked = await asyncio.to_thread(
                ledger.mark_failed,
                claim.delivery_id,
                claim.claim_token,
                error="adapter_unavailable",
            )
            error_code = "adapter_unavailable"
        elif (
            isinstance(transport, SendOnceResult)
            and transport.outcome is SendOnceOutcome.DELIVERED
        ):
            outcome = "DELIVERED"
            error_code = None
            marked = await asyncio.to_thread(
                ledger.mark_delivered,
                claim.delivery_id,
                claim.claim_token,
                receipt_id=transport.message_id,
            )
        elif (
            isinstance(transport, SendOnceResult)
            and transport.outcome is SendOnceOutcome.DELIVERY_UNCERTAIN
        ):
            outcome = "DELIVERY_UNCERTAIN"
            error_code = "transport_uncertain"
            marked = await asyncio.to_thread(
                ledger.mark_uncertain,
                claim.delivery_id,
                claim.claim_token,
                error=error_code,
            )
        else:
            marked = await asyncio.to_thread(
                ledger.mark_failed,
                claim.delivery_id,
                claim.claim_token,
                error="transport_failed",
            )
        if marked is not True:
            outcome = "DELIVERY_UNCERTAIN"
            error_code = "terminal_update_failed"
    except Exception:
        outcome = "DELIVERY_UNCERTAIN"
        error_code = "terminal_update_failed"

    _cleanup_resolver(
        runner.session_store,
        session_key=session_key,
        binding_digest=digest,
    )
    logger.info("Gateway delivery finished outcome=%s", outcome)
    return GatewayDeliveryResult(
        outcome=outcome,
        transport_attempted=transport is not None,
        delivery_id=claim.delivery_id,
        error_code=error_code,
    )
