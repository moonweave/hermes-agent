"""Narrow read-only access to Hermes-persisted investment evidence.

This projects the existing spillover store through the guarded read_file
implementation. It intentionally accepts only files in the active profile's
spillover directory; it does not expose arbitrary filesystem reads or writes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from tools.file_tools import read_file_tool
from tools.registry import registry, tool_error
from tools.tool_result_storage import SPILLOVER_MAX_AGE_HOURS, get_spillover_dir


_INVESTMENT_EVIDENCE_SCHEMA = {
    "name": "investment_evidence",
    "description": (
        "Read a paginated page from a Hermes-generated persisted investment "
        "tool result. Only the active profile cache/spillover directory is "
        "allowed; the operation is read-only and preserves the provider's "
        "source/as-of/freshness fields. Missing or expired evidence is an "
        "explicit degraded/unknown result and never triggers a provider retry."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reference": {
                "type": "string",
                "description": (
                    "The exact persisted-result path from the tool preview, "
                    "or its filename under the active spillover directory."
                ),
            },
            "offset": {
                "type": "integer",
                "description": "1-indexed line offset (default 1).",
                "default": 1,
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines (default 500, maximum 2000).",
                "default": 500,
                "minimum": 1,
                "maximum": 2000,
            },
        },
        "required": ["reference"],
    },
}


def _spillover_root() -> Path:
    """Resolve the canonical root independently for each call."""
    return get_spillover_dir().resolve(strict=False)


def _safe_reference(reference: object) -> tuple[Path, str] | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    root = _spillover_root()
    raw = Path(reference.strip()).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    # Reject symlinked path components, not merely a symlink leaf. This keeps
    # the narrow root meaningful even if a parent is replaced between calls.
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    current = root
    for component in relative.parts:
        current = current / component
        try:
            if current.is_symlink():
                return None
        except OSError:
            return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    if resolved == root or not resolved.is_file():
        return None
    return resolved, resolved.name


def _expired(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime > SPILLOVER_MAX_AGE_HOURS * 3600
    except OSError:
        return True


def _metadata(path: Path, filename: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "source_filename": filename,
        "evidence_ref": f"hermes://spillover/{filename}",
        "persisted_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "wrapper_freshness": "persistence_time_only",
        "freshness_authority": "provider_payload_as_of",
        "provider_retry": False,

    }


def _handle_investment_evidence(args: dict, **_: object) -> str:
    for key, default, maximum in (("offset", 1, None), ("limit", 500, 2000)):
        value = args.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (maximum and value > maximum):
            return json.dumps({"success": False, "status": "degraded", "unknown": True,
                               "error": f"invalid {key}", "provider_retry": False})
    safe = _safe_reference(args.get("reference"))
    if safe is None:
        return json.dumps({"success": False, "status": "degraded", "unknown": True,
            "error": "investment_evidence: missing or unsafe file under the active profile cache/spillover root",
            "provider_retry": False})
    path, filename = safe
    if _expired(path):
        return json.dumps(
            {"success": False, "status": "degraded", "unknown": True,
             "reason": "persisted evidence is missing or expired",
             **_metadata(path, filename)},
            ensure_ascii=False,
        )
    # Revalidate before dispatching to the existing guarded reader. The
    # reader supplies path/device/sensitive-text/redaction/pagination guards.
    checked = _safe_reference(str(path))
    if checked is None or checked[0] != path:
        return tool_error("investment_evidence: evidence path changed or is unsafe")
    result = read_file_tool(
        path=str(path),
        offset=args.get("offset", 1),
        limit=args.get("limit", 500),
        task_id="investment-evidence",
    )
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return json.dumps(
            {"success": True, "status": "ok", "content": result,
             **_metadata(path, filename)},
            ensure_ascii=False,
        )
    if isinstance(payload, dict):
        payload.update(_metadata(path, filename))
        payload.setdefault("status", "ok" if not payload.get("error") else "degraded")
        payload.setdefault("success", not bool(payload.get("error")))
        payload["unknown"] = bool(payload.get("error"))
        if payload.get("truncated_by") == "bytes" and "remainder is not retrievable" in payload.get("hint", ""):
            payload.update(success=False, status="degraded", unknown=True,
                           reason="a single evidence line exceeds the read budget; full coverage unavailable")
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="investment_evidence",
    toolset="investment_evidence",
    schema=_INVESTMENT_EVIDENCE_SCHEMA,
    handler=_handle_investment_evidence,
    emoji="📚",
    max_result_size_chars=100_000,
)
