"""Narrow read-only access to Hermes-persisted investment evidence.

Uses the guarded line reader and bounded character pages for long lines.
Only the active profile's spillover directory is accepted; arbitrary
filesystem reads and writes are not exposed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time

from agent.redact import redact_sensitive_text
from agent.file_safety import get_read_block_error
from tools.registry import registry, tool_error
from tools.tool_result_storage import (
    SPILLOVER_MAX_AGE_HOURS,
    _INVESTMENT_EVIDENCE_PRODUCERS,
    _READ_ONLY_STRATEGY_PRODUCER,
    get_spillover_dir,
)

_CHARACTER_PAGE_SIZE = 20_000
_MAX_CHARACTER_SOURCE_BYTES = 8 * 1024 * 1024

_INVESTMENT_EVIDENCE_SCHEMA = {
    "name": "investment_evidence",
    "description": (
        "Read a paginated page from a Hermes-generated persisted investment "
        "tool result. Only the active profile cache/spillover directory is "
        "allowed; the operation is read-only and preserves the provider's "
        "source/as-of/freshness fields. Missing or expired evidence is an "
        "explicit degraded/unknown result and never triggers a provider retry. "
        "Long lines automatically use character pages; continue with the returned "
        "next_character_offset until truncated is false. Character paging accepts "
        "UTF-8 sources up to 8 MiB; larger sources return degraded, not partial success."
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
            "character_offset": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "1-indexed position in the redacted text for a 20000-character "
                    "page. Use next_character_offset from the previous result; "
                    "do not increment the line offset for character pages."
                ),
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


def _read_provenance(path: Path) -> dict[str, object]:
    sidecar = path.with_name(f"{path.name}.meta.json")
    descriptor = os.open(sidecar, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        sidecar_stat = os.fstat(descriptor)
        if not stat.S_ISREG(sidecar_stat.st_mode) or sidecar_stat.st_size > 64 * 1024:
            raise ValueError("metadata is not a bounded regular file")
        raw = os.read(descriptor, sidecar_stat.st_size + 1)
        if len(raw) != sidecar_stat.st_size:
            raise ValueError("metadata changed during the read")
    finally:
        os.close(descriptor)
    metadata = json.loads(raw.decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata is not an object")
    return metadata


def _provenance_error(
    path: Path,
    metadata: dict[str, object],
    *,
    session_id: str | None,
    requester_id: str | None,
) -> str | None:
    """Require storage-issued, non-account provenance bound to the bytes."""
    try:
        if metadata.get("version") != 1 or metadata.get("filename") != path.name:
            raise ValueError("metadata does not identify this evidence")
        producer = metadata.get("tool_name")
        if producer == _READ_ONLY_STRATEGY_PRODUCER:
            if metadata.get("strategy_read_only") is not True:
                raise ValueError("strategy requires explicit read-only execution provenance")
        elif producer not in _INVESTMENT_EVIDENCE_PRODUCERS:
            raise ValueError("producer is not authorized investment evidence")
        if not isinstance(metadata.get("tool_use_id"), str) or not metadata["tool_use_id"]:
            raise ValueError("tool call provenance is missing")
        if not session_id:
            raise ValueError("request session is missing")
        stored_session = metadata.get("session_id")
        if not isinstance(stored_session, str) or not stored_session:
            raise ValueError("evidence session provenance is missing")
        if session_id != stored_session:
            # Lineage is runtime-owned session state, not a claim in the
            # file sidecar or model-supplied tool arguments.
            from hermes_constants import get_hermes_home

            database = get_hermes_home() / "state.db"
            connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=1)
            try:
                lineage = dict(connection.execute(
                    "SELECT id, parent_session_id FROM sessions WHERE id IN (?, ?)",
                    (session_id, stored_session),
                ).fetchall())
            finally:
                connection.close()
            if (session_id not in lineage or stored_session not in lineage or
                    not (lineage[session_id] == stored_session or lineage[stored_session] == session_id)):
                raise ValueError("evidence belongs to another session")
        elif requester_id:
            stored_requester = metadata.get("requester_id")
            if stored_requester and stored_requester != requester_id:
                raise ValueError("evidence belongs to another requester")
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        return f"investment_evidence: evidence provenance unavailable ({exc})"
    return None


def _read_pinned_source(path: Path) -> tuple[bytes, dict[str, object]]:
    """Read one bounded, no-follow byte snapshot and its storage proof."""
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("evidence paging requires local descriptor-relative reads")
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_CHARACTER_SOURCE_BYTES:
            raise ValueError("evidence must be a regular file of at most 8 MiB")
        raw = handle.read(_MAX_CHARACTER_SOURCE_BYTES + 1)
        after = os.fstat(handle.fileno())
        current = path.stat(follow_symlinks=False)
        fingerprint = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(current) or len(raw) != after.st_size:
            raise ValueError("evidence changed during the read")
    metadata = _read_provenance(path)
    if metadata.get("content_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("evidence provenance does not match the read bytes")
    return raw, metadata


def _character_page(
    path: Path,
    filename: str,
    offset: int,
    *,
    session_id: str | None,
    requester_id: str | None,
) -> str:
    # Pin every directory and the leaf without following links, including
    # replacements after _safe_reference. Redact before slicing so a secret
    # crossing a page boundary cannot lose the context needed to redact it.
    raw, provenance = _read_pinned_source(path)
    provenance_error = _provenance_error(
        path, provenance, session_id=session_id, requester_id=requester_id,
    )
    if provenance_error:
        raise ValueError(provenance_error)
    text = raw.decode("utf-8")
    if "\x00" in text:
        raise ValueError("evidence is not UTF-8 text")
    text = redact_sensitive_text(text, force=True, file_read=True, redact_url_credentials=True)
    return _render_character_page(text, path, filename, offset)


def _render_character_page(text: str, path: Path, filename: str, offset: int) -> str:
    if offset > len(text) + 1:
        raise ValueError("character_offset exceeds the redacted evidence")
    start = offset - 1
    content = text[start:start + _CHARACTER_PAGE_SIZE]
    end = start + len(content)
    return json.dumps({
        "success": True, "status": "ok", "unknown": False,
        "content": content, "pagination_mode": "characters",
        "character_offset": offset, "total_characters": len(text),
        "truncated": end < len(text),
        "next_character_offset": end + 1 if end < len(text) else None,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        **_metadata(path, filename),
    }, ensure_ascii=False)


def _guarded_character_page(path: Path, filename: str, offset: int, *, session_id: str | None, requester_id: str | None) -> str:
    try:
        return _character_page(path, filename, offset, session_id=session_id, requester_id=requester_id)
    except (OSError, ValueError, NotImplementedError) as exc:
        return json.dumps({"success": False, "status": "degraded", "unknown": True,
                           "error": str(exc), "provider_retry": False})


def _handle_investment_evidence(
    args: dict,
    task_id: str | None = None,
    session_id: str | None = None,
    requester_id: str | None = None,
    **_: object,
) -> str:
    for key, default, maximum in (("offset", 1, None), ("limit", 500, 2000), ("character_offset", 1, None)):
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
    try:
        provenance = _read_provenance(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        provenance = None
        provenance_error = f"investment_evidence: evidence provenance unavailable ({exc})"
    else:
        provenance_error = _provenance_error(
            path, provenance, session_id=session_id, requester_id=requester_id or task_id,
        )
    if provenance_error:
        return json.dumps({"success": False, "status": "degraded", "unknown": True,
                           "error": provenance_error, "provider_retry": False})
    block_error = get_read_block_error(str(path))
    if block_error:
        return json.dumps({"success": False, "status": "degraded", "unknown": True,
                           "error": block_error, "provider_retry": False})
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
    if "character_offset" in args:
        return _guarded_character_page(path, filename, args["character_offset"], session_id=session_id, requester_id=requester_id or task_id)
    try:
        raw, provenance = _read_pinned_source(path)
        provenance_error = _provenance_error(
            path, provenance, session_id=session_id, requester_id=requester_id or task_id,
        )
        if provenance_error:
            raise ValueError(provenance_error)
        text = raw.decode("utf-8")
        if "\x00" in text:
            raise ValueError("evidence is not UTF-8 text")
        text = redact_sensitive_text(text, force=True, file_read=True, redact_url_credentials=True)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return json.dumps({"success": False, "status": "degraded", "unknown": True,
                           "error": str(exc), "provider_retry": False})
    lines = text.splitlines(keepends=True)
    if len(text) > _CHARACTER_PAGE_SIZE and any(len(line) > _CHARACTER_PAGE_SIZE for line in lines):
        return _render_character_page(text, path, filename, 1)
    offset = args.get("offset", 1)
    limit = args.get("limit", 500)
    content = "".join(lines[offset - 1:offset - 1 + limit])
    return json.dumps({
        "success": True, "status": "ok", "unknown": False, "content": content,
        "offset": offset, "limit": limit, "total_lines": len(lines),
        "truncated": offset - 1 + limit < len(lines),
        **_metadata(path, filename),
    }, ensure_ascii=False)


registry.register(
    name="investment_evidence",
    toolset="investment_evidence",
    schema=_INVESTMENT_EVIDENCE_SCHEMA,
    handler=_handle_investment_evidence,
    emoji="📚",
    max_result_size_chars=100_000,
)
