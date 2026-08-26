"""Durable, at-most-once transport ledger for capability-backed plugins.

The ledger deliberately does not persist the opaque host-issued binding or the
caller delivery key.  Stable SHA-256 digests are sufficient for idempotency and
leave capability material outside SQLite.  A claimed delivery is never made
pending again: a restart converts it to ``DELIVERY_UNCERTAIN`` because the
platform may have accepted the request before the process lost its receipt.
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from hermes_constants import get_hermes_home

CONTENT_RETENTION_SECONDS = 24 * 60 * 60
METADATA_RETENTION_SECONDS = 30 * 24 * 60 * 60

_LOCK = threading.RLock()
_MAX_DATABASE_BYTES = 64 * 1024 * 1024
_MAX_CANONICAL_INPUT_BYTES = 200_000
_MAX_CANONICALIZATION_PASSES = 8
_PLUGIN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_DELIVERY_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ACCOUNT_VALUE_PART = r"[A-Za-z0-9*._-]*\d[A-Za-z0-9*._-]*"
_ACCOUNT_LABEL_RE = re.compile(
    r"(?i)(account(?:[\s_-]*(?:number|no\.?))?|계좌(?:\s*번호)?)"
    rf"\s*[:=]\s*(?:{_ACCOUNT_VALUE_PART}(?:\s+{_ACCOUNT_VALUE_PART})*)"
)
_TOKEN_LABEL_RE = re.compile(
    r"(?i)(?:access[\s_-]*token|auth[\s_-]*token|token)\s*[:=]\s*[^\s;]+"
)
_SENSITIVE_TEXT_LABEL_RE = re.compile(
    r"(?im)(prompt|full[\s_-]*role(?:[\s_-]*output)?|hmac|프롬프트)"
    r"\s*[:=]\s*[^\r\n;]+"
)
_POSITION_VALUE = r"[+-]?\d+(?:\.\d+)?(?:\s*,\s*\d+)*(?:\s*%)?(?:\s*(?:원|주|KRW))?"
_POSITION_VALUE_LABEL_RE = re.compile(
    r"(?i)(보유\s*수량|보유\s*량|holding[\s_-]*quantity|holdings?|quantity|"
    r"average[\s_-]*price|avg[\s_-]*price|purchase[\s_-]*price|"
    r"acquisition[\s_-]*price|entry[\s_-]*price|cost[\s_-]*basis|"
    r"current[\s_-]*price|evaluation[\s_-]*price|현재\s*가|매입\s*가|매수\s*가|"
    r"평균\s*단가|평단|"
    r"평가\s*금액|평가\s*액|가격|price|평가\s*손익률|손익률|수익률|"
    r"profit[\s_-]*loss|pnl|수량|실현\s*손익|평가\s*손익|총\s*손익|hmac)"
    rf"\s*[:=]\s*(?:{_POSITION_VALUE}|[^\s;]+)"
)
_STRUCTURED_FRAGMENT_RE = re.compile(r"(?s)```.*?```|\{.*?\}|\[.*?\]")
_QUOTED_ASSIGNMENT_KEY_RE = re.compile(r"""["']([^"'\\\r\n]{1,100})["']\s*[:=]""")
_BARE_ASSIGNMENT_KEY_RE = re.compile(
    r"(?im)(?:^|[,\{\[]\s*)([A-Za-z가-힣][A-Za-z0-9가-힣\s_.-]{0,99})\s*[:=]"
)
_LINE_ASSIGNMENT_KEY_RE = re.compile(
    r"""^\s*(?:"([^"\\\r\n]{1,100})"|'([^'\\\r\n]{1,100})'|"""
    r"([A-Za-z가-힣][A-Za-z0-9가-힣\s_.-]{0,99}))\s*[:=]\s*\S"
)
_SCHEMA_KEY_SEPARATOR_RE = re.compile(r"[\s_.-]+")
_CANONICAL_ESCAPE_RE = re.compile(
    r"""\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|["'\\/bfnrt])"""
)
_KNOWN_SENSITIVE_SCHEMA_KEYS = frozenset({
    "account",
    "accountid",
    "accountno",
    "accountnumber",
    "acquisitionprice",
    "averageprice",
    "avgprice",
    "costbasis",
    "currentprice",
    "entryprice",
    "evaluationprice",
    "evidence",
    "fullroleoutput",
    "fullrole",
    "hmac",
    "holding",
    "holdingquantity",
    "holdings",
    "maxweight",
    "officialauthority",
    "officialdecision",
    "officialmaxweight",
    "officialpermission",
    "officialtrigger",
    "officialverdict",
    "opinion",
    "permission",
    "pnl",
    "portfolio",
    "price",
    "profitloss",
    "prompt",
    "purchaseprice",
    "quantity",
    "role",
    "roleoutput",
    "trigger",
    "verdict",
    "계좌",
    "계좌번호",
    "근거",
    "가격",
    "매수가",
    "매입가",
    "보유량",
    "보유수량",
    "수량",
    "수익률",
    "손익률",
    "실현손익",
    "의견",
    "총손익",
    "최대비중",
    "트리거",
    "판정",
    "평가금액",
    "평가액",
    "평가손익",
    "평가손익률",
    "평균단가",
    "평단",
    "프롬프트",
    "현재가",
    "허가",
    "역할",
})
_ROLE_SCHEMA_KEYS = frozenset({"role", "역할"})
_ROLE_AUTHORITY_SCHEMA_KEYS = frozenset({
    "evidence",
    "fullrole",
    "fullroleoutput",
    "opinion",
    "prompt",
    "roleoutput",
    "근거",
    "의견",
    "프롬프트",
})
_MAX_SENSITIVE_ASSIGNMENT_LINE_GAP = 2


class DeliveryState(str, Enum):
    PENDING = "PENDING"
    SEND_CLAIMED = "SEND_CLAIMED"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    DELIVERY_UNCERTAIN = "DELIVERY_UNCERTAIN"


class DeliveryConflictError(ValueError):
    """The same logical delivery identity was reused with different content."""


class LedgerTrustError(RuntimeError):
    """The profile directory or SQLite file set failed its trust contract."""


class LedgerCommitUncertain(LedgerTrustError):
    """A commit completed on the held inode after its canonical name changed."""


class LedgerSanitizationError(ValueError):
    """Sensitive-content sanitization could not complete safely."""


@dataclass(frozen=True)
class DeliveryRecord:
    delivery_id: str
    plugin_id: str
    binding_digest: str
    delivery_key_digest: str
    recovery_context_digest: str | None
    state: DeliveryState
    sanitized_content: str | None
    created_at: float
    updated_at: float
    receipt_id: str | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class DeliveryClaim:
    delivery_id: str
    claim_token: str
    sanitized_content: str


@dataclass(frozen=True)
class PendingDelivery:
    delivery_id: str
    content: str
    binding_digest: str
    recovery_context_digest: str


@dataclass(frozen=True)
class SanitizedDeliveryContent:
    text: str


def _db_path() -> Path:
    return get_hermes_home() / "plugin-delivery.sqlite3"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "strict")).hexdigest()


def _looks_like_structured_payload(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return False
    if isinstance(parsed, (dict, list)):
        return True
    return False


def _decode_canonical_escape(match: re.Match[str]) -> str:
    token = match.group(0)[1:]
    if token.startswith(("u", "U")):
        codepoint = int(token[1:], 16)
        if 0xD800 <= codepoint <= 0xDFFF or codepoint > 0x10FFFF:
            raise LedgerSanitizationError("invalid unicode escape in delivery content")
        return chr(codepoint)
    return {
        '"': '"',
        "'": "'",
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }[token]


def _canonicalize_for_schema_detection(value: str) -> str:
    try:
        encoded_size = len(value.encode("utf-8", "strict"))
    except UnicodeError as exc:
        raise LedgerSanitizationError("delivery content encoding is invalid") from exc
    if encoded_size > _MAX_CANONICAL_INPUT_BYTES:
        raise LedgerSanitizationError("delivery content exceeds canonicalization limit")

    canonical = value
    for _ in range(_MAX_CANONICALIZATION_PASSES):
        updated = _CANONICAL_ESCAPE_RE.sub(_decode_canonical_escape, canonical)
        if updated == canonical:
            return canonical
        canonical = updated
    if _CANONICAL_ESCAPE_RE.search(canonical):
        raise LedgerSanitizationError("delivery escape nesting exceeds limit")
    return canonical


def _canonical_schema_key(value: str) -> str:
    return _SCHEMA_KEY_SEPARATOR_RE.sub("", value).casefold()


def _contains_unfenced_sensitive_assignment_cluster(value: str) -> bool:
    assignments: list[tuple[int, str]] = []
    for line_number, line in enumerate(value.splitlines()):
        match = _LINE_ASSIGNMENT_KEY_RE.match(line)
        if match is None:
            continue
        raw_key = next(group for group in match.groups() if group is not None)
        key = _canonical_schema_key(raw_key)
        if key in _KNOWN_SENSITIVE_SCHEMA_KEYS:
            assignments.append((line_number, key))

    keys = {key for _, key in assignments}
    if keys & _ROLE_SCHEMA_KEYS and keys & _ROLE_AUTHORITY_SCHEMA_KEYS:
        return True
    return any(
        later_line - earlier_line <= _MAX_SENSITIVE_ASSIGNMENT_LINE_GAP
        for (earlier_line, _), (later_line, _) in zip(assignments, assignments[1:])
    )


def _contains_known_schema_assignment(value: str) -> bool:
    quoted_keys = _QUOTED_ASSIGNMENT_KEY_RE.findall(value)
    if any(
        _canonical_schema_key(key) in _KNOWN_SENSITIVE_SCHEMA_KEYS
        for key in quoted_keys
    ):
        return True
    for fragment in _STRUCTURED_FRAGMENT_RE.findall(value):
        keys = _QUOTED_ASSIGNMENT_KEY_RE.findall(fragment)
        keys.extend(_BARE_ASSIGNMENT_KEY_RE.findall(fragment))
        if any(
            _canonical_schema_key(key) in _KNOWN_SENSITIVE_SCHEMA_KEYS for key in keys
        ):
            return True
    return _contains_unfenced_sensitive_assignment_cluster(value)


def sanitize_delivery_content(
    value: str, *, limit: int = 100_000
) -> SanitizedDeliveryContent:
    raw = str(value or "")
    canonical = _canonicalize_for_schema_detection(raw)
    if _looks_like_structured_payload(canonical) or _contains_known_schema_assignment(
        canonical
    ):
        raise LedgerSanitizationError("structured delivery payloads are forbidden")
    text = str(value or "")
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text, force=True)
    except Exception as exc:
        raise LedgerSanitizationError("central redactor unavailable") from exc
    if not isinstance(text, str):
        raise LedgerSanitizationError("central redactor returned invalid content")
    text = _ACCOUNT_LABEL_RE.sub(r"\1: [REDACTED]", text)
    text = _TOKEN_LABEL_RE.sub("token: [REDACTED]", text)
    text = _SENSITIVE_TEXT_LABEL_RE.sub(r"\1: [REDACTED]", text)
    text = _POSITION_VALUE_LABEL_RE.sub(r"\1: [REDACTED]", text)
    return SanitizedDeliveryContent(text=text[:limit])


def _sanitize_text(value: str, *, limit: int) -> str:
    return sanitize_delivery_content(value, limit=limit).text


def _trusted_parent_fd(path: Path) -> int:
    parent = path.parent
    try:
        before = os.stat(parent, follow_symlinks=False)
    except OSError as exc:
        raise LedgerTrustError("ledger parent must already exist") from exc
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_mode & 0o022
    ):
        raise LedgerTrustError("ledger parent is not a trusted private directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(parent, flags)
    except OSError as exc:
        raise LedgerTrustError("ledger parent could not be opened safely") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or opened.st_mode & 0o022
        or opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
    ):
        os.close(descriptor)
        raise LedgerTrustError("ledger parent changed during trust validation")
    return descriptor


def _trusted_regular(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and value.st_uid == os.geteuid()
        and value.st_nlink == 1
        and value.st_mode & 0o022 == 0
    )


def _open_trusted_entry(
    parent_fd: int,
    name: str,
    *,
    create: bool,
    required: bool,
) -> int | None:
    flags = (
        os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
    except FileExistsError:
        if not create:
            raise
        return _open_trusted_entry(parent_fd, name, create=False, required=required)
    except FileNotFoundError:
        if required:
            raise LedgerTrustError(f"required SQLite entry is missing: {name}")
        return None
    except OSError as exc:
        raise LedgerTrustError(f"unsafe SQLite entry rejected: {name}") from exc

    opened = os.fstat(descriptor)
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        os.close(descriptor)
        raise LedgerTrustError(f"SQLite entry changed during open: {name}") from exc
    if (
        not _trusted_regular(opened)
        or not _trusted_regular(current)
        or opened.st_dev != current.st_dev
        or opened.st_ino != current.st_ino
    ):
        os.close(descriptor)
        raise LedgerTrustError(f"SQLite entry is not a trusted regular file: {name}")
    return descriptor


def _entry_identity_matches(
    parent_fd: int,
    filename: str,
    opened: os.stat_result,
) -> bool:
    try:
        current = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    return (
        _trusted_regular(current)
        and current.st_dev == opened.st_dev
        and current.st_ino == opened.st_ino
    )


def _reject_legacy_sqlite_sidecars(parent_fd: int, filename: str) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        try:
            os.stat(f"{filename}{suffix}", dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise LedgerTrustError("legacy SQLite sidecar trust check failed") from exc
        raise LedgerTrustError(
            f"legacy SQLite sidecar is forbidden: {filename}{suffix}"
        )


def _clean_orphan_snapshots(parent_fd: int, filename: str) -> None:
    prefix = f".{filename}.tmp."
    removed = False
    try:
        names = os.listdir(parent_fd)
    except OSError as exc:
        raise LedgerTrustError("ledger parent could not be inspected safely") from exc
    for name in names:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if len(suffix) != 32 or any(
            character not in "0123456789abcdef" for character in suffix
        ):
            continue
        descriptor = _open_trusted_entry(
            parent_fd,
            name,
            create=False,
            required=True,
        )
        assert descriptor is not None
        try:
            opened = os.fstat(descriptor)
            if not _entry_identity_matches(parent_fd, name, opened):
                raise LedgerTrustError(
                    "orphaned ledger snapshot changed during cleanup"
                )
            os.unlink(name, dir_fd=parent_fd)
            removed = True
        finally:
            os.close(descriptor)
    if removed:
        os.fsync(parent_fd)


def _open_lock(parent_fd: int, filename: str) -> int:
    lock_name = f"{filename}.lock"
    descriptor = _open_trusted_entry(
        parent_fd,
        lock_name,
        create=True,
        required=True,
    )
    assert descriptor is not None
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        if not _entry_identity_matches(parent_fd, lock_name, opened):
            raise LedgerTrustError("ledger lock changed while acquiring it")
        os.fsync(parent_fd)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_database(descriptor: int) -> bytes:
    before = os.fstat(descriptor)
    if before.st_size > _MAX_DATABASE_BYTES:
        raise LedgerTrustError("ledger database exceeds the size limit")
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise LedgerTrustError("ledger database changed while reading")
        chunks.append(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    if (
        after.st_size != before.st_size
        or after.st_ctime_ns != before.st_ctime_ns
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise LedgerTrustError("ledger database changed while reading")
    return b"".join(chunks)


def _sqlite_snapshot_supported() -> bool:
    return callable(getattr(sqlite3.Connection, "serialize", None)) and callable(
        getattr(sqlite3.Connection, "deserialize", None)
    )


def _require_secure_delete(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA secure_delete=ON")
    enabled = conn.execute("PRAGMA secure_delete").fetchone()
    if not enabled or str(enabled[0]).lower() not in {"1", "on"}:
        raise LedgerTrustError("SQLite secure deletion is required")


def _memory_connection(image: bytes | None) -> sqlite3.Connection:
    if not _sqlite_snapshot_supported():
        raise LedgerTrustError("SQLite serialize/deserialize support is required")
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        _require_secure_delete(conn)
        if image is not None:
            conn.deserialize(image)
            _require_secure_delete(conn)
        conn.row_factory = sqlite3.Row
        journal_mode = conn.execute("PRAGMA journal_mode=MEMORY").fetchone()
        if not journal_mode or str(journal_mode[0]).lower() != "memory":
            raise LedgerTrustError("in-memory SQLite journal mode unavailable")
        conn.execute("PRAGMA temp_store=MEMORY")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise LedgerTrustError("ledger database integrity check failed")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS plugin_deliveries (
                delivery_id TEXT PRIMARY KEY,
                plugin_id TEXT NOT NULL,
                binding_digest TEXT NOT NULL,
                delivery_key_digest TEXT NOT NULL,
                recovery_context_digest TEXT,
                content_digest TEXT NOT NULL,
                sanitized_content TEXT,
                state TEXT NOT NULL,
                claim_digest TEXT,
                receipt_id TEXT,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                content_expires_at REAL NOT NULL,
                UNIQUE(plugin_id, binding_digest, delivery_key_digest)
            )"""
        )
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(plugin_deliveries)")
        }
        if "recovery_context_digest" not in columns:
            conn.execute(
                "ALTER TABLE plugin_deliveries ADD COLUMN recovery_context_digest TEXT"
            )
        return conn
    except LedgerTrustError:
        if conn is not None:
            conn.close()
        raise
    except (sqlite3.Error, MemoryError, ValueError) as exc:
        if conn is not None:
            conn.close()
        raise LedgerTrustError("SQLite image could not be loaded safely") from exc


@dataclass
class _OpenedLedger:
    conn: sqlite3.Connection
    parent_fd: int
    lock_fd: int
    database_fd: int | None
    database_stat: os.stat_result | None
    filename: str

    def close(self) -> None:
        try:
            self.conn.close()
        finally:
            if self.database_fd is not None:
                os.close(self.database_fd)
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self.lock_fd)
                os.close(self.parent_fd)


def _destination_matches_initial(opened: _OpenedLedger) -> bool:
    if opened.database_stat is None:
        try:
            os.stat(
                opened.filename,
                dir_fd=opened.parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return False
    return _entry_identity_matches(
        opened.parent_fd,
        opened.filename,
        opened.database_stat,
    )


def _serialize_database(conn: sqlite3.Connection) -> bytes:
    try:
        image = conn.serialize()
    except (sqlite3.Error, MemoryError) as exc:
        raise LedgerTrustError("SQLite image serialization failed") from exc
    if not image or len(image) > _MAX_DATABASE_BYTES:
        raise LedgerTrustError("serialized ledger database has an invalid size")
    return image


def _write_all(descriptor: int, image: bytes) -> None:
    offset = 0
    while offset < len(image):
        written = os.write(descriptor, image[offset:])
        if written <= 0:
            raise OSError("short write while publishing ledger snapshot")
        offset += written


def _publish_database(opened: _OpenedLedger, image: bytes) -> None:
    parent_fd = opened.parent_fd
    temp_name = f".{opened.filename}.tmp.{secrets.token_hex(16)}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    temp_fd: int | None = None
    renamed = False
    try:
        _reject_legacy_sqlite_sidecars(parent_fd, opened.filename)
        temp_fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        os.fchmod(temp_fd, 0o600)
        _write_all(temp_fd, image)
        os.fsync(temp_fd)
        temp_stat = os.fstat(temp_fd)
        if not _trusted_regular(temp_stat) or temp_stat.st_size != len(image):
            raise LedgerTrustError("published ledger snapshot is not trusted")
        if not _entry_identity_matches(parent_fd, temp_name, temp_stat):
            raise LedgerTrustError("ledger snapshot temp entry changed before publish")
        if not _destination_matches_initial(opened):
            raise LedgerTrustError("ledger destination changed before publish")
        _reject_legacy_sqlite_sidecars(parent_fd, opened.filename)
        os.replace(
            temp_name,
            opened.filename,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        renamed = True
        if not _entry_identity_matches(parent_fd, opened.filename, temp_stat):
            raise LedgerTrustError("ledger destination changed during publish")
        _reject_legacy_sqlite_sidecars(parent_fd, opened.filename)
        os.fsync(parent_fd)
        if not _entry_identity_matches(parent_fd, opened.filename, temp_stat):
            raise LedgerTrustError("ledger destination changed after publish")
        _reject_legacy_sqlite_sidecars(parent_fd, opened.filename)
    except Exception as exc:
        if renamed:
            if isinstance(exc, LedgerCommitUncertain):
                raise
            raise LedgerCommitUncertain(
                "atomic ledger snapshot was renamed but durability is uncertain"
            ) from exc
        if isinstance(exc, LedgerTrustError):
            raise
        raise LedgerTrustError("atomic ledger snapshot publication failed") from exc
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if not renamed:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def _open_ledger() -> _OpenedLedger:
    path = _db_path()
    parent_fd = _trusted_parent_fd(path)
    lock_fd: int | None = None
    database_fd: int | None = None
    conn: sqlite3.Connection | None = None
    try:
        lock_fd = _open_lock(parent_fd, path.name)
        _clean_orphan_snapshots(parent_fd, path.name)
        _reject_legacy_sqlite_sidecars(parent_fd, path.name)
        database_fd = _open_trusted_entry(
            parent_fd,
            path.name,
            create=False,
            required=False,
        )
        database_stat = None if database_fd is None else os.fstat(database_fd)
        image = None if database_fd is None else _read_database(database_fd)
        if database_stat is not None and not _entry_identity_matches(
            parent_fd, path.name, database_stat
        ):
            raise LedgerTrustError("ledger database changed while opening")
        conn = _memory_connection(image)
        return _OpenedLedger(
            conn=conn,
            parent_fd=parent_fd,
            lock_fd=lock_fd,
            database_fd=database_fd,
            database_stat=database_stat,
            filename=path.name,
        )
    except Exception:
        if conn is not None:
            conn.close()
        if database_fd is not None:
            os.close(database_fd)
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(parent_fd)
        raise


@contextmanager
def _transaction(*, immediate: bool = False) -> Iterator[sqlite3.Connection]:
    opened = _open_ledger()
    conn = opened.conn
    try:
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        yield conn
        if immediate:
            conn.commit()
            _publish_database(opened, _serialize_database(conn))
        else:
            conn.rollback()
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        opened.close()


def _row_to_record(row: sqlite3.Row) -> DeliveryRecord:
    return DeliveryRecord(
        delivery_id=row["delivery_id"],
        plugin_id=row["plugin_id"],
        binding_digest=row["binding_digest"],
        delivery_key_digest=row["delivery_key_digest"],
        recovery_context_digest=row["recovery_context_digest"],
        state=DeliveryState(row["state"]),
        sanitized_content=row["sanitized_content"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        receipt_id=row["receipt_id"],
        last_error=row["last_error"],
    )


def reserve_delivery(
    *,
    plugin_id: str,
    binding_handle: str,
    delivery_key: str,
    sanitized_content: str,
    recovery_context_digest: str | None = None,
    now: float | None = None,
) -> DeliveryRecord:
    """Persist one logical final result without retaining capability material."""
    if not _PLUGIN_ID_RE.fullmatch(plugin_id):
        raise ValueError("invalid plugin_id")
    if not binding_handle or not delivery_key:
        raise ValueError("binding_handle and delivery_key are required")
    if recovery_context_digest is not None and (
        not isinstance(recovery_context_digest, str)
        or not _SHA256_DIGEST_RE.fullmatch(recovery_context_digest)
    ):
        raise ValueError("invalid recovery_context_digest")
    content = _sanitize_text(sanitized_content, limit=100_000)
    if not content.strip():
        raise ValueError("sanitized_content must not be empty")
    timestamp = time.time() if now is None else float(now)
    binding_digest = _digest(binding_handle)
    key_digest = _digest(delivery_key)
    content_digest = _digest(content)
    delivery_id = _digest(f"{plugin_id}\0{binding_digest}\0{key_digest}")[:32]

    with _LOCK, _transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO plugin_deliveries (
                   delivery_id, plugin_id, binding_digest, delivery_key_digest,
                   recovery_context_digest, content_digest, sanitized_content,
                   state, created_at, updated_at, content_expires_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(plugin_id, binding_digest, delivery_key_digest)
               DO NOTHING""",
            (
                delivery_id,
                plugin_id,
                binding_digest,
                key_digest,
                recovery_context_digest,
                content_digest,
                content,
                DeliveryState.PENDING.value,
                timestamp,
                timestamp,
                timestamp + CONTENT_RETENTION_SECONDS,
            ),
        )
        row = conn.execute(
            """SELECT * FROM plugin_deliveries
               WHERE plugin_id=? AND binding_digest=? AND delivery_key_digest=?""",
            (plugin_id, binding_digest, key_digest),
        ).fetchone()
        if row is None:  # pragma: no cover - SQLite invariant
            raise RuntimeError("delivery reservation disappeared")
        if row["content_digest"] != content_digest:
            raise DeliveryConflictError(
                "logical delivery already reserved with different content"
            )
        if row["recovery_context_digest"] != recovery_context_digest:
            raise DeliveryConflictError(
                "logical delivery already reserved with different recovery context"
            )
        return _row_to_record(row)


def claim_for_send(
    *,
    plugin_id: str,
    binding_handle: str,
    delivery_key: str,
    now: float | None = None,
) -> DeliveryClaim | None:
    """Atomically claim a pending delivery; duplicate callers receive ``None``."""
    timestamp = time.time() if now is None else float(now)
    claim_token = secrets.token_hex(24)
    binding_digest = _digest(binding_handle)
    key_digest = _digest(delivery_key)
    with _LOCK, _transaction(immediate=True) as conn:
        row = conn.execute(
            """SELECT delivery_id, sanitized_content, content_expires_at
               FROM plugin_deliveries
               WHERE plugin_id=? AND binding_digest=? AND delivery_key_digest=?
                 AND state=?""",
            (plugin_id, binding_digest, key_digest, DeliveryState.PENDING.value),
        ).fetchone()
        if row is None:
            return None
        if row["sanitized_content"] is None or row["content_expires_at"] <= timestamp:
            conn.execute(
                """UPDATE plugin_deliveries
                   SET state=?, sanitized_content=NULL, updated_at=?,
                       last_error='content_retention_expired'
                   WHERE delivery_id=? AND state=?""",
                (
                    DeliveryState.FAILED.value,
                    timestamp,
                    row["delivery_id"],
                    DeliveryState.PENDING.value,
                ),
            )
            return None
        changed = conn.execute(
            """UPDATE plugin_deliveries
               SET state=?, claim_digest=?, sanitized_content=NULL, updated_at=?
               WHERE delivery_id=? AND state=?""",
            (
                DeliveryState.SEND_CLAIMED.value,
                _digest(claim_token),
                timestamp,
                row["delivery_id"],
                DeliveryState.PENDING.value,
            ),
        ).rowcount
        if changed != 1:
            return None
        return DeliveryClaim(
            delivery_id=row["delivery_id"],
            claim_token=claim_token,
            sanitized_content=row["sanitized_content"],
        )


def claim_pending_delivery(
    *,
    delivery_id: str,
    plugin_id: str,
    binding_digest: str,
    recovery_context_digest: str,
    now: float | None = None,
) -> DeliveryClaim | None:
    """Claim recovered pending work without reconstructing opaque identifiers."""
    if not isinstance(delivery_id, str) or not _DELIVERY_ID_RE.fullmatch(delivery_id):
        raise ValueError("invalid delivery_id")
    if not isinstance(plugin_id, str) or not _PLUGIN_ID_RE.fullmatch(plugin_id):
        raise ValueError("invalid plugin_id")
    if not isinstance(binding_digest, str) or not _SHA256_DIGEST_RE.fullmatch(
        binding_digest
    ):
        raise ValueError("invalid binding_digest")
    if not isinstance(recovery_context_digest, str) or not _SHA256_DIGEST_RE.fullmatch(
        recovery_context_digest
    ):
        raise ValueError("invalid recovery_context_digest")

    timestamp = time.time() if now is None else float(now)
    with _LOCK, _transaction(immediate=True) as conn:
        row = conn.execute(
            """SELECT delivery_id, sanitized_content, content_expires_at
               FROM plugin_deliveries
               WHERE delivery_id=? AND plugin_id=? AND binding_digest=?
                 AND recovery_context_digest=?
                 AND state=?""",
            (
                delivery_id,
                plugin_id,
                binding_digest,
                recovery_context_digest,
                DeliveryState.PENDING.value,
            ),
        ).fetchone()
        if row is None:
            return None
        if row["sanitized_content"] is None or row["content_expires_at"] <= timestamp:
            conn.execute(
                """UPDATE plugin_deliveries
                   SET state=?, sanitized_content=NULL, updated_at=?,
                       last_error='content_retention_expired'
                   WHERE delivery_id=? AND plugin_id=? AND binding_digest=?
                     AND recovery_context_digest=?
                     AND state=?""",
                (
                    DeliveryState.FAILED.value,
                    timestamp,
                    delivery_id,
                    plugin_id,
                    binding_digest,
                    recovery_context_digest,
                    DeliveryState.PENDING.value,
                ),
            )
            return None

        claim_token = secrets.token_hex(24)
        changed = conn.execute(
            """UPDATE plugin_deliveries
               SET state=?, claim_digest=?, sanitized_content=NULL, updated_at=?
               WHERE delivery_id=? AND plugin_id=? AND binding_digest=?
                 AND recovery_context_digest=?
                 AND state=?""",
            (
                DeliveryState.SEND_CLAIMED.value,
                _digest(claim_token),
                timestamp,
                delivery_id,
                plugin_id,
                binding_digest,
                recovery_context_digest,
                DeliveryState.PENDING.value,
            ),
        ).rowcount
        if changed != 1:
            return None
        return DeliveryClaim(
            delivery_id=row["delivery_id"],
            claim_token=claim_token,
            sanitized_content=row["sanitized_content"],
        )


def cancel_pending_delivery(
    *,
    delivery_id: str,
    plugin_id: str,
    binding_digest: str,
    recovery_context_digest: str,
    reason: str = "cancelled",
    now: float | None = None,
) -> bool:
    """Durably cancel exact pending recovery work before transport dispatch."""
    if not isinstance(delivery_id, str) or not _DELIVERY_ID_RE.fullmatch(delivery_id):
        raise ValueError("invalid delivery_id")
    if not isinstance(plugin_id, str) or not _PLUGIN_ID_RE.fullmatch(plugin_id):
        raise ValueError("invalid plugin_id")
    if not isinstance(binding_digest, str) or not _SHA256_DIGEST_RE.fullmatch(
        binding_digest
    ):
        raise ValueError("invalid binding_digest")
    if not isinstance(recovery_context_digest, str) or not _SHA256_DIGEST_RE.fullmatch(
        recovery_context_digest
    ):
        raise ValueError("invalid recovery_context_digest")

    safe_reason = _sanitize_text(reason, limit=500).strip() or "cancelled"
    timestamp = time.time() if now is None else float(now)
    with _LOCK, _transaction(immediate=True) as conn:
        changed = conn.execute(
            """UPDATE plugin_deliveries
               SET state=?, claim_digest=NULL, sanitized_content=NULL,
                   updated_at=?, last_error=?
               WHERE delivery_id=? AND plugin_id=? AND binding_digest=?
                 AND recovery_context_digest=?
                 AND state=?""",
            (
                DeliveryState.FAILED.value,
                timestamp,
                safe_reason,
                delivery_id,
                plugin_id,
                binding_digest,
                recovery_context_digest,
                DeliveryState.PENDING.value,
            ),
        ).rowcount
        return changed == 1


def _finish_claim(
    delivery_id: str,
    claim_token: str,
    state: DeliveryState,
    *,
    receipt_id: str | None = None,
    error: str | None = None,
    now: float | None = None,
) -> bool:
    if state not in {
        DeliveryState.DELIVERED,
        DeliveryState.FAILED,
        DeliveryState.DELIVERY_UNCERTAIN,
    }:
        raise ValueError("invalid terminal state")
    timestamp = time.time() if now is None else float(now)
    safe_receipt = _sanitize_text(receipt_id or "", limit=256) or None
    safe_error = _sanitize_text(error or "", limit=500) or None
    with _LOCK, _transaction(immediate=True) as conn:
        changed = conn.execute(
            """UPDATE plugin_deliveries
               SET state=?, claim_digest=NULL, receipt_id=?, last_error=?, updated_at=?
               WHERE delivery_id=? AND state=? AND claim_digest=?""",
            (
                state.value,
                safe_receipt,
                safe_error,
                timestamp,
                delivery_id,
                DeliveryState.SEND_CLAIMED.value,
                _digest(claim_token),
            ),
        ).rowcount
        return changed == 1


def mark_delivered(
    delivery_id: str,
    claim_token: str,
    *,
    receipt_id: str | None = None,
    now: float | None = None,
) -> bool:
    return _finish_claim(
        delivery_id,
        claim_token,
        DeliveryState.DELIVERED,
        receipt_id=receipt_id,
        now=now,
    )


def mark_failed(
    delivery_id: str,
    claim_token: str,
    *,
    error: str | None = None,
    now: float | None = None,
) -> bool:
    return _finish_claim(
        delivery_id,
        claim_token,
        DeliveryState.FAILED,
        error=error,
        now=now,
    )


def mark_uncertain(
    delivery_id: str,
    claim_token: str,
    *,
    error: str | None = None,
    now: float | None = None,
) -> bool:
    return _finish_claim(
        delivery_id,
        claim_token,
        DeliveryState.DELIVERY_UNCERTAIN,
        error=error,
        now=now,
    )


def recover_after_restart(*, now: float | None = None) -> list[PendingDelivery]:
    """Return only never-attempted work and quarantine ambiguous claims."""
    timestamp = time.time() if now is None else float(now)
    with _LOCK, _transaction(immediate=True) as conn:
        conn.execute(
            """UPDATE plugin_deliveries
               SET state=?, claim_digest=NULL, updated_at=?,
                   last_error='restart_after_send_claim'
               WHERE state=?""",
            (
                DeliveryState.DELIVERY_UNCERTAIN.value,
                timestamp,
                DeliveryState.SEND_CLAIMED.value,
            ),
        )
        conn.execute(
            """UPDATE plugin_deliveries
               SET state=?, sanitized_content=NULL, updated_at=?,
                   last_error='content_retention_expired'
               WHERE state=? AND content_expires_at<=?""",
            (
                DeliveryState.FAILED.value,
                timestamp,
                DeliveryState.PENDING.value,
                timestamp,
            ),
        )
        conn.execute(
            """UPDATE plugin_deliveries
               SET state=?, sanitized_content=NULL, updated_at=?,
                   last_error='recovery_context_missing'
               WHERE state=? AND recovery_context_digest IS NULL""",
            (
                DeliveryState.FAILED.value,
                timestamp,
                DeliveryState.PENDING.value,
            ),
        )
        rows = conn.execute(
            """SELECT delivery_id, sanitized_content, binding_digest,
                      recovery_context_digest
               FROM plugin_deliveries
               WHERE state=? AND sanitized_content IS NOT NULL
                 AND recovery_context_digest IS NOT NULL
               ORDER BY created_at, delivery_id""",
            (DeliveryState.PENDING.value,),
        ).fetchall()
        return [
            PendingDelivery(
                delivery_id=row["delivery_id"],
                content=row["sanitized_content"],
                binding_digest=row["binding_digest"],
                recovery_context_digest=row["recovery_context_digest"],
            )
            for row in rows
        ]


def get_delivery(delivery_id: str) -> DeliveryRecord | None:
    with _LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM plugin_deliveries WHERE delivery_id=?", (delivery_id,)
        ).fetchone()
        return None if row is None else _row_to_record(row)


def prune(*, now: float | None = None) -> None:
    timestamp = time.time() if now is None else float(now)
    with _LOCK, _transaction(immediate=True) as conn:
        conn.execute(
            """UPDATE plugin_deliveries
               SET sanitized_content=NULL, updated_at=CASE
                   WHEN state=? THEN ? ELSE updated_at END,
                   state=CASE WHEN state=? THEN ? ELSE state END,
                   last_error=CASE WHEN state=? THEN 'content_retention_expired'
                                   ELSE last_error END
               WHERE content_expires_at<=? AND sanitized_content IS NOT NULL""",
            (
                DeliveryState.PENDING.value,
                timestamp,
                DeliveryState.PENDING.value,
                DeliveryState.FAILED.value,
                DeliveryState.PENDING.value,
                timestamp,
            ),
        )
        conn.execute(
            "DELETE FROM plugin_deliveries WHERE created_at<=?",
            (timestamp - METADATA_RETENTION_SECONDS,),
        )
