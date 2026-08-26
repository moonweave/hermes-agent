from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sqlite3
import stat
import threading
from pathlib import Path

import pytest

from gateway import plugin_delivery_ledger as ledger


def _recovery_context_digest(
    *,
    session: str = "session-1",
    chat: str = "chat-1",
    topic: str = "topic-1",
    user: str = "user-1",
    profile: str = "profile-1",
) -> str:
    canonical = "\0".join((session, chat, topic, user, profile))
    return hashlib.sha256(canonical.encode()).hexdigest()


_DEFAULT_RECOVERY_CONTEXT_DIGEST = _recovery_context_digest()


def _required_record_context(record: ledger.DeliveryRecord) -> str:
    assert record.recovery_context_digest is not None
    return record.recovery_context_digest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    path = tmp_path / "profile" / "plugin-delivery.sqlite3"
    path.parent.mkdir(mode=0o700)
    monkeypatch.setattr(ledger, "_db_path", lambda: path)
    return path


def _reserve(
    *,
    content: str = "sanitized final",
    binding: str = "opaque.secret.binding",
    recovery_context_digest: str | None = _DEFAULT_RECOVERY_CONTEXT_DIGEST,
    reconciliation_id: str | None = None,
):
    kwargs = {}
    if reconciliation_id is not None:
        kwargs["reconciliation_id"] = reconciliation_id
    return ledger.reserve_delivery(
        plugin_id="test.plugin",
        binding_handle=binding,
        delivery_key="turn-1",
        sanitized_content=content,
        recovery_context_digest=recovery_context_digest,
        now=1_000.0,
        **kwargs,
    )


def _claim_in_process(start, results):
    start.wait()
    claim = ledger.claim_for_send(
        plugin_id="test.plugin",
        binding_handle="opaque.secret.binding",
        delivery_key="turn-1",
        now=1_001.0,
    )
    results.put(claim is not None)


def _recovery_claim_in_process(
    start, results, delivery_id, binding_digest, recovery_context_digest
):
    start.wait()
    claim = ledger.claim_pending_delivery(
        delivery_id=delivery_id,
        plugin_id="test.plugin",
        binding_digest=binding_digest,
        recovery_context_digest=recovery_context_digest,
        now=1_002.0,
    )
    results.put(claim is not None)


def _cancel_pending_in_process(
    start, results, delivery_id, binding_digest, recovery_context_digest
):
    start.wait()
    cancelled = ledger.cancel_pending_delivery(
        delivery_id=delivery_id,
        plugin_id="test.plugin",
        binding_digest=binding_digest,
        recovery_context_digest=recovery_context_digest,
        reason="process_cancelled",
        now=1_002.0,
    )
    results.put(cancelled)


def _page_spanning_content(label: str) -> str:
    return (f"{label}|" * 400)[:10_000]


def _force_secure_delete_default_off(monkeypatch) -> None:
    real_connect = sqlite3.connect

    class InsecureDefaultConnection:
        def __init__(self, conn):
            object.__setattr__(self, "_conn", conn)
            conn.execute("PRAGMA secure_delete=OFF")

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def __setattr__(self, name, value):
            setattr(self._conn, name, value)

        def deserialize(self, image):
            self._conn.deserialize(image)
            self._conn.execute("PRAGMA secure_delete=OFF")

    def insecure_default_connect(*args, **kwargs):
        return InsecureDefaultConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(ledger.sqlite3, "connect", insecure_default_connect)


def test_database_is_private_and_does_not_store_opaque_handles_or_keys(_isolated_db):
    record = _reserve()

    assert record.state is ledger.DeliveryState.PENDING
    assert os.stat(_isolated_db).st_mode & 0o777 == 0o600
    raw = _isolated_db.read_bytes()
    assert b"opaque.secret.binding" not in raw
    assert b"turn-1" not in raw


def test_database_is_atomic_sqlite_image_without_sqlite_sidecars(_isolated_db):
    _reserve()

    with sqlite3.connect(_isolated_db) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT count(*) FROM plugin_deliveries").fetchone() == (1,)
    assert not Path(f"{_isolated_db}-wal").exists()
    assert not Path(f"{_isolated_db}-shm").exists()
    assert not Path(f"{_isolated_db}-journal").exists()


def test_recovery_claim_securely_erases_multirow_content_from_database_image(
    _isolated_db, monkeypatch
):
    _force_secure_delete_default_off(monkeypatch)
    needle = b"recovery-claim-sensitive-Z"
    erased = _page_spanning_content(needle.decode())
    record = _reserve(content=erased, binding="erase-binding")
    _reserve(content=_page_spanning_content("retained-row-Y"), binding="retain-binding")
    assert needle in _isolated_db.read_bytes()
    ledger.recover_after_restart(now=1_001.0)

    claim = ledger.claim_pending_delivery(
        delivery_id=record.delivery_id,
        plugin_id=record.plugin_id,
        binding_digest=record.binding_digest,
        recovery_context_digest=_required_record_context(record),
        now=1_002.0,
    )

    assert claim is not None
    assert needle not in _isolated_db.read_bytes()


def test_restart_expiry_securely_erases_multirow_content_from_database_image(
    _isolated_db, monkeypatch
):
    _force_secure_delete_default_off(monkeypatch)
    needle = b"restart-expiry-sensitive-X"
    erased = _page_spanning_content(needle.decode())
    _reserve(content=erased, binding="expiring-binding")
    _reserve(
        content=_page_spanning_content("other-expiring-row-W"), binding="other-binding"
    )
    assert needle in _isolated_db.read_bytes()

    assert (
        ledger.recover_after_restart(now=1_000.0 + ledger.CONTENT_RETENTION_SECONDS + 1)
        == []
    )

    assert needle not in _isolated_db.read_bytes()


def test_recovery_claim_expiry_securely_erases_content_from_database_image(
    _isolated_db, monkeypatch
):
    _force_secure_delete_default_off(monkeypatch)
    needle = b"claim-expiry-sensitive-V"
    erased = _page_spanning_content(needle.decode())
    record = _reserve(content=erased, binding="claim-expiry-binding")
    _reserve(content=_page_spanning_content("retained-row-U"), binding="retain-binding")
    assert needle in _isolated_db.read_bytes()

    assert (
        ledger.claim_pending_delivery(
            delivery_id=record.delivery_id,
            plugin_id=record.plugin_id,
            binding_digest=record.binding_digest,
            recovery_context_digest=_required_record_context(record),
            now=1_000.0 + ledger.CONTENT_RETENTION_SECONDS + 1,
        )
        is None
    )

    assert needle not in _isolated_db.read_bytes()


def test_secure_delete_pragma_failure_is_fail_closed(_isolated_db, monkeypatch):
    real_connect = sqlite3.connect

    class SecureDeleteDisabledConnection:
        def __init__(self, conn):
            object.__setattr__(self, "_conn", conn)

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def __setattr__(self, name, value):
            setattr(self._conn, name, value)

        def execute(self, statement, *args, **kwargs):
            cursor = self._conn.execute(statement, *args, **kwargs)
            if statement.strip().lower() == "pragma secure_delete":
                return type(
                    "DisabledPragmaResult",
                    (),
                    {"fetchone": staticmethod(lambda: (0,))},
                )()
            return cursor

    monkeypatch.setattr(
        ledger.sqlite3,
        "connect",
        lambda *args, **kwargs: SecureDeleteDisabledConnection(
            real_connect(*args, **kwargs)
        ),
    )

    with pytest.raises(ledger.LedgerTrustError, match="secure deletion"):
        _reserve()

    assert not _isolated_db.exists()


def test_new_sqlite_entries_fsync_the_trusted_parent_directory(monkeypatch):
    real_fsync = os.fsync
    fsynced_modes: list[int] = []

    def recording_fsync(descriptor):
        fsynced_modes.append(os.fstat(descriptor).st_mode)
        return real_fsync(descriptor)

    monkeypatch.setattr(ledger.os, "fsync", recording_fsync)

    _reserve()

    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_restart_removes_trusted_orphan_snapshot_before_new_write(_isolated_db):
    orphan = _isolated_db.parent / f".{_isolated_db.name}.tmp.{'d' * 32}"
    orphan.write_bytes(b"orphaned-sensitive-final")
    orphan.chmod(0o600)

    _reserve()

    assert not orphan.exists()
    assert b"orphaned-sensitive-final" not in _isolated_db.read_bytes()


@pytest.mark.parametrize(
    "suffix",
    ["my-important-notes", "A" * 32, "abc123"],
)
def test_noncanonical_temp_like_file_survives_ledger_open_unchanged(
    _isolated_db, suffix
):
    unrelated = _isolated_db.parent / f".{_isolated_db.name}.tmp.{suffix}"
    unrelated.write_bytes(b"important-unrelated-content")
    unrelated.chmod(0o600)

    _reserve()

    assert unrelated.read_bytes() == b"important-unrelated-content"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_unsafe_orphan_snapshot_is_rejected_without_mutating_victim(
    _isolated_db, tmp_path, kind
):
    orphan = _isolated_db.parent / f".{_isolated_db.name}.tmp.{'a' * 32}"
    victim = tmp_path / "orphan-victim"
    victim.write_bytes(b"orphan-do-not-touch")
    victim.chmod(0o640)
    if kind == "symlink":
        orphan.symlink_to(victim)
    elif kind == "hardlink":
        os.link(victim, orphan)
    else:
        os.mkfifo(orphan, 0o600)

    with pytest.raises(ledger.LedgerTrustError):
        _reserve()

    assert victim.read_bytes() == b"orphan-do-not-touch"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o640


def test_defense_in_depth_redacts_labeled_account_and_token_content(_isolated_db):
    record = _reserve(content="account_number: 1234-5678-9012 token=super-secret-token")

    assert record.sanitized_content is not None
    assert "1234-5678-9012" not in record.sanitized_content
    assert "super-secret-token" not in record.sanitized_content
    raw = _isolated_db.read_bytes()
    assert b"1234-5678-9012" not in raw
    assert b"super-secret-token" not in raw


@pytest.mark.parametrize(
    "sensitive",
    [
        "보유수량: 10",
        "quantity=10",
        "가격: 70000",
        "price=70000",
        "PnL=-5%",
        "평가손익=-1000",
        "현재가: 71,000",
        "매입가=69,000",
        "평균단가: 68,500",
        "평가금액=7,100,000",
        "손익률: +5.2%",
        "평가손익률=-3.1%",
        "profit_loss=-1000",
        "수량: 10",
        "실현손익=12,000",
        "HMAC=deadbeef",
        "가격:70,000",
    ],
)
def test_labeled_position_price_and_pnl_are_not_retained(_isolated_db, sensitive):
    record = _reserve(content=f"advisory result; {sensitive}; manual MTS")

    assert record.sanitized_content is not None
    assert sensitive not in record.sanitized_content
    assert "[REDACTED]" in record.sanitized_content
    assert sensitive.encode("utf-8") not in _isolated_db.read_bytes()


def test_comma_formatted_price_leaves_no_numeric_suffix(_isolated_db):
    record = _reserve(content="가격:70,000; manual MTS")

    assert record.sanitized_content == "가격: [REDACTED]; manual MTS"
    assert ",000" not in record.sanitized_content
    assert b",000" not in _isolated_db.read_bytes()


def test_central_redactor_failure_rejects_before_database_write(
    _isolated_db, monkeypatch
):
    import agent.redact

    def fail_redaction(*_args, **_kwargs):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr(agent.redact, "redact_sensitive_text", fail_redaction)

    with pytest.raises(ledger.LedgerSanitizationError):
        _reserve(content="final answer")

    assert not _isolated_db.exists()


def test_central_redactor_invalid_result_rejects_before_database_write(
    _isolated_db, monkeypatch
):
    import agent.redact

    monkeypatch.setattr(agent.redact, "redact_sensitive_text", lambda *_a, **_k: None)

    with pytest.raises(ledger.LedgerSanitizationError):
        _reserve(content="final answer")

    assert not _isolated_db.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "Risk", "opinion": "hold", "quantity": 10},
        {"result": {"account": "1234", "current_price": 70_000}},
        [{"role": "Red Team", "profit_loss": -1_000, "hmac": "secret"}],
    ],
)
def test_structured_or_raw_role_payload_is_rejected_before_database_write(
    _isolated_db, payload
):
    with pytest.raises(ledger.LedgerSanitizationError):
        _reserve(content=json.dumps(payload, ensure_ascii=False))

    assert not _isolated_db.exists()


def test_python_literal_raw_role_payload_is_rejected_before_database_write(
    _isolated_db,
):
    payload = {
        "role": "Risk",
        "opinion": {"account": "1234", "quantity": 10, "hmac": "secret"},
    }

    with pytest.raises(ledger.LedgerSanitizationError):
        _reserve(content=repr(payload))

    assert not _isolated_db.exists()


@pytest.mark.parametrize(
    "content, forbidden",
    [
        (
            '결론은 관찰입니다. ```json\n{"role":"Risk","opinion":"hold",'
            '"evidence":["x"],"quantity":10,"hmac":"deadbeef"}\n```',
            [b'"opinion":"hold"', b"deadbeef"],
        ),
        (
            "요약: {'full_role_output': {'prompt': 'raw', 'account': '1234', "
            "'current_price': 70000, 'profit_loss': -1000}} 끝",
            [b"full_role_output", b"current_price", b"profit_loss"],
        ),
        (
            '일반 문장 안의 조각 "holding_quantity": 10 과 "PnL": -5%',
            [b"holding_quantity", b'"PnL"'],
        ),
        (
            '검토 결과 {"purchase_price":69000} 이상입니다.',
            [b"purchase_price", b"69000"],
        ),
        (
            '요약 {"평단":68500,"보유량":10,"수익률":5.2} 끝',
            ["평단".encode(), "보유량".encode(), "수익률".encode()],
        ),
        (
            "팀 결과:\n```yaml\nrole: Risk\nopinion: hold\nevidence: stale\n```",
            [b"role: Risk", b"opinion: hold", b"evidence: stale"],
        ),
        (
            r"요약 {\"role\":\"Risk\",\"quantity\":10,\"hmac\":\"secret\"}",
            [b"role", b"quantity", b"hmac", b"secret"],
        ),
        (
            '결론 {"average_price":68500} 이상',
            [b"average_price", b"68500"],
        ),
        (
            r"요약 {\\\"role\\\":\\\"Risk\\\",\\\"opinion\\\":\\\"hold\\\"}",
            [b"role", b"opinion", b"Risk", b"hold"],
        ),
        (
            r'요약 {"\u0072ole":"Risk","\u006fpinion":"hold"}',
            [b"u0072ole", b"u006fpinion", b"Risk", b"hold"],
        ),
        (
            '팀 결과:\n```toml\nrole = "Risk"\nopinion = "hold"\n```',
            [b'role = "Risk"', b'opinion = "hold"'],
        ),
        (
            "팀 결과:\n```ini\nevidence = stale\n```",
            [b"evidence = stale"],
        ),
    ],
)
def test_embedded_structured_role_payload_is_rejected_without_raw_db_retention(
    _isolated_db, content, forbidden
):
    _reserve()

    with pytest.raises(ledger.LedgerSanitizationError):
        _reserve(binding="embedded-payload", content=content)

    raw = _isolated_db.read_bytes()
    for value in forbidden:
        assert value not in raw


@pytest.mark.parametrize(
    "content, forbidden",
    [
        (
            "팀 결과:\nrole: Risk\nopinion: SELL ALL\nevidence: raw private rationale",
            [b"SELL ALL", b"raw private rationale"],
        ),
        (
            '팀 결과:\nrole = "Risk"\nopinion = "SELL ALL"\nevidence = "raw private rationale"',
            [b"SELL ALL", b"raw private rationale"],
        ),
        (
            "팀 결과:\n  role: Risk\n\n  opinion: SELL ALL\n  evidence: raw private rationale",
            [b"SELL ALL", b"raw private rationale"],
        ),
    ],
)
def test_unfenced_sensitive_assignment_cluster_is_rejected_without_raw_retention(
    _isolated_db, content, forbidden
):
    _reserve()

    with pytest.raises(ledger.LedgerSanitizationError):
        _reserve(binding="unfenced-cluster", content=content)

    raw = _isolated_db.read_bytes()
    for value in forbidden:
        assert value not in raw


@pytest.mark.parametrize(
    "content",
    [
        "일반 결론은 관찰입니다. Risk 역할 의견을 참고하세요.",
        "예시:\n```text\nordinary explanation without schema assignments\n```",
        "코드:\n```python\nresult = calculate_market_regime()\n```",
        "의견: 관찰\nfreshness: current\n사용자 행동은 manual MTS 확인입니다.",
        "role: Risk\n이 문장은 단일 역할 label을 설명하는 일반 prose입니다.",
    ],
)
def test_safe_prose_and_safe_fences_remain_deliverable(content):
    record = _reserve(content=content)

    assert record.sanitized_content == content


def test_schema_detection_fails_closed_on_excessive_escape_nesting(_isolated_db):
    content = '{"role":"Risk"}'
    for _ in range(ledger._MAX_CANONICALIZATION_PASSES + 1):
        content = content.replace("\\", "\\\\").replace('"', r"\"")

    with pytest.raises(ledger.LedgerSanitizationError):
        _reserve(content=content)

    assert not _isolated_db.exists()


def test_schema_detection_fails_closed_above_bounded_input(_isolated_db):
    content = "x" * (ledger._MAX_CANONICAL_INPUT_BYTES + 1)

    with pytest.raises(ledger.LedgerSanitizationError):
        _reserve(content=content)

    assert not _isolated_db.exists()


@pytest.mark.parametrize(
    "key",
    [
        "average_price",
        "avg-price",
        "purchase_price",
        "acquisition.price",
        "entry_price",
        "cost basis",
        "current_price",
        "evaluation_price",
        "holding_quantity",
        "quantity",
        "profit_loss",
        "account_number",
        "hmac",
        "prompt",
        "full-role-output",
        "role",
        "opinion",
        "evidence",
        "official_verdict",
        "official_permission",
        "official_trigger",
        "official_max_weight",
        "평단",
        "보유량",
        "수익률",
        "계좌번호",
    ],
)
def test_known_sensitive_schema_alias_assignment_is_rejected(key):
    content = f'요약 {{"{key}": "raw-value"}} 끝'

    with pytest.raises(ledger.LedgerSanitizationError):
        ledger.sanitize_delivery_content(content)


@pytest.mark.parametrize(
    "sensitive, forbidden",
    [
        ("account number = 1234 5678 9012", ["1234", "5678", "9012"]),
        ("계좌 번호 : 123-456 789", ["123-456", "789"]),
        ("current_price: 70,000", ["70,000"]),
        ("purchase_price = 69,000", ["69,000"]),
        ("holding quantity = 10", ["10"]),
        ("profit loss: -1,000", ["-1,000"]),
        ("prompt: raw secret phrase", ["raw secret phrase"]),
        ("full_role_output = complete private opinion", ["complete private opinion"]),
        ("HMAC: abc def ghi", ["abc def ghi"]),
    ],
)
def test_plain_final_text_redacts_spaced_sensitive_values_fail_closed_locally(
    _isolated_db, monkeypatch, sensitive, forbidden
):
    import agent.redact

    monkeypatch.setattr(
        agent.redact, "redact_sensitive_text", lambda value, **_kwargs: value
    )

    record = _reserve(content=f"결론; {sensitive}; manual MTS")

    assert record.sanitized_content is not None
    assert "[REDACTED]" in record.sanitized_content
    raw = _isolated_db.read_bytes()
    for value in forbidden:
        assert value not in record.sanitized_content
        assert value.encode() not in raw


@pytest.mark.parametrize(
    "sensitive, forbidden",
    [
        ("보유량 : 1,000", ["1,000"]),
        ("평단 = 70, 000", ["70, 000"]),
        ("매수가: 68,500", ["68,500"]),
        ("평가액 = 7,100,000 원", ["7,100,000"]),
        ("수익률 : -5.2 %", ["-5.2"]),
        ("총손익 = -1, 000", ["-1, 000"]),
    ],
)
def test_korean_position_aliases_are_fully_redacted_from_sqlite_image(
    _isolated_db, monkeypatch, sensitive, forbidden
):
    import agent.redact

    monkeypatch.setattr(
        agent.redact, "redact_sensitive_text", lambda value, **_kwargs: value
    )

    record = _reserve(content=f"결론; {sensitive}; manual MTS")

    assert record.sanitized_content is not None
    assert "[REDACTED]" in record.sanitized_content
    raw = _isolated_db.read_bytes()
    for value in forbidden:
        assert value not in record.sanitized_content
        assert value.encode() not in raw


def test_reservation_is_idempotent_but_content_is_generated_once():
    first = _reserve()
    second = _reserve()

    assert second.delivery_id == first.delivery_id
    with pytest.raises(ledger.DeliveryConflictError):
        _reserve(content="a different final")


def test_atomic_claim_allows_exactly_one_racing_sender():
    _reserve()

    def claim():
        return ledger.claim_for_send(
            plugin_id="test.plugin",
            binding_handle="opaque.secret.binding",
            delivery_key="turn-1",
            now=1_001.0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: claim(), range(8)))

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    stored = ledger.get_delivery(first_id := winners[0].delivery_id)
    assert stored is not None
    assert stored.state is ledger.DeliveryState.SEND_CLAIMED
    assert stored.sanitized_content is None
    assert winners[0].sanitized_content == "sanitized final"
    raw = ledger._db_path().read_bytes()
    assert b"sanitized final" not in raw
    assert winners[0].claim_token.encode() not in raw
    assert first_id


def test_cooperative_process_lock_allows_exactly_one_racing_sender():
    _reserve()
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_claim_in_process, args=(start, results))
        for _ in range(6)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert [results.get(timeout=1) for _ in processes].count(True) == 1


def test_unsafe_parent_is_rejected_without_mutation(_isolated_db):
    _isolated_db.parent.chmod(0o777)

    with pytest.raises(ledger.LedgerTrustError):
        _reserve()

    assert not _isolated_db.exists()
    assert stat.S_IMODE(_isolated_db.parent.stat().st_mode) == 0o777


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_unsafe_database_target_is_rejected_without_mutation(
    _isolated_db, tmp_path, kind
):
    victim = tmp_path / "victim"
    victim.write_bytes(b"do-not-touch")
    victim.chmod(0o640)
    if kind == "symlink":
        _isolated_db.symlink_to(victim)
    elif kind == "hardlink":
        os.link(victim, _isolated_db)
    else:
        os.mkfifo(_isolated_db, 0o600)

    with pytest.raises(ledger.LedgerTrustError):
        _reserve()

    assert victim.read_bytes() == b"do-not-touch"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o640


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_unsafe_sqlite_sidecar_is_rejected_without_mutation(
    _isolated_db, tmp_path, suffix, kind
):
    _isolated_db.write_bytes(b"")
    _isolated_db.chmod(0o600)
    sidecar = type(_isolated_db)(f"{_isolated_db}{suffix}")
    victim = tmp_path / f"victim{suffix}"
    victim.write_bytes(b"sidecar-do-not-touch")
    victim.chmod(0o640)
    if kind == "symlink":
        sidecar.symlink_to(victim)
    elif kind == "hardlink":
        os.link(victim, sidecar)
    else:
        os.mkfifo(sidecar, 0o600)

    with pytest.raises(ledger.LedgerTrustError):
        _reserve()

    assert victim.read_bytes() == b"sidecar-do-not-touch"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o640


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_unsafe_cooperative_lock_is_rejected_without_mutation(
    _isolated_db, tmp_path, kind
):
    lock = Path(f"{_isolated_db}.lock")
    victim = tmp_path / "lock-victim"
    victim.write_bytes(b"lock-do-not-touch")
    victim.chmod(0o640)
    if kind == "symlink":
        lock.symlink_to(victim)
    elif kind == "hardlink":
        os.link(victim, lock)
    else:
        os.mkfifo(lock, 0o600)

    with pytest.raises(ledger.LedgerTrustError):
        _reserve()

    assert victim.read_bytes() == b"lock-do-not-touch"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o640


def test_unsafe_sidecar_is_rejected_before_missing_database_is_created(
    _isolated_db, tmp_path
):
    victim = tmp_path / "sidecar-victim"
    victim.write_bytes(b"do-not-touch")
    sidecar = type(_isolated_db)(f"{_isolated_db}-wal")
    sidecar.symlink_to(victim)

    with pytest.raises(ledger.LedgerTrustError):
        _reserve()

    assert not _isolated_db.exists()
    assert victim.read_bytes() == b"do-not-touch"


def test_path_replacement_at_connect_seam_never_mutates_victim(
    _isolated_db, tmp_path, monkeypatch
):
    _isolated_db.touch(mode=0o600)
    victim = tmp_path / "victim.sqlite3"
    with sqlite3.connect(victim) as conn:
        conn.execute("CREATE TABLE victim_marker (value TEXT NOT NULL)")
        conn.execute("INSERT INTO victim_marker VALUES ('untouched')")
    victim.chmod(0o600)
    victim_before = victim.read_bytes()
    displaced = tmp_path / "displaced.sqlite3"
    real_connect = sqlite3.connect
    raced = False

    def racing_connect(database, *args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            os.replace(_isolated_db, displaced)
            _isolated_db.symlink_to(victim)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(ledger.sqlite3, "connect", racing_connect)

    with pytest.raises(ledger.LedgerTrustError):
        _reserve()

    assert raced is True
    assert victim.read_bytes() == victim_before
    with real_connect(victim) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables == {"victim_marker"}


def test_failure_before_atomic_rename_preserves_previous_valid_database(
    _isolated_db, monkeypatch
):
    first = _reserve()

    def fail_before_rename(*_args, **_kwargs):
        raise OSError("rename failpoint")

    monkeypatch.setattr(ledger.os, "replace", fail_before_rename)

    with pytest.raises(ledger.LedgerTrustError):
        _reserve(binding="second-binding", content="second final")

    with sqlite3.connect(_isolated_db) as conn:
        rows = conn.execute(
            "SELECT delivery_id, sanitized_content FROM plugin_deliveries"
        ).fetchall()
    assert rows == [(first.delivery_id, "sanitized final")]


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_database_exact_publish_seam_never_mutates_victim(
    _isolated_db, tmp_path, monkeypatch, kind
):
    victim = tmp_path / "database-victim"
    victim.write_bytes(b"database-victim-must-not-change")
    victim.chmod(0o600)
    victim_before = victim.read_bytes()
    real_replace = os.replace
    raced = False

    def racing_replace(source, destination, *args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            if kind == "symlink":
                _isolated_db.symlink_to(victim)
            else:
                os.link(victim, _isolated_db)
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(ledger.os, "replace", racing_replace)

    record = _reserve()

    assert record.state is ledger.DeliveryState.PENDING
    assert raced is True
    assert victim.read_bytes() == victim_before


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_sidecar_exact_publish_seam_never_mutates_victim(
    _isolated_db, tmp_path, monkeypatch, suffix, kind
):
    victim = tmp_path / f"victim{suffix}"
    victim.write_bytes(b"sidecar-victim-must-not-change")
    victim.chmod(0o600)
    victim_before = victim.read_bytes()
    sidecar = Path(f"{_isolated_db}{suffix}")
    real_replace = os.replace
    raced = False

    def racing_replace(source, destination, *args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            if kind == "symlink":
                sidecar.symlink_to(victim)
            else:
                os.link(victim, sidecar)
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(ledger.os, "replace", racing_replace)

    with pytest.raises(ledger.LedgerCommitUncertain):
        _reserve()

    assert raced is True
    assert victim.read_bytes() == victim_before


def test_failure_after_atomic_rename_before_directory_fsync_is_uncertain_and_restartable(
    _isolated_db, monkeypatch
):
    real_replace = os.replace
    real_fsync = os.fsync
    renamed = False

    def recording_replace(*args, **kwargs):
        nonlocal renamed
        result = real_replace(*args, **kwargs)
        renamed = True
        return result

    def fail_post_rename_directory_fsync(descriptor):
        if renamed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failpoint")
        return real_fsync(descriptor)

    monkeypatch.setattr(ledger.os, "replace", recording_replace)
    monkeypatch.setattr(ledger.os, "fsync", fail_post_rename_directory_fsync)

    with pytest.raises(ledger.LedgerCommitUncertain):
        _reserve()

    monkeypatch.setattr(ledger.os, "replace", real_replace)
    monkeypatch.setattr(ledger.os, "fsync", real_fsync)
    recovered = ledger.recover_after_restart(now=1_001.0)
    assert len(recovered) == 1
    assert recovered[0].content == "sanitized final"


def test_missing_sqlite_snapshot_support_fails_closed(_isolated_db, monkeypatch):
    monkeypatch.setattr(ledger, "_sqlite_snapshot_supported", lambda: False)

    with pytest.raises(ledger.LedgerTrustError):
        _reserve()

    assert not _isolated_db.exists()


def test_restart_resumes_only_pending_and_claimed_becomes_uncertain():
    pending = _reserve(binding="pending-binding")
    claimed = _reserve(binding="claimed-binding", content="second final")
    claim = ledger.claim_for_send(
        plugin_id="test.plugin",
        binding_handle="claimed-binding",
        delivery_key="turn-1",
        now=1_001.0,
    )
    assert claim is not None

    recoverable = ledger.recover_after_restart(now=1_002.0)

    assert [item.delivery_id for item in recoverable] == [pending.delivery_id]
    assert vars(recoverable[0]) == {
        "delivery_id": pending.delivery_id,
        "content": "sanitized final",
        "binding_digest": pending.binding_digest,
        "recovery_context_digest": pending.recovery_context_digest,
        "reconciliation_digest": None,
    }
    restarted = ledger.get_delivery(claimed.delivery_id)
    assert restarted is not None
    assert restarted.state is ledger.DeliveryState.DELIVERY_UNCERTAIN
    assert (
        ledger.claim_for_send(
            plugin_id="test.plugin",
            binding_handle="claimed-binding",
            delivery_key="turn-1",
            now=1_003.0,
        )
        is None
    )


def test_recovered_pending_delivery_can_be_claimed_and_completed(_isolated_db):
    record = _reserve()
    [pending] = ledger.recover_after_restart(now=1_001.0)

    claim = ledger.claim_pending_delivery(
        delivery_id=pending.delivery_id,
        plugin_id="test.plugin",
        binding_digest=pending.binding_digest,
        recovery_context_digest=pending.recovery_context_digest,
        now=1_002.0,
    )

    assert claim is not None
    assert claim.delivery_id == record.delivery_id
    assert claim.sanitized_content == "sanitized final"
    claimed = ledger.get_delivery(record.delivery_id)
    assert claimed is not None
    assert claimed.state is ledger.DeliveryState.SEND_CLAIMED
    assert claimed.sanitized_content is None
    raw = _isolated_db.read_bytes()
    assert b"sanitized final" not in raw
    assert claim.claim_token.encode() not in raw
    assert ledger._digest(claim.claim_token).encode() in raw
    assert ledger.mark_delivered(
        claim.delivery_id,
        claim.claim_token,
        receipt_id="recovery-42",
        now=1_003.0,
    )
    delivered = ledger.get_delivery(record.delivery_id)
    assert delivered is not None
    assert delivered.state is ledger.DeliveryState.DELIVERED
    assert (
        ledger.claim_pending_delivery(
            delivery_id=record.delivery_id,
            plugin_id=record.plugin_id,
            binding_digest=record.binding_digest,
            recovery_context_digest=_required_record_context(record),
            now=1_004.0,
        )
        is None
    )


def test_recovery_claim_is_bound_to_original_host_context_without_raw_storage(
    _isolated_db,
):
    original = _recovery_context_digest()
    record = ledger.reserve_delivery(
        plugin_id="test.plugin",
        binding_handle="opaque.secret.binding",
        delivery_key="context-bound-turn",
        sanitized_content="context-bound final",
        recovery_context_digest=original,
        now=1_000.0,
    )
    [pending] = ledger.recover_after_restart(now=1_001.0)

    for moved in (
        _recovery_context_digest(session="session-2"),
        _recovery_context_digest(chat="chat-2"),
        _recovery_context_digest(topic="topic-2"),
        _recovery_context_digest(user="user-2"),
        _recovery_context_digest(profile="profile-2"),
    ):
        assert (
            ledger.claim_pending_delivery(
                delivery_id=record.delivery_id,
                plugin_id=record.plugin_id,
                binding_digest=record.binding_digest,
                recovery_context_digest=moved,
                now=1_002.0,
            )
            is None
        )
        unchanged = ledger.get_delivery(record.delivery_id)
        assert unchanged is not None
        assert unchanged.state is ledger.DeliveryState.PENDING
        assert unchanged.sanitized_content == "context-bound final"

    claim = ledger.claim_pending_delivery(
        delivery_id=pending.delivery_id,
        plugin_id=record.plugin_id,
        binding_digest=pending.binding_digest,
        recovery_context_digest=pending.recovery_context_digest,
        now=1_003.0,
    )

    assert claim is not None
    raw = _isolated_db.read_bytes()
    for raw_context in (b"session-1", b"chat-1", b"topic-1", b"user-1", b"profile-1"):
        assert raw_context not in raw


def test_recovery_context_is_immutable_for_idempotent_reservation():
    record = _reserve()

    with pytest.raises(ledger.DeliveryConflictError, match="recovery context"):
        _reserve(recovery_context_digest=_recovery_context_digest(session="moved"))

    unchanged = ledger.get_delivery(record.delivery_id)
    assert unchanged is not None
    assert unchanged.recovery_context_digest == _DEFAULT_RECOVERY_CONTEXT_DIGEST


def test_missing_recovery_context_is_fail_closed_only_at_restart(_isolated_db):
    record = _reserve(recovery_context_digest=None)
    claim = ledger.claim_for_send(
        plugin_id=record.plugin_id,
        binding_handle="opaque.secret.binding",
        delivery_key="turn-1",
        now=1_001.0,
    )
    assert claim is not None
    assert ledger.mark_failed(
        claim.delivery_id, claim.claim_token, error="immediate path", now=1_002.0
    )

    pending = _reserve(
        binding="legacy-pending-binding",
        content="legacy recovery secret",
        recovery_context_digest=None,
    )
    assert ledger.recover_after_restart(now=1_003.0) == []
    failed = ledger.get_delivery(pending.delivery_id)
    assert failed is not None
    assert failed.state is ledger.DeliveryState.FAILED
    assert failed.last_error == "recovery_context_missing"
    assert failed.sanitized_content is None
    assert b"legacy recovery secret" not in _isolated_db.read_bytes()


def test_reservation_rejects_malformed_recovery_context_before_database_write(
    _isolated_db,
):
    with pytest.raises(ValueError, match="recovery_context_digest"):
        _reserve(recovery_context_digest="not-a-sha256")

    assert not _isolated_db.exists()


def test_reconciliation_id_is_digest_only_and_supports_reserve_then_claim(
    _isolated_db,
):
    reconciliation_id = "consultation-018f4f7e-7d8f-7000-8000-123456789abc"
    record = _reserve(reconciliation_id=reconciliation_id)

    assert (
        record.reconciliation_digest
        == hashlib.sha256(reconciliation_id.encode()).hexdigest()
    )
    assert reconciliation_id.encode() not in _isolated_db.read_bytes()
    looked_up = ledger.get_delivery_by_reconciliation_id(
        plugin_id="test.plugin",
        reconciliation_id=reconciliation_id,
    )
    assert looked_up is not None
    assert looked_up == record
    assert looked_up.state is ledger.DeliveryState.PENDING
    assert not hasattr(looked_up, "claim_token")

    duplicate = _reserve(reconciliation_id=reconciliation_id)
    assert duplicate == record
    claim = ledger.claim_for_send(
        plugin_id="test.plugin",
        binding_handle="opaque.secret.binding",
        delivery_key="turn-1",
        now=1_001.0,
    )
    assert claim is not None
    assert (
        ledger.claim_for_send(
            plugin_id="test.plugin",
            binding_handle="opaque.secret.binding",
            delivery_key="turn-1",
            now=1_001.1,
        )
        is None
    )


@pytest.mark.parametrize(
    "reconciliation_id",
    ["", "has space", "has/slash", "한글-id", "x" * 129],
)
def test_reconciliation_id_rejects_unsafe_or_unbounded_values(
    _isolated_db, reconciliation_id
):
    with pytest.raises(ValueError, match="reconciliation_id"):
        _reserve(reconciliation_id=reconciliation_id)

    assert not _isolated_db.exists()


def test_reconciliation_lookup_is_plugin_scoped_and_immutable():
    reconciliation_id = "consultation-immutable-1"
    record = _reserve(reconciliation_id=reconciliation_id)

    assert (
        ledger.get_delivery_by_reconciliation_id(
            plugin_id="other.plugin",
            reconciliation_id=reconciliation_id,
        )
        is None
    )
    with pytest.raises(ledger.DeliveryConflictError, match="reconciliation"):
        ledger.reserve_delivery(
            plugin_id="test.plugin",
            binding_handle="other.binding",
            delivery_key="turn-2",
            sanitized_content="other final",
            recovery_context_digest=_DEFAULT_RECOVERY_CONTEXT_DIGEST,
            reconciliation_id=reconciliation_id,
            now=1_001.0,
        )
    with pytest.raises(ledger.DeliveryConflictError, match="reconciliation"):
        _reserve(reconciliation_id="consultation-immutable-2")

    unchanged = ledger.get_delivery(record.delivery_id)
    assert unchanged == record


def test_reconciliation_lookup_reports_every_lifecycle_state_after_restart():
    reconciliation_id = "consultation-lifecycle-1"
    record = _reserve(reconciliation_id=reconciliation_id)

    def state() -> ledger.DeliveryState:
        found = ledger.get_delivery_by_reconciliation_id(
            plugin_id="test.plugin",
            reconciliation_id=reconciliation_id,
        )
        assert found is not None
        return found.state

    assert state() is ledger.DeliveryState.PENDING
    claim = ledger.claim_for_send(
        plugin_id="test.plugin",
        binding_handle="opaque.secret.binding",
        delivery_key="turn-1",
        now=1_001.0,
    )
    assert claim is not None
    assert state() is ledger.DeliveryState.SEND_CLAIMED
    assert ledger.recover_after_restart(now=1_002.0) == []
    assert state() is ledger.DeliveryState.DELIVERY_UNCERTAIN

    delivered_id = "consultation-lifecycle-delivered"
    delivered = ledger.reserve_delivery(
        plugin_id="test.plugin",
        binding_handle="delivered.binding",
        delivery_key="turn-delivered",
        sanitized_content="delivered final",
        recovery_context_digest=_DEFAULT_RECOVERY_CONTEXT_DIGEST,
        reconciliation_id=delivered_id,
        now=1_003.0,
    )
    delivered_claim = ledger.claim_for_send(
        plugin_id="test.plugin",
        binding_handle="delivered.binding",
        delivery_key="turn-delivered",
        now=1_004.0,
    )
    assert delivered_claim is not None
    assert ledger.mark_delivered(
        delivered.delivery_id, delivered_claim.claim_token, now=1_005.0
    )
    delivered_status = ledger.get_delivery_by_reconciliation_id(
        plugin_id="test.plugin", reconciliation_id=delivered_id
    )
    assert delivered_status is not None
    assert delivered_status.state is ledger.DeliveryState.DELIVERED

    failed_id = "consultation-lifecycle-failed"
    failed = ledger.reserve_delivery(
        plugin_id="test.plugin",
        binding_handle="failed.binding",
        delivery_key="turn-failed",
        sanitized_content="failed final",
        recovery_context_digest=_DEFAULT_RECOVERY_CONTEXT_DIGEST,
        reconciliation_id=failed_id,
        now=1_006.0,
    )
    assert ledger.cancel_pending_delivery(
        delivery_id=failed.delivery_id,
        plugin_id=failed.plugin_id,
        binding_digest=failed.binding_digest,
        recovery_context_digest=_required_record_context(failed),
        now=1_007.0,
    )
    failed_status = ledger.get_delivery_by_reconciliation_id(
        plugin_id="test.plugin", reconciliation_id=failed_id
    )
    assert failed_status is not None
    assert failed_status.state is ledger.DeliveryState.FAILED

    assert ledger.recover_after_restart(now=1_008.0) == []
    for terminal_id, expected_state in (
        (reconciliation_id, ledger.DeliveryState.DELIVERY_UNCERTAIN),
        (delivered_id, ledger.DeliveryState.DELIVERED),
        (failed_id, ledger.DeliveryState.FAILED),
    ):
        restarted = ledger.get_delivery_by_reconciliation_id(
            plugin_id="test.plugin", reconciliation_id=terminal_id
        )
        assert restarted is not None
        assert restarted.state is expected_state


def test_concurrent_reconciliation_reservation_has_one_mapping():
    reconciliation_id = "consultation-concurrent-1"

    def reserve(index: int):
        try:
            return ledger.reserve_delivery(
                plugin_id="test.plugin",
                binding_handle=f"binding-{index}",
                delivery_key=f"turn-{index}",
                sanitized_content=f"final {index}",
                recovery_context_digest=_DEFAULT_RECOVERY_CONTEXT_DIGEST,
                reconciliation_id=reconciliation_id,
                now=1_000.0 + index,
            )
        except ledger.DeliveryConflictError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(reserve, range(8)))

    winners = [record for record in outcomes if record is not None]
    assert len(winners) == 1
    assert (
        ledger.get_delivery_by_reconciliation_id(
            plugin_id="test.plugin", reconciliation_id=reconciliation_id
        )
        == winners[0]
    )


def test_cancel_pending_delivery_is_durable_and_securely_erases_content(
    _isolated_db, monkeypatch
):
    _force_secure_delete_default_off(monkeypatch)
    needle = b"cancel-pending-sensitive-Q"
    reconciliation_id = "consultation-cancel-secure-1"
    record = _reserve(
        content=_page_spanning_content(needle.decode()),
        reconciliation_id=reconciliation_id,
    )
    assert needle in _isolated_db.read_bytes()
    assert reconciliation_id.encode() not in _isolated_db.read_bytes()

    assert ledger.cancel_pending_delivery(
        delivery_id=record.delivery_id,
        plugin_id=record.plugin_id,
        binding_digest=record.binding_digest,
        recovery_context_digest=_required_record_context(record),
        reason="resolver_cleanup_cancelled",
        now=1_001.0,
    )

    cancelled = ledger.get_delivery(record.delivery_id)
    assert cancelled is not None
    assert cancelled.state is ledger.DeliveryState.FAILED
    assert cancelled.last_error == "resolver_cleanup_cancelled"
    assert cancelled.sanitized_content is None
    assert needle not in _isolated_db.read_bytes()
    assert not ledger.cancel_pending_delivery(
        delivery_id=record.delivery_id,
        plugin_id=record.plugin_id,
        binding_digest=record.binding_digest,
        recovery_context_digest=_required_record_context(record),
        reason="must_not_replace_terminal_reason",
        now=1_001.5,
    )
    unchanged = ledger.get_delivery(record.delivery_id)
    assert unchanged is not None
    assert unchanged.last_error == "resolver_cleanup_cancelled"
    assert ledger.recover_after_restart(now=1_002.0) == []
    recovered_status = ledger.get_delivery_by_reconciliation_id(
        plugin_id="test.plugin", reconciliation_id=reconciliation_id
    )
    assert recovered_status is not None
    assert recovered_status.state is ledger.DeliveryState.FAILED


def test_cancel_pending_delivery_mismatch_and_nonpending_are_noops():
    record = _reserve()
    context = _required_record_context(record)
    mismatches = (
        {"delivery_id": "0" * 32},
        {"plugin_id": "wrong.plugin"},
        {"binding_digest": "0" * 64},
        {"recovery_context_digest": "0" * 64},
    )
    base = {
        "delivery_id": record.delivery_id,
        "plugin_id": record.plugin_id,
        "binding_digest": record.binding_digest,
        "recovery_context_digest": context,
    }
    for mismatch in mismatches:
        assert not ledger.cancel_pending_delivery(
            **(base | mismatch), reason="must_not_mutate", now=1_001.0
        )
        unchanged = ledger.get_delivery(record.delivery_id)
        assert unchanged is not None
        assert unchanged.state is ledger.DeliveryState.PENDING
        assert unchanged.sanitized_content == "sanitized final"

    claim = ledger.claim_pending_delivery(**base, now=1_002.0)
    assert claim is not None
    assert not ledger.cancel_pending_delivery(**base, now=1_003.0)
    claimed = ledger.get_delivery(record.delivery_id)
    assert claimed is not None
    assert claimed.state is ledger.DeliveryState.SEND_CLAIMED


def test_thread_claim_cancel_race_has_exactly_one_winner():
    record = _reserve()
    context = _required_record_context(record)
    start = threading.Event()

    def claim() -> bool:
        start.wait()
        return (
            ledger.claim_pending_delivery(
                delivery_id=record.delivery_id,
                plugin_id=record.plugin_id,
                binding_digest=record.binding_digest,
                recovery_context_digest=context,
                now=1_001.0,
            )
            is not None
        )

    def cancel() -> bool:
        start.wait()
        return ledger.cancel_pending_delivery(
            delivery_id=record.delivery_id,
            plugin_id=record.plugin_id,
            binding_digest=record.binding_digest,
            recovery_context_digest=context,
            now=1_001.0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim), pool.submit(cancel)]
        start.set()
        outcomes = [future.result() for future in futures]

    assert outcomes.count(True) == 1
    final = ledger.get_delivery(record.delivery_id)
    assert final is not None
    assert final.state in {
        ledger.DeliveryState.SEND_CLAIMED,
        ledger.DeliveryState.FAILED,
    }
    assert final.sanitized_content is None


def test_process_claim_cancel_race_has_exactly_one_winner():
    record = _reserve()
    context_digest = _required_record_context(record)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_recovery_claim_in_process,
            args=(
                start,
                results,
                record.delivery_id,
                record.binding_digest,
                context_digest,
            ),
        ),
        context.Process(
            target=_cancel_pending_in_process,
            args=(
                start,
                results,
                record.delivery_id,
                record.binding_digest,
                context_digest,
            ),
        ),
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert [results.get(timeout=1) for _ in processes].count(True) == 1
    final = ledger.get_delivery(record.delivery_id)
    assert final is not None
    assert final.state in {
        ledger.DeliveryState.SEND_CLAIMED,
        ledger.DeliveryState.FAILED,
    }
    assert final.sanitized_content is None


@pytest.mark.parametrize(
    "marker_name, expected_state",
    [
        ("mark_failed", ledger.DeliveryState.FAILED),
        ("mark_uncertain", ledger.DeliveryState.DELIVERY_UNCERTAIN),
    ],
)
def test_recovered_claim_supports_non_delivery_terminal_markers(
    marker_name, expected_state
):
    record = _reserve()
    ledger.recover_after_restart(now=1_001.0)
    claim = ledger.claim_pending_delivery(
        delivery_id=record.delivery_id,
        plugin_id=record.plugin_id,
        binding_digest=record.binding_digest,
        recovery_context_digest=_required_record_context(record),
        now=1_002.0,
    )
    assert claim is not None

    marker = getattr(ledger, marker_name)
    assert marker(
        claim.delivery_id,
        claim.claim_token,
        error="recovery terminal",
        now=1_003.0,
    )
    terminal = ledger.get_delivery(record.delivery_id)
    assert terminal is not None
    assert terminal.state is expected_state
    assert (
        ledger.claim_pending_delivery(
            delivery_id=record.delivery_id,
            plugin_id=record.plugin_id,
            binding_digest=record.binding_digest,
            recovery_context_digest=_required_record_context(record),
            now=1_004.0,
        )
        is None
    )


def test_recovery_claim_mismatch_does_not_mutate_pending_delivery():
    record = _reserve()

    assert (
        ledger.claim_pending_delivery(
            delivery_id=record.delivery_id,
            plugin_id="wrong.plugin",
            binding_digest=record.binding_digest,
            recovery_context_digest=_required_record_context(record),
            now=1_001.0,
        )
        is None
    )
    assert (
        ledger.claim_pending_delivery(
            delivery_id=record.delivery_id,
            plugin_id="test.plugin",
            binding_digest="0" * 64,
            recovery_context_digest=_required_record_context(record),
            now=1_001.0,
        )
        is None
    )
    assert (
        ledger.claim_pending_delivery(
            delivery_id="0" * 32,
            plugin_id="test.plugin",
            binding_digest=record.binding_digest,
            recovery_context_digest=_required_record_context(record),
            now=1_001.0,
        )
        is None
    )
    unchanged = ledger.get_delivery(record.delivery_id)
    assert unchanged is not None
    assert unchanged.state is ledger.DeliveryState.PENDING
    assert unchanged.sanitized_content == "sanitized final"


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "delivery_id": "short",
            "plugin_id": "test.plugin",
            "binding_digest": "a" * 64,
        },
        {
            "delivery_id": "A" * 32,
            "plugin_id": "test.plugin",
            "binding_digest": "a" * 64,
        },
        {
            "delivery_id": "a" * 32,
            "plugin_id": "bad plugin",
            "binding_digest": "a" * 64,
        },
        {
            "delivery_id": "a" * 32,
            "plugin_id": "test.plugin",
            "binding_digest": "short",
        },
        {
            "delivery_id": "a" * 32,
            "plugin_id": "test.plugin",
            "binding_digest": "A" * 64,
        },
        {
            "delivery_id": "a" * 32,
            "plugin_id": "test.plugin",
            "binding_digest": "a" * 64,
            "recovery_context_digest": "short",
        },
    ],
)
def test_recovery_claim_rejects_malformed_identifiers(_isolated_db, kwargs):
    kwargs.setdefault("recovery_context_digest", _DEFAULT_RECOVERY_CONTEXT_DIGEST)
    with pytest.raises(ValueError):
        ledger.claim_pending_delivery(**kwargs, now=1_001.0)

    assert not _isolated_db.exists()


def test_simultaneous_recovery_claim_has_exactly_one_thread_winner():
    record = _reserve()
    ledger.recover_after_restart(now=1_001.0)

    def claim():
        return ledger.claim_pending_delivery(
            delivery_id=record.delivery_id,
            plugin_id=record.plugin_id,
            binding_digest=record.binding_digest,
            recovery_context_digest=_required_record_context(record),
            now=1_002.0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: claim(), range(8)))

    assert len([claim for claim in claims if claim is not None]) == 1


def test_simultaneous_recovery_claim_has_exactly_one_process_winner():
    record = _reserve()
    ledger.recover_after_restart(now=1_001.0)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_recovery_claim_in_process,
            args=(
                start,
                results,
                record.delivery_id,
                record.binding_digest,
                _required_record_context(record),
            ),
        )
        for _ in range(6)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert [results.get(timeout=1) for _ in processes].count(True) == 1


def test_recovered_claim_becomes_uncertain_after_next_restart():
    record = _reserve()
    ledger.recover_after_restart(now=1_001.0)
    claim = ledger.claim_pending_delivery(
        delivery_id=record.delivery_id,
        plugin_id=record.plugin_id,
        binding_digest=record.binding_digest,
        recovery_context_digest=_required_record_context(record),
        now=1_002.0,
    )
    assert claim is not None

    assert ledger.recover_after_restart(now=1_003.0) == []
    restarted = ledger.get_delivery(record.delivery_id)
    assert restarted is not None
    assert restarted.state is ledger.DeliveryState.DELIVERY_UNCERTAIN
    assert (
        ledger.claim_pending_delivery(
            delivery_id=record.delivery_id,
            plugin_id=record.plugin_id,
            binding_digest=record.binding_digest,
            recovery_context_digest=_required_record_context(record),
            now=1_004.0,
        )
        is None
    )


def test_recovery_claim_expires_pending_content_without_returning_it():
    record = _reserve()

    assert (
        ledger.claim_pending_delivery(
            delivery_id=record.delivery_id,
            plugin_id=record.plugin_id,
            binding_digest=record.binding_digest,
            recovery_context_digest=_required_record_context(record),
            now=1_000.0 + ledger.CONTENT_RETENTION_SECONDS + 1,
        )
        is None
    )
    expired = ledger.get_delivery(record.delivery_id)
    assert expired is not None
    assert expired.state is ledger.DeliveryState.FAILED
    assert expired.sanitized_content is None


def test_claim_completion_requires_matching_claim_token():
    record = _reserve()
    claim = ledger.claim_for_send(
        plugin_id="test.plugin",
        binding_handle="opaque.secret.binding",
        delivery_key="turn-1",
        now=1_001.0,
    )
    assert claim is not None

    assert (
        ledger.mark_delivered(record.delivery_id, "wrong", receipt_id="42", now=1_002.0)
        is False
    )
    assert (
        ledger.mark_delivered(
            record.delivery_id, claim.claim_token, receipt_id="42", now=1_002.0
        )
        is True
    )
    delivered = ledger.get_delivery(record.delivery_id)
    assert delivered is not None
    assert delivered.state is ledger.DeliveryState.DELIVERED
    assert delivered.receipt_id == "42"


def test_claim_can_be_marked_failed_immediately():
    record = _reserve()
    claim = ledger.claim_for_send(
        plugin_id="test.plugin",
        binding_handle="opaque.secret.binding",
        delivery_key="turn-1",
        now=1_001.0,
    )
    assert claim is not None

    assert ledger.mark_failed(
        record.delivery_id,
        claim.claim_token,
        error="preflight_failed",
        now=1_001.1,
    )
    failed = ledger.get_delivery(record.delivery_id)
    assert failed is not None
    assert failed.state is ledger.DeliveryState.FAILED


def test_uncertain_delivery_is_terminal_and_cannot_be_claimed_again():
    record = _reserve()
    claim = ledger.claim_for_send(
        plugin_id="test.plugin",
        binding_handle="opaque.secret.binding",
        delivery_key="turn-1",
        now=1_001.0,
    )
    assert claim is not None
    assert ledger.mark_uncertain(
        record.delivery_id,
        claim.claim_token,
        error="timeout after request",
        now=1_002.0,
    )

    assert (
        ledger.claim_for_send(
            plugin_id="test.plugin",
            binding_handle="opaque.secret.binding",
            delivery_key="turn-1",
            now=1_003.0,
        )
        is None
    )


def test_retention_removes_content_at_24h_and_metadata_at_30d():
    record = _reserve()
    claim = ledger.claim_for_send(
        plugin_id="test.plugin",
        binding_handle="opaque.secret.binding",
        delivery_key="turn-1",
        now=1_001.0,
    )
    assert claim is not None
    ledger.mark_failed(
        record.delivery_id, claim.claim_token, error="definitive rejection", now=1_002.0
    )

    ledger.prune(now=1_000.0 + ledger.CONTENT_RETENTION_SECONDS + 1)
    retained = ledger.get_delivery(record.delivery_id)
    assert retained is not None
    assert retained.sanitized_content is None

    ledger.prune(now=1_000.0 + ledger.METADATA_RETENTION_SECONDS + 1)
    assert ledger.get_delivery(record.delivery_id) is None


def test_expired_pending_content_is_not_returned_for_delivery():
    record = _reserve()

    assert (
        ledger.recover_after_restart(now=1_000.0 + ledger.CONTENT_RETENTION_SECONDS + 1)
        == []
    )
    expired = ledger.get_delivery(record.delivery_id)
    assert expired is not None
    assert expired.state is ledger.DeliveryState.FAILED
    assert expired.sanitized_content is None


def test_schema_contains_no_raw_prompt_account_or_token_columns(_isolated_db):
    _reserve()
    with sqlite3.connect(_isolated_db) as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(plugin_deliveries)")
        }
    assert not columns.intersection({
        "prompt",
        "account",
        "account_number",
        "token",
        "binding_handle",
    })
