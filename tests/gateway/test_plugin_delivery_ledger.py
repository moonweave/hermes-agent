from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from gateway import plugin_delivery_ledger as ledger


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    path = tmp_path / "profile" / "plugin-delivery.sqlite3"
    path.parent.mkdir(mode=0o700)
    monkeypatch.setattr(ledger, "_db_path", lambda: path)
    return path


def _reserve(
    *, content: str = "sanitized final", binding: str = "opaque.secret.binding"
):
    return ledger.reserve_delivery(
        plugin_id="test.plugin",
        binding_handle=binding,
        delivery_key="turn-1",
        sanitized_content=content,
        now=1_000.0,
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
