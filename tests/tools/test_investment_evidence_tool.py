from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest

from tools import investment_evidence_tool as evidence


def _write_evidence(path: Path, content: str, tool_name: str = "mcp__kospi_investment__get_market_context") -> None:
    path.write_text(content, encoding="utf-8")
    path.with_name(f"{path.name}.meta.json").write_text(json.dumps({
        "version": 1, "filename": path.name, "tool_name": tool_name,
        "tool_use_id": "sanitized-tool-call",
        "session_id": "test-session", "requester_id": "test-requester",
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }), encoding="utf-8")


@pytest.fixture
def spillover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = tmp_path / "cache" / "spillover"
    root.mkdir(parents=True)
    monkeypatch.setattr(evidence, "get_spillover_dir", lambda: root)
    real_handler = evidence._handle_investment_evidence

    def scoped_handler(args, task_id=None, **kwargs):
        kwargs.setdefault("session_id", "test-session")
        kwargs.setdefault("requester_id", "test-requester")
        return real_handler(args, task_id=task_id, **kwargs)

    monkeypatch.setattr(evidence, "_handle_investment_evidence", scoped_handler)
    return root


def test_reader_pages_existing_spillover_and_separates_wrapper_freshness(
    spillover: Path,
) -> None:
    path = spillover / "call_sanitized.txt"
    _write_evidence(path, "provider_as_of=2026-09-07T01:00:00Z\nMEANINGFUL_FACT\n")

    result = json.loads(
        evidence._handle_investment_evidence(
            {"reference": str(path), "offset": 2, "limit": 1}
        )
    )

    assert result["success"] is True
    assert "MEANINGFUL_FACT" in result["content"]
    assert result["source_filename"] == path.name
    assert result["evidence_ref"] == f"hermes://spillover/{path.name}"
    assert result["wrapper_freshness"] == "persistence_time_only"
    assert result["freshness_authority"] == "provider_payload_as_of"
    assert result["provider_retry"] is False


def test_reader_rejects_traversal_and_symlink(spillover: Path, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("credential=never-return", encoding="utf-8")
    link = spillover / "link.txt"
    link.symlink_to(outside)

    traversal = json.loads(
        evidence._handle_investment_evidence({"reference": "../secret.txt"})
    )
    linked = json.loads(evidence._handle_investment_evidence({"reference": str(link)}))

    assert "active profile cache/spillover" in traversal["error"]
    assert "active profile cache/spillover" in linked["error"]
    assert "credential=never-return" not in json.dumps(linked)


def test_reader_reports_expired_data_without_provider_retry(spillover: Path) -> None:
    path = spillover / "old.txt"
    _write_evidence(path, "flow=unknown")
    old = os.stat(path).st_mtime - 25 * 3600
    os.utime(path, (old, old))

    result = json.loads(evidence._handle_investment_evidence({"reference": path.name}))

    assert result["success"] is False
    assert result["status"] == "degraded"
    assert result["unknown"] is True
    assert result["provider_retry"] is False


def test_registry_exposes_the_dedicated_reader() -> None:
    entry = evidence.registry.get_entry("investment_evidence")
    assert entry is not None
    assert entry.toolset == "investment_evidence"
    assert "write_file" not in entry.schema["description"]


def test_real_storage_and_full_pagination(tmp_path, monkeypatch):
    from tools.tool_result_storage import maybe_persist_tool_result, extract_persisted_path, get_spillover_dir
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(evidence, 'get_spillover_dir', get_spillover_dir)
    content = ''.join(f'line_{i}: observed source data\n' for i in range(1800)) + 'TAIL_FACT: provider_flow_unavailable\n'
    preview = maybe_persist_tool_result(
        content, 'mcp__kospi_investment__get_market_context', 'tail_evidence',
        threshold=1000, session_id="test-session", requester_id="test-requester",
    )
    reference = extract_persisted_path(preview)
    assert reference and 'TAIL_FACT' not in preview
    pages = []
    for offset in range(1, 1802, 400):
        page = json.loads(evidence._handle_investment_evidence({'reference': reference, 'offset': offset, 'limit': 400}, session_id="test-session", requester_id="test-requester"))
        assert page['success'] and page['freshness_authority'] == 'provider_payload_as_of'
        pages.append(page['content'])
    assert 'TAIL_FACT: provider_flow_unavailable' in ''.join(pages)


def test_real_single_line_nested_json_is_read_completely(spillover):
    source = json.dumps({"result": json.dumps({
        "rows": ["합성 근거 " + "x" * 900 for _ in range(80)],
        "tail": "TAIL_FACT_SINGLE_LINE_Q91",
    }, ensure_ascii=False)}, ensure_ascii=False)
    path = spillover / "single_line.txt"
    _write_evidence(path, source)
    args = {"reference": path.name}
    pages = []
    for _ in range(10):
        page = json.loads(evidence._handle_investment_evidence(args))
        assert page["success"], page
        pages.append(page["content"])
        if not page.get("truncated"):
            break
        assert page["next_character_offset"] > args.get("character_offset", 0)
        args = {"reference": path.name, "character_offset": page["next_character_offset"]}
    assert "TAIL_FACT_SINGLE_LINE_Q91" in "".join(pages)
    assert json.loads("".join(pages)) == json.loads(source)
    assert path.read_text(encoding="utf-8") == source
    assert all(len(page) <= 20_000 for page in pages)


def test_character_pages_redact_before_slicing(spillover):
    token = "sk-proj-" + "A" * 70
    path = spillover / "redaction.txt"
    _write_evidence(path, "p" * 19973 + "Bearer " + token + " visible_tail")
    first = json.loads(evidence._handle_investment_evidence({"reference": path.name, "character_offset": 1}))
    combined = first["content"]
    if first["truncated"]:
        last = json.loads(evidence._handle_investment_evidence({
            "reference": path.name, "character_offset": first["next_character_offset"],
        }))
        combined += last["content"]
    assert token not in combined
    assert "A" * 40 not in combined
    assert "visible_tail" in combined


@pytest.mark.parametrize("offset", [0, -1, True, "1", 10**30])
def test_character_offset_rejects_invalid_or_out_of_range_values(spillover, offset):
    _write_evidence(spillover / "small.txt", "evidence")
    page = json.loads(evidence._handle_investment_evidence({"reference": "small.txt", "character_offset": offset}))
    assert page["success"] is False
    assert page["status"] == "degraded" and page["unknown"] is True


def test_character_page_rejects_leaf_swap_without_exposing_outside_file(spillover, tmp_path, monkeypatch):
    path = spillover / "race.txt"
    _write_evidence(path, "safe")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-confidential-evidence", encoding="utf-8")
    original_open = os.open

    def raced_open(name, flags, *args, **kwargs):
        if name == "race.txt" and "dir_fd" in kwargs:
            path.unlink()
            path.symlink_to(outside)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(evidence.os, "open", raced_open)
    monkeypatch.setattr(evidence.os, "supports_dir_fd", os.supports_dir_fd | {raced_open})
    page = json.loads(evidence._handle_investment_evidence({"reference": path.name, "character_offset": 1}))
    assert page["success"] is False
    assert "outside-confidential-evidence" not in json.dumps(page)
    assert path.is_symlink()


def test_oversized_character_source_is_degraded_without_partial_success(spillover):
    path = spillover / "large.txt"
    _write_evidence(path, "x" * (8 * 1024 * 1024 + 1))
    page = json.loads(evidence._handle_investment_evidence({"reference": path.name, "character_offset": 1}))
    assert page["success"] is False
    assert page["status"] == "degraded" and page["unknown"] is True


def test_character_page_fifo_swap_is_opened_nonblocking_and_rejected(spillover, monkeypatch):
    path = spillover / "fifo-race.txt"
    _write_evidence(path, "safe")
    original_open = os.open

    def raced_open(name, flags, *args, **kwargs):
        if name == path.name and "dir_fd" in kwargs:
            assert flags & os.O_NONBLOCK
            path.unlink()
            os.mkfifo(path)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(evidence.os, "open", raced_open)
    monkeypatch.setattr(evidence.os, "supports_dir_fd", os.supports_dir_fd | {raced_open})
    page = json.loads(evidence._handle_investment_evidence({"reference": path.name, "character_offset": 1}))
    assert page["success"] is False and page["status"] == "degraded"
    assert path.is_fifo()


def test_character_mode_preserves_the_existing_internal_read_guard(spillover, monkeypatch):
    path = spillover / "blocked.txt"
    _write_evidence(path, "must-not-be-returned")
    monkeypatch.setattr(evidence, "get_read_block_error", lambda _: "protected internal file")
    page = json.loads(evidence._handle_investment_evidence({"reference": path.name, "character_offset": 1}))
    assert page["success"] is False and page["status"] == "degraded"
    assert "must-not-be-returned" not in json.dumps(page)
    assert page["error"] == "protected internal file"


def test_independent_analysts_do_not_share_the_line_read_dedup_cache(spillover):
    path = spillover / "independent.txt"
    _write_evidence(path, "INDEPENDENT_EVIDENCE_READ\n")
    args = {"reference": path.name, "offset": 1, "limit": 1}
    for task_id in ("evidence-analyst-a", "evidence-analyst-b"):
        page = json.loads(evidence._handle_investment_evidence(args, task_id=task_id))
        assert page["success"]
        assert "INDEPENDENT_EVIDENCE_READ" in page["content"]


@pytest.mark.parametrize("character", [False, True])
def test_reader_denies_unauthorized_and_legacy_provenance(spillover, character):
    path = spillover / "account-result.txt"
    _write_evidence(path, "account_balance=100", tool_name="get_balance")
    args = {"reference": path.name}
    if character:
        args["character_offset"] = 1
    denied = json.loads(evidence._handle_investment_evidence(args))
    assert denied["success"] is False and denied["unknown"] is True
    assert "account_balance" not in json.dumps(denied)

    legacy = spillover / "legacy.txt"
    legacy.write_text("legacy private result", encoding="utf-8")
    args = {"reference": legacy.name}
    if character:
        args["character_offset"] = 1
    denied = json.loads(evidence._handle_investment_evidence(args))
    assert denied["success"] is False and denied["status"] == "degraded"


def test_reader_denies_unrelated_session_and_digest_replacement(spillover):
    path = spillover / "isolated.txt"
    _write_evidence(path, "ORIGINAL_FACT\n")
    path.with_name(f"{path.name}.meta.json").write_text(json.dumps({
        "version": 1, "filename": path.name,
        "tool_name": "mcp__kospi_investment__get_pressure_context",
        "tool_use_id": "pressure-call", "session_id": "other-session",
        "requester_id": "other-requester",
        "content_sha256": hashlib.sha256(b"ORIGINAL_FACT\n").hexdigest(),
    }))
    denied = json.loads(evidence._handle_investment_evidence({"reference": path.name}))
    assert denied["success"] is False and "ORIGINAL_FACT" not in json.dumps(denied)

    _write_evidence(path, "REPLACED_FACT\n")
    meta = json.loads(
        path.with_name(f"{path.name}.meta.json").read_text(encoding="utf-8")
    )
    meta["content_sha256"] = hashlib.sha256(b"ORIGINAL_FACT\n").hexdigest()
    path.with_name(f"{path.name}.meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    denied = json.loads(evidence._handle_investment_evidence({"reference": path.name, "character_offset": 1}))
    assert denied["success"] is False and "REPLACED_FACT" not in json.dumps(denied)


def test_parent_session_can_read_delegated_child_evidence(spillover):
    from hermes_state import SessionDB

    db = SessionDB(spillover.parent.parent / "state.db")
    db.create_session("test-session", "cli")
    db.create_session("child-session", "delegate", parent_session_id="test-session")
    db.close()
    path = spillover / "child.txt"
    _write_evidence(path, "DELEGATED_FACT\n")
    meta_path = path.with_name(f"{path.name}.meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(session_id="child-session", parent_session_id="test-session", requester_id="child-task")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    result = json.loads(evidence._handle_investment_evidence(
        {"reference": path.name}, session_id="test-session", requester_id="parent-task",
    ))
    assert result["success"] is True and "DELEGATED_FACT" in result["content"]


@pytest.mark.parametrize("producer", [
    "mcp__kospi_investment__get_pressure_context",
    "mcp__kr_fundamentals__get_recent_disclosure_events_tool",
])
def test_enabled_non_account_producers_round_trip(spillover, producer):
    path = spillover / "allowed.txt"
    _write_evidence(path, "ALLOWED_RESEARCH_FACT\n", tool_name=producer)
    result = json.loads(evidence._handle_investment_evidence({"reference": path.name}))
    assert result["success"] is True
    assert "ALLOWED_RESEARCH_FACT" in result["content"]


def _persist_native(content, session, producer="mcp__kospi_investment__get_flow_context"):
    from tools.tool_result_storage import extract_persisted_path, maybe_persist_tool_result

    return extract_persisted_path(maybe_persist_tool_result(
        content, producer, "call_native_probe", threshold=1,
        session_id=session, requester_id=session,
    ))


@pytest.mark.parametrize("character", [False, True])
def test_native_child_reads_parent_evidence_but_unrelated_session_cannot(spillover, character):
    from hermes_state import SessionDB

    db = SessionDB(spillover.parent.parent / "state.db")
    db.create_session("parent", "cli")
    db.create_session("child", "delegate", parent_session_id="parent")
    db.create_session("unrelated", "cli")
    db.close()
    reference = _persist_native("PARENT_RESEARCH_TAIL", "parent")
    args = {"reference": reference}
    if character:
        args["character_offset"] = 1
    result = json.loads(evidence.registry.dispatch(
        "investment_evidence", args, session_id="child", task_id="child",
    ))
    assert result["success"] and "PARENT_RESEARCH_TAIL" in result["content"]
    denied = evidence.registry.dispatch(
        "investment_evidence", args, session_id="unrelated", task_id="unrelated",
    )
    assert not json.loads(denied)["success"] and "PARENT_RESEARCH_TAIL" not in denied


@pytest.mark.parametrize("character", [False, True])
@pytest.mark.parametrize("producer,session", [
    ("mcp__kospi_investment__get_balance", "parent"),
    ("mcp__kospi_investment__get_flow_context", "unrelated"),
])
def test_replaced_source_and_matching_sidecar_are_reauthorized(
    spillover, monkeypatch, character, producer, session,
):
    reference = _persist_native("PERMITTED_RESEARCH", "parent")
    original = evidence._read_pinned_source
    swapped = []

    def replace_before_read(path):
        _persist_native("UNAUTHORIZED_REPLACEMENT", session, producer)
        swapped.append(True)
        return original(path)

    monkeypatch.setattr(evidence, "_read_pinned_source", replace_before_read)
    args = {"reference": reference}
    if character:
        args["character_offset"] = 1
    result = evidence.registry.dispatch(
        "investment_evidence", args, session_id="parent", task_id="parent",
    )
    assert swapped
    assert not json.loads(result)["success"]
    assert "UNAUTHORIZED_REPLACEMENT" not in result


def test_sidecar_claim_of_parent_does_not_grant_lineage(spillover):
    path = spillover / "forged-parent.txt"
    _write_evidence(path, "NOT_AUTHORIZED")
    sidecar = path.with_name(path.name + ".meta.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["parent_session_id"] = "invented-parent"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    result = evidence.registry.dispatch(
        "investment_evidence", {"reference": str(path)},
        session_id="invented-parent", task_id="invented-parent",
    )
    assert not json.loads(result)["success"] and "NOT_AUTHORIZED" not in result


@pytest.mark.parametrize("producer,allowed", [
    ("mcp__kospi_investment__get_event_context", True),
    ("mcp__kospi_investment__get_situation_brief", True),
    ("mcp__kospi_investment__get_fundamental_context", True),
    ("mcp__kospi_investment__get_balance", False),
    ("mcp__kospi_investment__get_dual_account_snapshot", False),
])
def test_aggregate_storage_preserves_research_producer_and_session(spillover, producer, allowed):
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.budget_config import BudgetConfig
    from tools.tool_result_storage import enforce_turn_budget, extract_persisted_path

    messages = [make_tool_result_message(producer, "x" * 12000 + "AGGREGATE_TAIL", "aggregate_call")]
    enforce_turn_budget(messages, config=BudgetConfig(turn_budget=500, preview_size=50),
                        session_id="parent", requester_id="parent")
    reference = extract_persisted_path(messages[0]["content"])
    assert reference and "AGGREGATE_TAIL" not in messages[0]["content"]
    sidecar = json.loads(
        Path(reference + ".meta.json").read_text(encoding="utf-8")
    )
    assert sidecar["tool_name"] == producer and sidecar["session_id"] == "parent"
    result = evidence.registry.dispatch("investment_evidence", {"reference": reference},
                                        session_id="parent", task_id="parent")
    assert json.loads(result)["success"] is allowed
    assert ("AGGREGATE_TAIL" in result) is allowed


@pytest.mark.parametrize("read_only", [True, False, None, "true", 1])
def test_strategy_reader_requires_runtime_read_only_proof(spillover, read_only):
    from tools.tool_result_storage import maybe_persist_tool_result, extract_persisted_path

    # A payload claiming read_only cannot authorize itself.
    content = json.dumps({"structuredContent": {"read_only": True,
        "rows": ["근거 " * 500] * 40, "tail": "STRATEGY_END"}}, ensure_ascii=False)
    preview = maybe_persist_tool_result(
        content, "mcp__kospi_investment__analyze_strategy", "strategy_test",
        threshold=1000, session_id="test-session", requester_id="test-requester",
        tool_arguments={"read_only": read_only},
        investment_evidence_enabled=True,
    )
    reference = extract_persisted_path(preview)
    args = {"reference": reference}
    pages = []
    for _ in range(30):
        page = json.loads(evidence._handle_investment_evidence(args))
        if read_only is not True:
            assert page["success"] is False
            assert page["status"] == "degraded" and page["provider_retry"] is False
            assert "read-only execution provenance" in page["error"]
            return
        assert page["success"] is True, page
        pages.append(page["content"])
        if not page["truncated"]:
            break
        args["character_offset"] = page["next_character_offset"]
    else:
        pytest.fail("strategy evidence never reached the tail")
    assert "investment_evidence(reference=" in preview and "Use the read_file" not in preview
    assert "STRATEGY_END" in "".join(pages)
    assert json.loads("".join(pages))["structuredContent"]["read_only"] is True
    denied = json.loads(evidence._handle_investment_evidence(
        {"reference": reference}, session_id="unrelated-session"))
    assert denied["success"] is False
    Path(reference).write_text(content + "changed", encoding="utf-8")
    assert json.loads(evidence._handle_investment_evidence({"reference": reference}))["success"] is False


def test_legacy_strategy_and_account_producers_remain_denied(spillover):
    from tools.tool_result_storage import maybe_persist_tool_result, extract_persisted_path

    for producer in ("mcp__kospi_investment__analyze_strategy", "mcp__kospi_investment__get_balance"):
        preview = maybe_persist_tool_result(
            "account text" * 2000, producer, "legacy", threshold=1000,
            session_id="test-session", requester_id="test-requester",
        )
        result = json.loads(evidence._handle_investment_evidence({"reference": extract_persisted_path(preview)}))
        assert result["success"] is False


@pytest.mark.parametrize("proof", [None, {"tool_name": "other", "read_only": True},
    {"tool_name": "mcp__kospi_investment__analyze_strategy", "read_only": False},
    {"tool_name": "mcp__kospi_investment__analyze_strategy", "read_only": True}])
def test_aggregate_strategy_storage_uses_execution_proof(spillover, proof):
    from tools.tool_result_storage import enforce_turn_budget, extract_persisted_path
    from tools.budget_config import BudgetConfig

    messages = [{"tool_call_id": "aggregate-strategy", "name": "mcp__kospi_investment__analyze_strategy",
                 "content": json.dumps({"read_only": True, "data": "raw evidence " * 400})}]
    enforce_turn_budget(messages, config=BudgetConfig(turn_budget=1000, preview_size=100),
        session_id="test-session", requester_id="test-requester",
        execution_provenance={"aggregate-strategy": proof} if proof else None)
    reference = extract_persisted_path(messages[0]["content"])
    result = json.loads(evidence._handle_investment_evidence({"reference": reference}))
    authorized = bool(proof and proof["tool_name"] == messages[0]["name"] and proof["read_only"] is True)
    assert result["success"] is authorized, result
    if authorized:
        assert "raw evidence" in result["content"]
        assert json.loads(evidence._handle_investment_evidence(
            {"reference": reference}, session_id="unrelated"))["success"] is False


@pytest.mark.parametrize("sandbox_visible", ["/sandbox/cache/spillover/remote.txt", None])
def test_remote_investment_evidence_keeps_host_reference(spillover, monkeypatch, sandbox_visible):
    from tools import tool_result_storage as storage

    monkeypatch.setattr(storage, "_is_host_side_env", lambda env: False)
    monkeypatch.setattr(storage, "_sandbox_visible_spillover_path", lambda path, env: sandbox_visible)
    preview = storage.maybe_persist_tool_result(
        "raw evidence " * 400, "mcp__kospi_investment__analyze_strategy", "remote",
        env=object(), threshold=1000, session_id="test-session", requester_id="test-requester",
        tool_arguments={"read_only": True}, investment_evidence_enabled=True)
    reference = storage.extract_persisted_path(preview)
    assert Path(reference).is_relative_to(spillover)
    assert "investment_evidence(reference=" in preview
    assert "Use the read_file" not in preview
    assert json.loads(evidence._handle_investment_evidence({"reference": reference}))["success"] is True


def test_failed_host_storage_does_not_offer_unreadable_sandbox_evidence(spillover, monkeypatch):
    from tools import tool_result_storage as storage

    monkeypatch.setattr(storage, "_write_to_spillover", lambda *a, **k: None)
    monkeypatch.setattr(storage, "_is_host_side_env", lambda env: False)
    result = json.loads(storage.maybe_persist_tool_result(
        "raw evidence " * 400, "mcp__kospi_investment__analyze_strategy", "unsaved",
        env=object(), threshold=1000, tool_arguments={"read_only": True},
        investment_evidence_enabled=True))
    assert result["success"] is False and result["status"] == "degraded"
    assert result["unknown"] is True and result["provider_retry"] is False


@pytest.mark.parametrize("producer", ["web_search", "web_extract"])
def test_web_results_use_investment_reader_only_when_enabled(
    spillover, producer,
):
    from tools.tool_result_storage import maybe_persist_tool_result

    content = "PUBLIC_WEB_EVIDENCE\n" * 200
    enabled = maybe_persist_tool_result(
        content, producer, f"{producer}-enabled", threshold=100,
        session_id="test-session", requester_id="test-requester",
        investment_evidence_enabled=True,
    )
    disabled = maybe_persist_tool_result(
        content, producer, f"{producer}-disabled", threshold=100,
        session_id="test-session", requester_id="test-requester",
        investment_evidence_enabled=False,
    )

    assert "investment_evidence(reference=" in enabled
    assert "Use the read_file" not in enabled
    assert "Use the read_file" in disabled


def test_web_evidence_keeps_session_and_digest_boundaries(spillover):
    from tools.tool_result_storage import extract_persisted_path, maybe_persist_tool_result

    preview = maybe_persist_tool_result(
        "WEB_SOURCE_FACT\n" * 200, "web_extract", "web-boundary",
        threshold=100, session_id="test-session", requester_id="test-requester",
        investment_evidence_enabled=True,
    )
    reference = extract_persisted_path(preview)
    assert reference
    allowed = json.loads(evidence._handle_investment_evidence({"reference": reference}))
    denied = json.loads(evidence._handle_investment_evidence(
        {"reference": reference}, session_id="unrelated-session"
    ))
    assert allowed["success"] is True and "WEB_SOURCE_FACT" in allowed["content"]
    assert denied["success"] is False and "WEB_SOURCE_FACT" not in json.dumps(denied)
