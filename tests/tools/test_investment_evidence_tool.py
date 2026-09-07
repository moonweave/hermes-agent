from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import investment_evidence_tool as evidence


@pytest.fixture
def spillover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cache" / "spillover"
    root.mkdir(parents=True)
    monkeypatch.setattr(evidence, "get_spillover_dir", lambda: root)
    return root


def test_reader_pages_existing_spillover_and_separates_wrapper_freshness(
    spillover: Path,
) -> None:
    path = spillover / "call_sanitized.txt"
    path.write_text("provider_as_of=2026-09-07T01:00:00Z\nMEANINGFUL_FACT\n", encoding="utf-8")

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
    path.write_text("flow=unknown", encoding="utf-8")
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
    preview = maybe_persist_tool_result(content, 'investment_fixture', 'tail_evidence', threshold=1000)
    reference = extract_persisted_path(preview)
    assert reference and 'TAIL_FACT' not in preview
    pages = []
    for offset in range(1, 1802, 400):
        page = json.loads(evidence._handle_investment_evidence({'reference': reference, 'offset': offset, 'limit': 400}))
        assert page['success'] and page['freshness_authority'] == 'provider_payload_as_of'
        pages.append(page['content'])
    assert 'TAIL_FACT: provider_flow_unavailable' in ''.join(pages)


def test_real_single_line_nested_json_is_read_completely(spillover):
    source = json.dumps({"result": json.dumps({
        "rows": ["합성 근거 " + "x" * 900 for _ in range(80)],
        "tail": "TAIL_FACT_SINGLE_LINE_Q91",
    }, ensure_ascii=False)}, ensure_ascii=False)
    path = spillover / "single_line.txt"
    path.write_text(source, encoding="utf-8")
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
    path.write_text("p" * 19973 + "Bearer " + token + " visible_tail", encoding="utf-8")
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
    (spillover / "small.txt").write_text("evidence")
    page = json.loads(evidence._handle_investment_evidence({"reference": "small.txt", "character_offset": offset}))
    assert page["success"] is False
    assert page["status"] == "degraded" and page["unknown"] is True


def test_character_page_rejects_leaf_swap_without_exposing_outside_file(spillover, tmp_path, monkeypatch):
    path = spillover / "race.txt"
    path.write_text("safe")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-confidential-evidence")
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


def test_oversized_character_source_is_degraded_without_partial_success(spillover):
    path = spillover / "large.txt"
    with path.open("wb") as handle:
        handle.truncate(8 * 1024 * 1024 + 1)
    page = json.loads(evidence._handle_investment_evidence({"reference": path.name, "character_offset": 1}))
    assert page["success"] is False
    assert page["status"] == "degraded" and page["unknown"] is True


def test_character_page_fifo_swap_is_opened_nonblocking_and_rejected(spillover, monkeypatch):
    path = spillover / "fifo-race.txt"
    path.write_text("safe")
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
