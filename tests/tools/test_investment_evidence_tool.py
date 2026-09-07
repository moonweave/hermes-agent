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
