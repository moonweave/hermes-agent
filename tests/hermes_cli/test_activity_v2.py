"""Regression contracts for privacy-safe v2 activity readers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli.sessions_cmd import _sessions_activity_v2


WORKSPACE = "/private/repo"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class _Sessions:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def list_sessions_rich(self, **kwargs):
        self.calls.append(kwargs)
        rows = self.rows
        if kwargs.get("source"):
            rows = [row for row in rows if row.get("source") == kwargs["source"]]
        if kwargs.get("exclude_sources"):
            rows = [row for row in rows if row.get("source") not in kwargs["exclude_sources"]]
        return sorted(rows, key=lambda row: -(row.get("last_active") or 0))[
            : kwargs["limit"]
        ]


def _session(
    *, source="desktop", profile="reviewer", workspace=WORKSPACE, git_repo_root=None, active
):
    return {
        "id": "private-session-id",
        "source": source,
        "profile_name": profile,
        "cwd": workspace,
        "git_repo_root": git_repo_root,
        "title": "private title",
        "preview": "private prompt",
        "last_active": active,
        "last_activity_description": "executing tool: private tool name",
        "ended_at": None,
    }


def test_sessions_activity_v2_aggregates_current_interactive_work_without_identifiers(
    capsys, monkeypatch
):
    now = int(time.time())
    monkeypatch.setattr("hermes_state.workspace_key", lambda row: row.get("cwd"))
    db = _Sessions(
        [
            _session(active=now - 5),
            _session(active=now - 8),
            _session(source="kanban", active=now - 2),
            _session(source="tool", active=now - 1),
            _session(active=now - 5000),
            _session(profile=None, active=now - 4),
            _session(workspace="", active=now - 3),
        ]
    )

    assert _sessions_activity_v2(
        db, argparse.Namespace(source=None, json=True)
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["contract_version"] == "hermes-sessions-activity-v2"
    assert payload["aggregates"] == [
        {
            "activity_ref": "activity_"
            + hashlib.sha256(
                f"reviewer\x1f{hashlib.sha256(WORKSPACE.encode()).hexdigest()}".encode()
            ).hexdigest()[:16],
            "profile": "reviewer",
            "workspace_digest": hashlib.sha256(WORKSPACE.encode()).hexdigest(),
            "recent_session_count": 2,
            "evidence_observed_at": now - 5,
            "evidence_expires_at": now + 115,
            "phase_code": None,
            "phase_observed_at": None,
            "phase_expires_at": None,
        }
    ]
    assert payload["coverage"] == {
        "complete": True,
        "total_aggregates": 1,
        "returned_aggregates": 1,
        "scan_truncated": False,
    }
    assert payload["unresolved_profile"] == {"recent_session_count": 1}
    assert payload["unresolved_workspace"] == {"recent_session_count": 1}
    rendered = json.dumps(payload)
    for private in ("private-session-id", "private title", "private prompt", WORKSPACE):
        assert private not in rendered
    assert db.calls[0]["exclude_sources"] == [
        "tool",
        "dispatcher",
        "delegate",
        "subagent",
        "kanban",
        "kanban-worker",
    ]


def test_sessions_activity_v2_normalizes_git_roots_and_symlinked_workspaces(
    capsys, tmp_path
):
    now = int(time.time())
    repo = tmp_path / "repo"
    nested = repo / "src"
    nested.mkdir(parents=True)
    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)
    db = _Sessions(
        [
            _session(active=now - 5, workspace=str(nested), git_repo_root=str(repo)),
            _session(
                active=now - 6,
                workspace=str(alias / "src"),
                git_repo_root=str(alias),
            ),
        ]
    )

    assert _sessions_activity_v2(db, argparse.Namespace(source=None, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)

    normalized = os.path.normcase(os.path.normpath(os.path.realpath(repo)))
    assert payload["aggregates"] == [
        {
            "activity_ref": "activity_"
            + hashlib.sha256(
                f"reviewer\x1f{hashlib.sha256(normalized.encode()).hexdigest()}".encode()
            ).hexdigest()[:16],
            "profile": "reviewer",
            "workspace_digest": hashlib.sha256(normalized.encode()).hexdigest(),
            "recent_session_count": 2,
            "evidence_observed_at": now - 5,
            "evidence_expires_at": now + 115,
            "phase_code": None,
            "phase_observed_at": None,
            "phase_expires_at": None,
        }
    ]
    rendered = json.dumps(payload)
    assert str(repo) not in rendered
    assert str(alias) not in rendered


def test_sessions_activity_v2_does_not_claim_truncation_after_window_boundary(
    capsys, monkeypatch
):
    now = int(time.time())
    monkeypatch.setattr("hermes_state.workspace_key", lambda row: row.get("cwd"))
    db = _Sessions([_session(active=now - 500 - index) for index in range(220)])

    assert _sessions_activity_v2(db, argparse.Namespace(source=None, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["aggregates"] == []
    assert payload["coverage"]["complete"] is True
    assert payload["coverage"]["scan_truncated"] is False


def test_board_activity_v2_returns_run_scoped_evidence_streams_with_v1_events(kanban_home, monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="private task title",
            body="private task body",
            assignee="reviewer",
        )
        assert kb.claim_task(conn, task_id, claimer="private-host:private-lock")
        [run_id] = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        statements = []
        conn.set_trace_callback(statements.append)
        activity = kb.board_activity_v2(conn, limit=20)
        conn.set_trace_callback(None)

    assert activity["contract_version"] == "hermes-kanban-activity-v2"
    assert [event["kind"] for event in activity["events"]] == ["created", "claimed"]
    assert activity["work_streams"] == [
        {
            "stream_ref": kb._dashboard_activity_ref(
                "stream", kb._activity_ref_salt(conn), run_id
            ),
            "profile": "reviewer",
            "evidence_state": "unverified",
            "evidence_observed_at": now,
            "evidence_expires_at": now + kb._ACTIVITY_EVIDENCE_SHORT_LEASE_SECONDS,
            "phase_code": None,
            "phase_observed_at": None,
            "phase_expires_at": None,
        }
    ]
    assert activity["coverage"] == {
        "complete": True,
        "total_streams": 1,
        "returned_streams": 1,
        "scan_truncated": False,
    }
    assert "BEGIN" in statements
    assert "ROLLBACK" in statements
    assert activity["work_streams"][0]["stream_ref"] != activity["events"][-1]["work_ref"]
    rendered = json.dumps(activity)
    for private in (task_id, "private task title", "private task body", "private-host:private-lock"):
        assert private not in rendered

    cli_payload = json.loads(kc.run_slash("activity-v2 --json --limit 20"))
    assert cli_payload["contract_version"] == "hermes-kanban-activity-v2"
    assert cli_payload["work_streams"] == activity["work_streams"]


def test_board_activity_v2_evidence_states_follow_heartbeat_claim_and_pid(kanban_home, monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    monkeypatch.setattr(kb, "_claimer_id", lambda: "local:operator")
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: int(pid) in {11, 12})
    with kb.connect() as conn:
        task_ids = [
            kb.create_task(conn, title=state, assignee=state)
            for state in ("live", "pid", "stale", "unverified")
        ]
        for task_id in task_ids:
            assert kb.claim_task(conn, task_id, claimer="local:private-lock")
        runs = {
            row["profile"]: row["id"]
            for row in conn.execute(
                "SELECT profile, id FROM task_runs WHERE status = 'running'"
            )
        }
        conn.execute(
            "UPDATE task_runs SET worker_pid = ?, last_heartbeat_at = ?, claim_expires = ? WHERE id = ?",
            (11, now - 5, now + 900, runs["live"]),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ?, last_heartbeat_at = NULL, claim_expires = ? WHERE id = ?",
            (12, now + 900, runs["pid"]),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ?, last_heartbeat_at = ?, claim_expires = ? WHERE id = ?",
            (13, now - kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS - 1, now + 900, runs["stale"]),
        )
        activity = kb.board_activity_v2(conn, limit=20)

    states = {entry["profile"]: entry["evidence_state"] for entry in activity["work_streams"]}
    assert states == {
        "live": "live_verified",
        "pid": "pid_alive",
        "stale": "stale",
        "unverified": "unverified",
    }
    intervals = {
        entry["profile"]: (entry["evidence_observed_at"], entry["evidence_expires_at"])
        for entry in activity["work_streams"]
    }
    for profile in ("stale", "unverified", "live", "pid"):
        observed_at, expires_at = intervals[profile]
        assert isinstance(observed_at, int)
        assert isinstance(expires_at, int)
        assert expires_at > observed_at
    assert intervals["live"][1] == now + kb._ACTIVITY_EVIDENCE_SHORT_LEASE_SECONDS
