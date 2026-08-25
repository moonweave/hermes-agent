"""Behavior contracts for run/session-scoped live captions and activity v3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.live_activity import sanitize_live_caption
from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli.sessions_cmd import _sessions_activity_v3
from hermes_state import SessionDB


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

    def list_sessions_rich(self, **kwargs):
        excluded = set(kwargs.get("exclude_sources") or [])
        return [row for row in self.rows if row.get("source") not in excluded]


def _session(
    *, session_id, active, caption=None, caption_at=None, source="desktop",
    profile="reviewer", cwd="/private/repo",
):
    return {
        "id": session_id,
        "source": source,
        "profile_name": profile,
        "cwd": cwd,
        "git_repo_root": None,
        "last_active": active,
        "last_activity_description": "executing tool",
        "live_caption": caption,
        "live_caption_observed_at": caption_at,
        "live_caption_expires_at": (caption_at + 90) if caption_at else None,
        "runtime_phase_code": "using_tool",
        "phase_observed_at": active,
        "phase_expires_at": active + 45,
        "ended_at": None,
    }


def test_live_caption_sanitizer_keeps_status_but_removes_private_material():
    fake_secret = "sk" + "-" + ("a" * 30)
    raw = (
        "[진행 상황](https://user:pass@example.com/work?token=secret#part) "
        "`/Users/person/private/repo/file.py`를 확인하고 있어요.\n"
        "설정은 /etc/private-service/config.toml 에서 읽었어요.\n"
        f"```sh\necho {fake_secret}\n```"
    )

    rendered = sanitize_live_caption(raw)

    assert rendered == "진행 상황 `[로컬 경로]`를 확인하고 있어요. 설정은 [로컬 경로] 에서 읽었어요."
    assert "pass" not in rendered
    assert fake_secret not in rendered
    assert "/Users/" not in rendered


def test_session_caption_is_owner_scoped_and_export_omits_ephemeral_fields(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("open", source="desktop")
    db.create_session("ended", source="desktop")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (100.0, "ended"))

    assert db.publish_live_caption("open", "현재 구조를 확인하고 있어요.", observed_at=200.0)
    assert not db.publish_live_caption("ended", "다시 살아나면 안 돼요.", observed_at=201.0)
    assert db.publish_runtime_phase("open", "using_tool", observed_at=202.0)
    assert db.publish_live_caption("open", "더 최신 문장", observed_at=210.0)
    assert not db.publish_live_caption("open", "늦게 도착한 이전 문장", observed_at=205.0)

    row = db.get_session("open")
    assert row["live_caption"] == "더 최신 문장"
    assert row["runtime_phase_code"] == "using_tool"
    exported = db.export_session("open")
    for key in (
        "live_caption", "live_caption_observed_at", "live_caption_expires_at",
        "runtime_phase_code", "phase_observed_at", "phase_expires_at",
    ):
        assert key not in exported
    db.end_session("open", end_reason="completed")
    ended = db.get_session("open")
    for key in (
        "live_caption", "live_caption_observed_at", "live_caption_expires_at",
        "runtime_phase_code", "phase_observed_at", "phase_expires_at",
    ):
        assert ended[key] is None
    assert not db.publish_live_caption("open", "종료 뒤 늦은 문장", observed_at=220.0)
    db.close()


def test_board_activity_v3_adds_caption_without_changing_v2(kanban_home, monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    monkeypatch.setattr(kb, "_claimer_id", lambda: "local:claim")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Live Work Caption v0.3 브라우저 연결 검증",
            assignee="reviewer",
        )
        assert kb.claim_task(conn, task_id, claimer="local:claim")
        run_id = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()[0]
        conn.execute(
            """UPDATE task_runs
               SET worker_pid = ?, last_heartbeat_at = ?, claim_expires = ?
               WHERE id = ?""",
            (123, now, now + 120, run_id),
        )
        assert kb.publish_run_caption(
            conn, task_id=task_id, run_id=run_id,
            caption="구현 경계를 확인하고 있어요.", observed_at=now,
        )
        assert kb.publish_run_phase(
            conn, task_id=task_id, run_id=run_id,
            phase_code="using_tool", observed_at=now,
        )
        v2 = kb.board_activity_v2(conn, limit=20)
        v3 = kb.board_activity_v3(conn, limit=20)

    assert "live_caption" not in v2["work_streams"][0]
    assert v2["contract_version"] == "hermes-kanban-activity-v2"
    stream = v3["work_streams"][0]
    assert v3["contract_version"] == "hermes-kanban-activity-v3"
    assert stream["live_caption"] == {
        "text": "구현 경계를 확인하고 있어요.",
        "observed_at": now,
        "expires_at": now + 90,
        "provenance": "agent_commentary",
    }
    assert stream["phase_code"] == "using_tool"
    assert stream["phase_expires_at"] == now + 45
    assert stream["work_summary"] == "Live Work Caption v0.3 브라우저 연결 검증"
    assert "work_summary" not in v2["work_streams"][0]

    cli_payload = json.loads(kc.run_slash("activity-v3 --json --limit 20"))
    assert cli_payload["contract_version"] == "hermes-kanban-activity-v3"


def test_operator_inbox_separates_owner_actions_agent_repairs_and_recent_results(
    kanban_home, monkeypatch
):
    now = 1_800_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    with kb.connect() as conn:
        owner_task = kb.create_task(
            conn, title="Open authenticated Research Grove window", assignee="webdesignqa"
        )
        assert kb.block_task(
            conn,
            owner_task,
            kind="needs_input",
            reason=(
                "Open an already authenticated browser window. "
                "Do not read /Users/person/private.env or https://user:pass@example.test/?token=secret"
            ),
        )
        repair_task = kb.create_task(
            conn, title="Repair failed browser assertions", assignee="builder"
        )
        assert kb.block_task(
            conn,
            repair_task,
            kind="needs_input",
            reason="needs_work: fix the deterministic browser assertion",
        )
        obsolete_task = kb.create_task(
            conn, title="Obsolete browser audit", assignee="webdesignqa"
        )
        assert kb.block_task(
            conn, obsolete_task, kind="capability", reason="Browser unavailable"
        )
        kb.create_task(
            conn,
            title="Replacement browser audit",
            assignee="webdesignqa",
            supersedes=obsolete_task,
        )
        completed_task = kb.create_task(
            conn, title="Publish verified release", assignee="vcm"
        )
        assert kb.complete_task(
            conn, completed_task, summary="Release checks passed without deployment"
        )

        payload = kb.board_operator_inbox(conn, limit=20)

    assert payload["contract_version"] == "hermes-kanban-operator-inbox-v1"
    assert [(item["kind"], item["title"]) for item in payload["items"]] == [
        ("needs_user", "Open authenticated Research Grove window"),
        ("agent_action", "Repair failed browser assertions"),
        ("finished", "Publish verified release"),
    ]
    owner = next(item for item in payload["items"] if item["kind"] == "needs_user")
    assert owner["summary"].startswith("Open an already authenticated browser window")
    assert all(len(item["title"]) <= 96 for item in payload["items"])
    assert all(item["summary"] is None or len(item["summary"]) <= 240 for item in payload["items"])
    rendered = json.dumps(payload)
    for private in (
        owner_task,
        repair_task,
        obsolete_task,
        completed_task,
        "/Users/person/private.env",
        "user:pass",
        "token=secret",
    ):
        assert private not in rendered

    cli_payload = json.loads(kc.run_slash("operator-inbox --json --limit 20"))
    assert cli_payload["contract_version"] == "hermes-kanban-operator-inbox-v1"


def test_run_caption_cannot_overwrite_sibling_or_resurrect_after_end(kanban_home):
    with kb.connect() as conn:
        first = kb.create_task(conn, title="first", assignee="reviewer")
        second = kb.create_task(conn, title="second", assignee="reviewer")
        assert kb.claim_task(conn, first, claimer="local:first")
        assert kb.claim_task(conn, second, claimer="local:second")
        first_run = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (first,)).fetchone()[0]
        second_run = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (second,)).fetchone()[0]
        assert kb.publish_run_caption(conn, task_id=first, run_id=first_run, caption="첫 작업", observed_at=10.0)
        assert kb.publish_run_caption(conn, task_id=second, run_id=second_run, caption="둘째 작업", observed_at=11.0)
        assert kb.complete_task(conn, first, summary="done", expected_run_id=first_run)
        assert not kb.publish_run_caption(conn, task_id=first, run_id=first_run, caption="늦은 문장", observed_at=12.0)
        first_row = conn.execute("SELECT live_caption FROM task_runs WHERE id = ?", (first_run,)).fetchone()
        second_row = conn.execute("SELECT live_caption FROM task_runs WHERE id = ?", (second_run,)).fetchone()

    assert first_row[0] is None
    assert second_row[0] == "둘째 작업"


def test_sessions_activity_v3_uses_latest_open_caption_and_keeps_v2_private(capsys, monkeypatch):
    now = int(time.time())
    monkeypatch.setattr("hermes_state.workspace_key", lambda row: row.get("cwd"))
    db = _Sessions([
        _session(session_id="older", active=now - 8, caption="이전 문장", caption_at=now - 8),
        _session(session_id="newer", active=now - 3, caption="최신 문장", caption_at=now - 3),
        _session(session_id="worker", active=now - 1, caption="중복되면 안 됨", caption_at=now - 1, source="kanban-worker"),
    ])

    assert _sessions_activity_v3(db, argparse.Namespace(source=None, json=True, window=120)) == 0
    payload = json.loads(capsys.readouterr().out)
    aggregate = payload["aggregates"][0]

    assert payload["contract_version"] == "hermes-sessions-activity-v3"
    assert aggregate["recent_session_count"] == 2
    assert aggregate["live_caption"]["text"] == "최신 문장"
    assert "older" not in json.dumps(payload)
    assert "newer" not in json.dumps(payload)
    assert "worker" not in json.dumps(payload)


def test_discord_default_profile_without_workspace_projects_to_hq(capsys):
    now = int(time.time())
    db = _Sessions([
        _session(
            session_id="operator",
            active=now - 2,
            caption="현재 요청의 실행 경계를 확인하고 있어요.",
            caption_at=now - 2,
            source="discord",
            profile="default",
            cwd=None,
        ),
    ])

    assert _sessions_activity_v3(db, argparse.Namespace(source=None, json=True, window=120)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["unresolved_workspace"]["recent_session_count"] == 0
    assert payload["aggregates"] == [{
        "activity_ref": payload["aggregates"][0]["activity_ref"],
        "profile": "default",
        "workspace_digest": None,
        "context_scope": "hq",
        "recent_session_count": 1,
        "evidence_observed_at": now - 2,
        "evidence_expires_at": now + 118,
        "live_caption": {
            "text": "현재 요청의 실행 경계를 확인하고 있어요.",
            "observed_at": now - 2,
            "expires_at": now + 88,
            "provenance": "agent_commentary",
        },
        "phase_code": "using_tool",
        "phase_observed_at": now - 2,
        "phase_expires_at": now + 43,
    }]


def test_discord_session_reset_continuation_projects_current_activity(tmp_path, capsys):
    now = int(time.time())
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "discord-root",
        source="discord",
        profile_name="default",
    )
    db.end_session("discord-root", end_reason="session_reset")
    db.create_session(
        "discord-current",
        source="discord",
        profile_name="default",
        parent_session_id="discord-root",
    )
    db.touch_session_activity(
        "discord-current",
        ts=now - 2,
        description="waiting for provider response",
        provenance="api_call",
    )
    assert db.publish_live_caption(
        "discord-current",
        "현재 요청의 다음 응답을 기다리고 있어요.",
        observed_at=now - 2,
    )
    assert db.publish_runtime_phase(
        "discord-current",
        "waiting",
        observed_at=now - 2,
    )

    assert _sessions_activity_v3(
        db,
        argparse.Namespace(source=None, json=True, window=120),
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    db.close()

    assert payload["coverage"]["complete"] is True
    assert len(payload["aggregates"]) == 1
    aggregate = payload["aggregates"][0]
    assert aggregate["profile"] == "default"
    assert aggregate["context_scope"] == "hq"
    assert aggregate["recent_session_count"] == 1
    assert aggregate["phase_code"] == "waiting"
    assert aggregate["live_caption"]["text"] == "현재 요청의 다음 응답을 기다리고 있어요."
    assert "discord-root" not in json.dumps(payload)
    assert "discord-current" not in json.dumps(payload)


def test_default_gateway_profile_is_persisted_as_observed_identity(monkeypatch):
    import run_agent
    from run_agent import AIAgent

    captured = {}

    class _DB:
        def create_session(self, **kwargs):
            captured.update(kwargs)

    agent = SimpleNamespace(
        _persist_disabled=False,
        _session_db_created=False,
        _session_db=_DB(),
        platform="discord",
        session_id="gateway-session",
        model="model",
        _session_init_model_config=None,
        _cached_system_prompt=None,
        _parent_session_id=None,
    )
    monkeypatch.setattr(run_agent, "_session_source_for_agent", lambda _platform: "discord")
    monkeypatch.setattr(run_agent, "_launch_cwd_for_session", lambda _source: None)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")

    AIAgent._ensure_db_session(agent)

    assert captured["profile_name"] == "default"


def test_streaming_and_non_streaming_share_one_caption_boundary():
    from run_agent import AIAgent

    class _DB:
        def __init__(self):
            self.captions = []

        def publish_live_caption(self, session_id, text, *, observed_at):
            self.captions.append((session_id, text))
            return True

    db = _DB()
    delivered = []
    agent = SimpleNamespace(
        show_commentary=True,
        session_id="session-1",
        _session_db=db,
        interim_assistant_callback=lambda text, **_kw: delivered.append(text),
        _delivered_interim_texts=set(),
        _live_caption_last_text=None,
        _strip_think_blocks=lambda text: text,
        _normalize_interim_visible_text=lambda text: " ".join(text.split()),
        _interim_text_was_delivered=lambda text: AIAgent._interim_text_was_delivered(agent, text),
        _record_delivered_interim_text=lambda text: AIAgent._record_delivered_interim_text(agent, text),
        _publish_live_caption=lambda text: AIAgent._publish_live_caption(agent, text),
        _extract_codex_interim_visible_parts=lambda msg: ["작업 경계를 확인하고 있어요."],
        _interim_assistant_visible_text=lambda msg: msg.get("content", ""),
        _interim_content_was_streamed=lambda _text: False,
        _current_turn_id="turn",
        _api_call_count=1,
        model="model",
        provider="codex",
        platform="cli",
    )

    AIAgent._fire_streamed_codex_commentary(agent, "작업 경계를 확인하고 있어요.")
    AIAgent._emit_interim_assistant_message(
        agent,
        {"role": "assistant", "content": "작업 경계를 확인하고 있어요."},
    )

    agent.show_commentary = False
    assert not AIAgent._publish_live_caption(agent, "화면 설정이 끈 문장")

    assert db.captions == [("session-1", "작업 경계를 확인하고 있어요.")]
    assert delivered == ["작업 경계를 확인하고 있어요."]
