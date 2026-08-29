from __future__ import annotations

import subprocess

import pytest


def _make_task(kb, *, assignee: str):
    return kb.Task(
        id="t_spawn_tools",
        title="spawn tools",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def test_default_spawn_pins_assignee_profile_cli_toolsets(monkeypatch, tmp_path):
    """Manual profile assignment should keep that profile's CLI tools.

    Regression guard for dispatcher-spawned workers that boot with
    HERMES_KANBAN_TASK: the worker must not collapse to only kanban lifecycle
    tools when the assigned profile's top-level ``toolsets`` is the default
    composite. The spawned CLI gets an explicit --toolsets pin resolved from
    platform_toolsets.cli; model_tools appends task-scoped kanban tools later.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - clarify
    - code_execution
    - delegation
    - file
    - memory
    - session_search
    - skills
    - terminal
    - web
toolsets:
  - hermes-cli
agent:
  disabled_toolsets: []
""".lstrip(),
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid = kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert pid == 4242
    assert captured["env"]["HERMES_HOME"] == str(profile)
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_spawn_tools"
    assert "--toolsets" in captured["cmd"]
    pinned = captured["cmd"][captured["cmd"].index("--toolsets") + 1].split(",")
    for required in ("terminal", "web", "file", "skills", "code_execution", "delegation"):
        assert required in pinned


def test_default_spawn_model_override_survives_real_cli_parse(monkeypatch, tmp_path):
    """The dispatcher's pre-``chat`` model flag must reach ``args.model``.

    This is an integration contract between Kanban's worker argv builder and
    the real CLI parser. A parser default once erased the explicit override,
    silently sending the worker to its profile default or fallback instead.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4244

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_task(kb, assignee="elias")
    task.model_override = "gpt-5.6-sol"
    kb._default_spawn(task, str(workspace))

    parser, _subparsers, _chat_parser = build_top_level_parser()
    # Profile selection is attached by the outer CLI bootstrap rather than
    # build_top_level_parser(); remove that already-validated prefix and parse
    # the worker flags/subcommand through the real shared parser.
    assert captured["cmd"][1:3] == ["-p", "elias"]
    args = parser.parse_args(captured["cmd"][3:])

    assert args.command == "chat"
    assert args.model == "gpt-5.6-sol"
    assert args.query == "work kanban task t_spawn_tools"


def test_resolve_worker_cli_toolsets_uses_profile_home_not_parent_config(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("platform_toolsets:\n  cli:\n    - kanban\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - terminal
    - web
toolsets:
  - hermes-cli
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    resolved = kb._resolve_worker_cli_toolsets(str(profile))

    assert resolved is not None
    assert "terminal" in resolved
    assert "web" in resolved
    assert "kanban" in resolved  # recovered worker lifecycle surface
    assert resolved != ["kanban"]


def test_browser_grant_without_terminal_gets_constrained_browser_schema(
    monkeypatch, tmp_path
):
    """Browser Use must not collapse a least-privilege QA profile to no browser.

    ``browser_exec`` is intentionally unavailable without terminal because it
    executes host Python.  A profile that grants browser but not terminal must
    fall back to the constrained built-in browser actions instead of losing the
    entire browser capability.
    """
    profile = tmp_path / ".hermes" / "profiles" / "webdesignqa"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        "platform_toolsets:\n  cli: [browser, web, kanban]\n",
        encoding="utf-8",
    )

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools import browser_tool, browser_use_cli
    from tools.registry import invalidate_check_fn_cache

    # Browser Use is the default implementation, while the constrained local
    # backend is also available for a no-terminal session.
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda *_a, **_kw: "agent-browser"
    )
    monkeypatch.setattr(browser_tool, "_chromium_installed", lambda: True)

    token = set_hermes_home_override(str(profile))
    try:
        toolsets = __import__(
            "hermes_cli.kanban_db", fromlist=["_resolve_worker_cli_toolsets"]
        )._resolve_worker_cli_toolsets(str(profile))
        invalidate_check_fn_cache()
        _clear_tool_defs_cache()
        schema = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
    finally:
        reset_hermes_home_override(token)

    names = {item["function"]["name"] for item in schema}
    assert {"browser_navigate", "browser_snapshot"} <= names
    assert "browser_exec" not in names
    assert "terminal" not in names
    assert "execute_code" not in names
    assert not {"read_file", "write_file", "patch"} & names


@pytest.mark.parametrize("workspace_kind", ["scratch", "dir"])
def test_browser_grant_fails_before_spawn_when_no_safe_backend(
    monkeypatch, tmp_path, workspace_kind
):
    """Unavailable browser grants must produce a pre-dispatch capability error."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "webdesignqa"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        "platform_toolsets:\n  cli: [browser, web, kanban]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from model_tools import _clear_tool_defs_cache
    from tools import browser_tool, browser_use_cli
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: None)
    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda *_a, **_kw: (_ for _ in ()).throw(FileNotFoundError("unavailable")),
    )
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_a, **_kw: pytest.fail("worker must not spawn without browser"),
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_task(kb, assignee="webdesignqa")
    task.workspace_kind = workspace_kind

    with pytest.raises(RuntimeError, match="browser.*unavailable|browser.*capability"):
        kb._default_spawn(task, str(workspace))


def test_running_dispatcher_resolves_profile_toolset_reload(monkeypatch, tmp_path):
    """A long-lived gateway dispatcher must read the updated profile grant."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "webdesignqa"
    profile.mkdir(parents=True)
    config_path = profile / "config.yaml"
    config_path.write_text(
        "platform_toolsets:\n  cli: [web, kanban]\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    before = kb._resolve_worker_cli_toolsets(str(profile))
    assert before is not None and "browser" not in before

    config_path.write_text(
        "platform_toolsets:\n  cli: [browser, web, kanban]\n", encoding="utf-8"
    )
    after = kb._resolve_worker_cli_toolsets(str(profile))

    assert after is not None and "browser" in after
