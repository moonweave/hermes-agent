"""Tests for the capability-gated ``ctx.call_mcp`` plugin surface (#64204).

The gate: ``plugins.entries.<plugin_id>.mcp_allowlist`` — a list of MCP
server names. Absent key = no MCP access (default-deny). Calls to unlisted
servers raise PermissionError naming the config key. All calls route
through the existing tools.mcp_tool handler machinery (mocked here — no
live MCP servers).
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_cli.plugins import PluginContext, PluginManifest


def _make_ctx(plugin_key: str = "my-plugin") -> PluginContext:
    manifest = PluginManifest(name=plugin_key, key=plugin_key)
    manager = MagicMock()
    return PluginContext(manifest, manager)


def _patch_config(monkeypatch, entries: dict) -> None:
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config",
        lambda *a, **k: {"plugins": {"entries": entries}},
    )


def _patch_handler(monkeypatch, response: str, captured: dict | None = None):
    """Replace tools.mcp_tool._make_tool_handler with a transport mock."""
    import tools.mcp_tool as mcp_mod

    def _fake_make_handler(server_name, tool_name, tool_timeout):
        if captured is not None:
            captured["server"] = server_name
            captured["tool"] = tool_name
            captured["timeout"] = tool_timeout

        def _handler(args, **kwargs):
            if captured is not None:
                captured["args"] = args
            return response

        return _handler

    monkeypatch.setattr(mcp_mod, "_make_tool_handler", _fake_make_handler)


# ---------------------------------------------------------------------------
# Default-deny and allowlist enforcement
# ---------------------------------------------------------------------------


def test_default_deny_when_key_absent(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {}})
    ctx = _make_ctx()
    with pytest.raises(PermissionError) as exc:
        ctx.call_mcp("github", "create_issue", {"title": "x"})
    # Error message names the exact config key the operator must set.
    assert "plugins.entries.my-plugin.mcp_allowlist" in str(exc.value)
    assert "github" in str(exc.value)


def test_default_deny_when_plugin_has_no_entry(monkeypatch):
    _patch_config(monkeypatch, {})
    ctx = _make_ctx()
    with pytest.raises(PermissionError):
        ctx.call_mcp("github", "create_issue")


def test_default_deny_when_config_unreadable(monkeypatch):
    import hermes_cli.config as config_mod

    def _boom(*a, **k):
        raise OSError("config torn mid-edit")

    monkeypatch.setattr(config_mod, "load_config", _boom)
    ctx = _make_ctx()
    with pytest.raises(PermissionError):
        ctx.call_mcp("github", "create_issue")


def test_unlisted_server_denied_even_with_other_grants(monkeypatch):
    _patch_config(
        monkeypatch, {"my-plugin": {"mcp_allowlist": ["knowledge_rag"]}}
    )
    ctx = _make_ctx()
    with pytest.raises(PermissionError) as exc:
        ctx.call_mcp("github", "create_issue")
    assert "github" in str(exc.value)


def test_non_list_allowlist_is_denied(monkeypatch):
    """A scalar/'*' value must not grant ambient access."""
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": "*"}})
    ctx = _make_ctx()
    with pytest.raises(PermissionError):
        ctx.call_mcp("github", "create_issue")


def test_denied_call_never_touches_transport(monkeypatch):
    _patch_config(monkeypatch, {})
    called = {}
    _patch_handler(monkeypatch, '{"result": "hi"}', called)
    ctx = _make_ctx()
    with pytest.raises(PermissionError):
        ctx.call_mcp("github", "create_issue")
    assert called == {}


# ---------------------------------------------------------------------------
# Allowed calls route through the existing MCP handler machinery
# ---------------------------------------------------------------------------


def test_allowed_call_routes_through_existing_handler(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}})
    captured = {}
    _patch_handler(monkeypatch, json.dumps({"result": "issue #7 created"}), captured)

    ctx = _make_ctx()
    result = ctx.call_mcp("github", "create_issue", {"title": "bug"})

    assert captured["server"] == "github"
    assert captured["tool"] == "create_issue"
    assert captured["args"] == {"title": "bug"}
    assert result == {"ok": True, "result": "issue #7 created"}


def test_eager_hidden_server_remains_available_to_allowlisted_plugin(monkeypatch):
    """Real eager registration hides tools but keeps direct plugin RPC live."""
    from mcp.types import CallToolResult, TextContent
    from tools import mcp_tool as mcp_mod
    from tools.registry import registry

    server_config = {
        "command": "fixture-team-mcp",
        "lazy": False,
        "tools": {
            "include": [],
            "resources": False,
            "prompts": False,
        },
    }
    server = mcp_mod.MCPServerTask("team")
    monkeypatch.setattr(
        mcp_mod.MCPServerTask, "_watch_stdio_children", lambda _self: None
    )
    server._tools = [
        SimpleNamespace(
            name="prepare_consultation",
            description="prepare",
            inputSchema={"type": "object"},
            annotations=None,
        )
    ]
    server.session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="prepared")]
            )
        )
    )

    async def discover(name, config):
        assert name == "team"
        assert config["lazy"] is False
        with mcp_mod._lock:
            mcp_mod._servers[name] = server
        return mcp_mod._register_server_tools(name, server, config)

    monkeypatch.setattr(mcp_mod, "_discover_and_register_server", discover)
    assert mcp_mod.register_mcp_servers({"team": server_config}) == []
    assert not any(name.startswith("mcp__team__") for name in registry.get_all_tool_names())

    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["team"]}})
    result = _make_ctx().call_mcp("team", "prepare_consultation", {"scope": "omitted"})

    assert result == {"ok": True, "result": "prepared"}
    server.session.call_tool.assert_awaited_once_with(
        "prepare_consultation", arguments={"scope": "omitted"}
    )

    with mcp_mod._lock:
        mcp_mod._servers.pop("team", None)


def test_error_result_maps_to_ok_false(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}})
    _patch_handler(monkeypatch, json.dumps({"error": "MCP server 'github' is not connected"}))

    ctx = _make_ctx()
    result = ctx.call_mcp("github", "create_issue")
    assert result["ok"] is False
    assert "not connected" in result["error"]


def test_structured_content_passthrough(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["rag"]}})
    _patch_handler(
        monkeypatch,
        json.dumps({"result": "text part", "structuredContent": {"hits": 3}}),
    )

    ctx = _make_ctx()
    result = ctx.call_mcp("rag", "query")
    assert result["ok"] is True
    assert result["result"] == "text part"
    assert result["structuredContent"] == {"hits": 3}


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


def test_timeout_forwarded_to_handler(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["slow"]}})
    captured = {}
    _patch_handler(monkeypatch, '{"result": ""}', captured)

    ctx = _make_ctx()
    ctx.call_mcp("slow", "long_op", timeout=120)
    assert captured["timeout"] == 120.0


def test_timeout_defaults_and_bounds(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["s"]}})
    captured = {}
    _patch_handler(monkeypatch, '{"result": ""}', captured)
    ctx = _make_ctx()

    ctx.call_mcp("s", "t")
    assert captured["timeout"] == 30.0

    ctx.call_mcp("s", "t", timeout=0)  # below floor → clamped to 1s
    assert captured["timeout"] == 1.0

    ctx.call_mcp("s", "t", timeout=99999)  # above ceiling → clamped to 600s
    assert captured["timeout"] == 600.0

    ctx.call_mcp("s", "t", timeout="nonsense")  # unparseable → default
    assert captured["timeout"] == 30.0


# ---------------------------------------------------------------------------
# Result size cap
# ---------------------------------------------------------------------------


def test_oversized_result_is_truncated(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["big"]}})
    huge = "x" * (PluginContext._MCP_RESULT_CHAR_CAP + 5000)
    _patch_handler(monkeypatch, huge)

    ctx = _make_ctx()
    result = ctx.call_mcp("big", "dump")
    assert result["ok"] is True
    assert result["truncated"] is True
    assert len(result["result"]) <= PluginContext._MCP_RESULT_CHAR_CAP + 20
    assert result["result"].endswith("… [truncated]")
