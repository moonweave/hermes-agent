"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from types import SimpleNamespace
from typing import Any, cast

from tools.delegate_tool import (
    _build_child_agent,
    _emit_parent_console,
    _strip_blocked_tools,
)


class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_requested_toolsets_intersected_with_parent(self):
        """LLM requests toolsets parent doesn't have — extras are dropped."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file"])

        # Simulate the intersection logic from _build_child_agent
        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "file", "web", "browser", "rl"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["file", "terminal"]
        assert "web" not in scoped
        assert "browser" not in scoped
        assert "rl" not in scoped


    def test_strip_blocked_removes_delegation(self):
        """Blocked toolsets (delegation, clarify, etc.) are always removed."""
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "terminal" in child

    def test_empty_intersection_yields_empty_toolsets(self):
        """If parent has no overlap with requested, child gets nothing extra."""
        parent = SimpleNamespace(enabled_toolsets=["terminal"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["web", "browser"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert scoped == []

    def test_explicit_empty_toolsets_reach_real_constructor_as_empty(self, monkeypatch):
        """An explicit empty list is a deny-all request, never inheritance."""
        captured = {}
        parent = SimpleNamespace(
            enabled_toolsets=["terminal", "file", "mcp-market"],
            valid_tool_names={"terminal", "read_file", "mcp_market_snapshot"},
            disabled_toolsets=[],
            model="test-model",
            provider="test-provider",
            base_url="https://example.invalid",
            api_key="test",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            reasoning_config=None,
            prefill_messages=None,
            _fallback_chain=None,
            _session_db=None,
            session_id="parent",
        )

        class Constructed:
            session_id = "child"
            enabled_toolsets: list[str] = []
            valid_tool_names = set()
            tools = []

        def fake_agent(**kwargs):
            captured.update(kwargs)
            child = Constructed()
            child.enabled_toolsets = kwargs["enabled_toolsets"]
            return child

        monkeypatch.setattr("run_agent.AIAgent", fake_agent)
        monkeypatch.setattr(
            "tools.delegate_tool._build_child_system_prompt", lambda *_a, **_k: "prompt"
        )
        monkeypatch.setattr(
            "tools.delegate_tool._load_config", lambda: {}
        )

        child = _build_child_agent(
            task_index=0,
            goal="strict tool-less",
            context=None,
            toolsets=[],
            model=None,
            max_iterations=1,
            task_count=1,
            parent_agent=parent,
            role="leaf",
        )

        assert captured["enabled_toolsets"] == []
        constructed = cast(Any, child)
        assert constructed.enabled_toolsets == []
        assert constructed.valid_tool_names == set()
        assert constructed.tools == []

    def test_explicit_empty_toolsets_build_real_aiagent_without_tools(self):
        """Exercise the production builder and AIAgent initialization path."""
        parent = SimpleNamespace(
            enabled_toolsets=["terminal", "file", "mcp-market"],
            valid_tool_names={"terminal", "read_file", "mcp_market_snapshot"},
            disabled_toolsets=[],
            model="test/model",
            provider="openrouter",
            base_url="https://example.invalid",
            api_key="test",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            reasoning_config=None,
            prefill_messages=None,
            _fallback_chain=None,
            _session_db=None,
            session_id="parent-real-construction",
        )

        child = _build_child_agent(
            task_index=0,
            goal="strict tool-less real construction",
            context=None,
            toolsets=[],
            model=None,
            max_iterations=1,
            task_count=1,
            parent_agent=parent,
            role="leaf",
        )
        try:
            constructed = cast(Any, child)
            assert constructed.enabled_toolsets == []
            assert constructed.valid_tool_names == set()
            assert constructed.tools == []
            assert constructed._delegate_role == "leaf"
            assert constructed._delegate_depth == 1
        finally:
            child.close()


class TestEmitParentConsole:
    """Progress lines (e.g. ``✓ [N/M] …``) must route through the parent's
    configured ``_safe_print`` in headless stdio hosts (ACP, gateway) so
    they don't land on stdout and corrupt JSON-RPC frames. Regression for a
    bug where delegate_task completion lines pushed to stdout caused
    ``Failed to parse JSON message: ✓ [3/3] …`` errors in the ACP adapter."""

    def test_routes_through_parent_safe_print_when_available(self, capsys):
        captured_lines = []
        parent = SimpleNamespace(_safe_print=lambda line: captured_lines.append(line))

        _emit_parent_console(parent, "  ✓ [1/3] Research done  (11.55s)")

        assert captured_lines == ["  ✓ [1/3] Research done  (11.55s)"]
        stdout_stderr = capsys.readouterr()
        assert stdout_stderr.out == ""
        assert stdout_stderr.err == ""


    def test_non_callable_safe_print_is_ignored(self, capsys):
        """Defensive: if _safe_print is set but not callable, fall back."""
        parent = SimpleNamespace(_safe_print="not-a-function")
        _emit_parent_console(parent, "  ✓ [3/3] non-callable guard")
        captured = capsys.readouterr()
        assert "non-callable guard" in captured.out
