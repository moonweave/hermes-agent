"""Runtime tests for tool-call loop guardrails."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert "repeated_exact_failure_warning" in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert progress == []
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_sequential_after_call_appends_guidance_to_tool_result_without_extra_messages():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "Tool loop warning" in messages[0]["content"]
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = getattr(agent, "_tool_guardrails")
    guardrails.after_call(
        "terminal",
        {"command": "bad-1"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    guardrails.after_call(
        "terminal",
        {"command": "bad-2"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    tc = _mock_tool_call("terminal", json.dumps({"command": "bad-3"}), "c-recover")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 1})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    content = messages[0]["content"]
    assert "same_tool_failure_warning" in content
    assert "Do not switch to text-only replies" in content
    assert "keep using tools" in content
    assert "pwd && ls -la" in content
    assert "absolute path" in content
    assert "different tool" in content


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [("c-allow", "web_search", allowed_args)]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [("tool.started", "web_search", allowed_args, {})]
    assert len(completed_events) == 1
    assert completed_events[0][1] == "web_search"


def test_relay_rewrite_precedes_sequential_policy_approval_checkpoint_and_dispatch():
    agent = _make_agent("write_file")
    original_args = {"path": "/original/path", "content": "old"}
    final_args = {"path": "/approved/path", "content": "new"}
    tc = _mock_tool_call("write_file", json.dumps(original_args), "c-rewrite")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    observed = {
        "plugin": [],
        "guardrail": [],
        "approval": [],
        "checkpoint": [],
        "start": [],
        "dispatch": [],
    }

    original_before_call = agent._tool_guardrails.before_call

    def observe_guardrail(name, args):
        observed["guardrail"].append((name, dict(args)))
        return original_before_call(name, args)

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(final_args)), dict(final_args)

    def observe_plugin(name, args, **kwargs):
        del kwargs
        observed["plugin"].append((name, dict(args)))
        return (None, None)

    def observe_approval(name, args):
        observed["approval"].append((name, dict(args)))
        return None

    def dispatch(name, args, task_id, **kwargs):
        del task_id, kwargs
        observed["dispatch"].append((name, dict(args)))
        return json.dumps({"ok": True})

    agent._checkpoint_mgr = SimpleNamespace(
        enabled=True,
        get_working_dir_for_path=lambda path: path,
        ensure_checkpoint=lambda path, reason: observed["checkpoint"].append(
            (path, reason)
        ),
    )
    agent.tool_start_callback = lambda _call_id, name, args: observed["start"].append(
        (name, dict(args))
    )

    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            side_effect=observe_plugin,
        ),
        patch.object(agent._tool_guardrails, "before_call", side_effect=observe_guardrail),
        patch(
            "acp_adapter.edit_approval.maybe_require_edit_approval",
            side_effect=observe_approval,
        ),
        patch("model_tools.registry.dispatch", side_effect=dispatch),
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    expected = [("write_file", final_args)]
    assert observed["plugin"] == expected
    assert observed["guardrail"] == expected
    assert observed["approval"] == expected
    assert observed["start"] == expected
    assert observed["dispatch"] == expected
    assert observed["checkpoint"] == [
        ("/approved/path", "before write_file")
    ]


def test_relay_rewrite_is_guarded_before_dispatch_in_concurrent_path():
    agent = _make_agent("web_search", config=_hard_stop_config())
    original_args = {"query": "original"}
    blocked_args = {"query": "blocked"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    tc = _mock_tool_call("web_search", json.dumps(original_args), "c-rewrite-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    starts = []

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(blocked_args)), dict(blocked_args)

    agent.tool_start_callback = lambda *args: starts.append(args)
    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    dispatch.assert_not_called()
    assert starts == []
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            return_value=("plugin policy", None),
        ),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    assert result["final_response"] == "done"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("repeated_exact_failure_warning" in content for content in tool_contents)




def test_guardrail_halt_emits_final_response_through_stream_delta_callback():
    """Regression for #30770: when the guardrail halts the loop, the
    synthesized halt message must be pushed through ``stream_delta_callback``
    so SSE/TUI clients see why the agent stopped instead of a silent stream
    close.  Without this the chat-completions SSE writer drains an empty
    queue and emits a finish chunk with zero content (indistinguishable
    from a crash for Open WebUI and similar clients).
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)
    # The mocked client returns SimpleNamespace responses which aren't
    # iterable as streaming chunks; force the non-streaming code path so
    # the guardrail-halt branch is reached without engaging the real
    # streaming machinery.
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt_text = result["final_response"]
    assert "stopped retrying" in halt_text

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )


def test_investment_strategy_provenance_survives_both_real_dispatch_paths(tmp_path, monkeypatch):
    from tools.tool_result_storage import extract_persisted_path
    from tools.investment_evidence_tool import _handle_investment_evidence

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    producer = "mcp__kospi_investment__analyze_strategy"
    payload = json.dumps({"structuredContent": {"read_only": True, "data": "evidence " * 30000}})
    for mode in ("sequential", "concurrent"):
        agent = _make_agent(producer, "investment_evidence")
        agent._turn_failed_evidence_reads = set()
        call = _mock_tool_call(producer, '{"read_only":true}', "strategy-" + mode)
        messages = []
        with patch("run_agent.handle_function_call", return_value=payload):
            getattr(agent, "_execute_tool_calls_" + mode)(
                SimpleNamespace(content="", tool_calls=[call]), messages, "task-1")
        path = extract_persisted_path(messages[-1]["content"])
        assert path, messages[-1]["content"][:500]
        page = json.loads(_handle_investment_evidence({"reference": path},
            session_id=agent.session_id, requester_id="task-1"))
        assert page["success"] is True, page
        failure = _mock_tool_call("investment_evidence", '{"reference":"missing.txt"}', "missing-" + mode)
        with patch("run_agent.handle_function_call", return_value='{"success":false,"status":"degraded"}'):
            getattr(agent, "_execute_tool_calls_" + mode)(
                SimpleNamespace(content="", tool_calls=[failure]), messages, "task-1")
        assert agent._turn_failed_evidence_reads


def test_evidence_disclosure_real_conversation_persists_and_resets(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from agent.investment_evidence_verifier import EVIDENCE_FAILURE_NOTICE

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _make_agent("investment_evidence")
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(agent.session_id, source="telegram")
    agent._session_db = db
    agent._session_db_created = True
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    agent.skip_background_review = True
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[
            _mock_tool_call("investment_evidence", '{"reference":"missing.txt"}', "missing-full-turn")]),
        _mock_response(content="분석 완료", finish_reason="stop"),
        _mock_response(content="안녕하세요", finish_reason="stop"),
    ]
    with patch("run_agent.handle_function_call", return_value='{"success":false,"status":"degraded"}'):
        result = agent.run_conversation("이 원문을 확인하고 결론을 알려줘")
    assert EVIDENCE_FAILURE_NOTICE in result["final_response"]
    rows = db.get_messages_as_conversation(agent.session_id)
    assert rows[-1]["content"] == result["final_response"]
    assert sum(EVIDENCE_FAILURE_NOTICE in str(row.get("content")) for row in rows) == 1
    following = agent.run_conversation("인사만 해줘", conversation_history=result["messages"])
    assert EVIDENCE_FAILURE_NOTICE not in following["final_response"]
    db.close()


def test_strategy_aggregate_proof_survives_all_dispatch_finalizers(tmp_path, monkeypatch):
    from tools.budget_config import BudgetConfig
    from tools.tool_result_storage import extract_persisted_path
    from tools.investment_evidence_tool import _handle_investment_evidence
    from agent.tool_executor import execute_tool_calls_segmented

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    budget = BudgetConfig(mcp_result_size=10000, turn_budget=1000, preview_size=100)
    monkeypatch.setattr("agent.tool_executor._budget_for_agent", lambda agent: budget)
    producer = "mcp__kospi_investment__analyze_strategy"
    content = json.dumps({"read_only": True, "data": "raw evidence " * 400})
    assert 1000 < len(content) < 10000
    for mode in ("sequential", "concurrent", "segmented"):
        agent = _make_agent(producer)
        messages = []
        call = _mock_tool_call(producer, '{"read_only":true}', "aggregate-" + mode)
        message = SimpleNamespace(content="", tool_calls=[call])
        with patch("run_agent.handle_function_call", return_value=content):
            if mode == "segmented":
                execute_tool_calls_segmented(agent, message, messages, "task-1")
            else:
                getattr(agent, "_execute_tool_calls_" + mode)(message, messages, "task-1")
        reference = extract_persisted_path(messages[-1]["content"])
        assert reference
        result = json.loads(_handle_investment_evidence({"reference": reference},
            session_id=agent.session_id, requester_id="task-1"))
        assert result["success"] is True, result


def test_unstarted_concurrent_calls_never_grant_strategy_provenance(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from tools.budget_config import BudgetConfig
    from tools.tool_result_storage import extract_persisted_path
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Force the non-execution diagnostic through both persistence layers.
    budget = BudgetConfig(default_result_size=1, mcp_result_size=1, turn_budget=1, preview_size=30)
    monkeypatch.setattr("agent.tool_executor._budget_for_agent", lambda agent: budget)
    producer = "mcp__kospi_investment__analyze_strategy"
    for failure in ("missing", "timeout", "interrupt", "shutdown"):
        for strategy_last in (False, True):
            class UnstartedPool:
                def __init__(self, **kwargs):
                    pass

                def submit(self, *args, **kwargs):
                    if failure == "shutdown":
                        raise RuntimeError("cannot schedule new futures after interpreter shutdown")
                    if failure == "interrupt":
                        agent._interrupt_requested = True
                    future = Future()
                    if failure == "missing":
                        future.set_result(None)
                    return future

                def shutdown(self, **kwargs):
                    pass

            monkeypatch.setattr("tools.daemon_pool.DaemonThreadPoolExecutor", UnstartedPool)
            monkeypatch.setattr("agent.tool_executor._resolve_concurrent_tool_timeout",
                                lambda: 0.0 if failure == "timeout" else None)
            agent = _make_agent(producer, "web_search")
            agent._interrupt_requested = False
            if failure == "interrupt":
                monkeypatch.setattr("agent.tool_executor.concurrent.futures.wait",
                                    lambda futures, **kwargs: (set(), set(futures)))
            strategy_id = f"unstarted-{failure}-{strategy_last}"
            other_id = f"other-{failure}-{strategy_last}"
            calls = [_mock_tool_call(producer, '{"read_only":true}', strategy_id),
                     _mock_tool_call("web_search", '{}', other_id)]
            if strategy_last:
                calls.reverse()
            messages = []
            with patch("run_agent.handle_function_call") as invoke:
                agent._execute_tool_calls_concurrent(SimpleNamespace(content="", tool_calls=calls), messages, "task-1")
                invoke.assert_not_called()
            proofs = agent._turn_strategy_execution_provenance
            assert other_id not in proofs
            assert proofs[strategy_id]["read_only"] is False
            strategy_message = next(message for message in messages if message['tool_call_id'] == strategy_id)
            reference = extract_persisted_path(strategy_message['content'])
            assert reference
            metadata = json.loads(Path(reference + '.meta.json').read_text())
            assert metadata['tool_name'] == producer
            assert metadata['strategy_read_only'] is False
