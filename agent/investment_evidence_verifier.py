"""Keep failed evidence reads visible even when the model omits them."""

from __future__ import annotations

import json

from tools.tool_result_storage import _READ_ONLY_STRATEGY_PRODUCER


def record_strategy_execution(agent, tool_name, arguments, tool_call_id, *, executed):
    if tool_name != _READ_ONLY_STRATEGY_PRODUCER:
        return
    state = getattr(agent, "_turn_strategy_execution_provenance", None)
    if state is None:
        state = agent._turn_strategy_execution_provenance = {}
    state[tool_call_id] = {
        "tool_name": tool_name,
        "read_only": executed and isinstance(arguments, dict) and arguments.get("read_only") is True,
    }

EVIDENCE_FAILURE_NOTICE = (
    "근거 확인 제한: 이번 답변에서 요청한 투자 원문 중 읽기에 실패한 부분이 있습니다. "
    "위 답변은 전체 근거 확인을 마친 결론이 아니며, 미확인 부분에 의존한 투자 판단은 확정할 수 없습니다."
)


def record_evidence_read(agent, tool_name, arguments, result):
    if tool_name != "investment_evidence":
        return
    state = getattr(agent, "_turn_failed_evidence_reads", None)
    if state is None:
        state = agent._turn_failed_evidence_reads = set()
    args = arguments if isinstance(arguments, dict) else {}
    key = json.dumps({
        "reference": args.get("reference"),
        "offset": args.get("offset", 1),
        "limit": args.get("limit", 500),
        "character_offset": args.get("character_offset"),
    }, sort_keys=True)
    try:
        payload = json.loads(result) if isinstance(result, str) else result
        success = isinstance(payload, dict) and payload.get("success") is True
    except (ValueError, TypeError):
        success = False
    if success:
        state.discard(key)
    else:
        state.add(key)


def disclose_evidence_failure(agent, response):
    if (not isinstance(response, str) or not response
            or not getattr(agent, "_turn_failed_evidence_reads", None)
            or EVIDENCE_FAILURE_NOTICE in response):
        return response
    return response.rstrip() + "\n\n" + EVIDENCE_FAILURE_NOTICE
