"""Small, privacy-safe helpers for ephemeral public work status."""

from __future__ import annotations

import re
from typing import Optional

from agent.redact import redact_sensitive_text


CAPTION_TTL_SECONDS = 90
PHASE_TTL_SECONDS = 45
PHASE_REFRESH_SECONDS = 15
RUNTIME_PHASE_CODES = frozenset({
    "starting", "thinking", "using_tool", "coordinating", "waiting",
    "organizing", "working",
})

_FENCED_CODE_RE = re.compile(r"```[^\n]*\n.*?```|```.*?```", re.DOTALL)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_ABSOLUTE_PATH_RE = re.compile(
    r"/(?:Users|home|private|var|tmp|Volumes|Applications|Library|etc|opt|usr)/"
    r"(?:[^\s`'\"<>]+/)*[^\s`'\"<>]+"
)
_BARE_URL_RE = re.compile(r"https?://[^\s<>)\]]+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def sanitize_live_caption(text: object, *, max_chars: int = 160) -> str:
    """Return one short public commentary sentence, never raw code or paths."""
    if not isinstance(text, str) or not text.strip():
        return ""
    value = redact_sensitive_text(
        text,
        force=True,
        redact_url_credentials=True,
    )
    value = _FENCED_CODE_RE.sub(" ", value)
    value = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1), value)
    value = _ABSOLUTE_PATH_RE.sub("[로컬 경로]", value)
    value = _BARE_URL_RE.sub("[웹 주소]", value)
    value = _SPACE_RE.sub(" ", value).strip()
    if len(value) <= max_chars:
        return value
    bounded = value[:max_chars].rstrip()
    sentence_end = max(bounded.rfind(mark) for mark in (".", "!", "?", "。", "요."))
    if sentence_end >= max_chars // 2:
        return bounded[: sentence_end + 1]
    return bounded.rstrip(" ,.;:—-") + "…"


def runtime_phase_for_activity(description: object) -> str:
    """Classify only mechanically observed runtime activity."""
    value = str(description or "").strip().lower()
    if any(token in value for token in ("compression", "compressing", "context compact")):
        return "organizing"
    if any(token in value for token in ("waiting", "backoff", "retrying", "rate limit")):
        return "waiting"
    if any(token in value for token in ("delegat", "kanban", "coordinat", "handoff")):
        return "coordinating"
    if any(token in value for token in ("tool", "browser", "terminal", "execute")):
        return "using_tool"
    if any(token in value for token in ("spawn", "claim", "starting", "initializ")):
        return "starting"
    if any(token in value for token in ("api", "thinking", "stream response", "model response")):
        return "thinking"
    return "working"


def valid_runtime_phase(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value in RUNTIME_PHASE_CODES else None
