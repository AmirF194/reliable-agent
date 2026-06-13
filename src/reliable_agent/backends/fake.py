"""Offline, deterministic backend — no API key, no network.

This is what makes the whole project runnable in CI and on a reviewer's laptop with
zero setup: the agent loop, reliability layer, tools, evals, and tracing all exercise
real code paths against scripted-but-realistic model behavior. Swap ``backend="fake"``
for ``"claude"`` or ``"cohere"`` and the exact same loop talks to a real model.
"""
from __future__ import annotations

import re

from .base import LLMBackend, LLMResponse, Tool, ToolCall, Turn


class FakeTransientError(Exception):
    """A simulated transient failure, so the retry layer can be exercised offline."""


# Normalize a few English math phrasings into symbols so tasks can read naturally.
_WORD_OPS = [
    (re.compile(r"\bdivided by\b", re.I), "/"),
    (re.compile(r"\b(?:multiplied by|times)\b", re.I), "*"),
    (re.compile(r"\bplus\b", re.I), "+"),
    (re.compile(r"\bminus\b", re.I), "-"),
]
_EXPR = re.compile(r"[\d(][\d\s().+\-*/%]*[+\-*/%][\d\s().+\-*/%]*\d")


def _find_expression(text: str) -> str | None:
    norm = text
    for pat, sym in _WORD_OPS:
        norm = pat.sub(sym, norm)
    match = _EXPR.search(norm)
    return match.group(0).strip() if match else None


def _toks(s: str) -> int:
    return max(1, len(s) // 4)


class FakeBackend(LLMBackend):
    """Deterministic behavior: call the calculator when the task is arithmetic, otherwise
    answer directly; on the turn after a tool result, summarize that result and stop."""

    name = "fake"
    RETRYABLE = (FakeTransientError,)

    def __init__(self, flaky: int = 0) -> None:
        # If flaky > 0, fail this many times with a retryable error before succeeding —
        # lets the example/tests show the retry layer working, fully offline.
        self._remaining_failures = flaky

    def generate(self, system: str, transcript: list[Turn], tools: list[Tool]) -> LLMResponse:
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise FakeTransientError("simulated transient error")

        task = next((t["text"] for t in transcript if t["role"] == "user"), "")
        last_tool = next((t for t in reversed(transcript) if t["role"] == "tool"), None)
        have_calc = any(tool.name == "calculator" for tool in tools)

        # Second pass: a tool result is available -> produce the final answer.
        if last_tool is not None:
            content = last_tool["content"]
            answer = f"Done. Result: {content}"
            return LLMResponse(answer, [], "end_turn", _toks(task), _toks(answer))

        # First pass, arithmetic -> call the calculator tool.
        expr = _find_expression(task) if have_calc else None
        if expr is not None:
            call = ToolCall(id="fake-call-1", name="calculator", arguments={"expression": expr})
            return LLMResponse("", [call], "tool_use", _toks(task), 4)

        # First pass, non-arithmetic -> answer directly.
        answer = (
            "Reliability matters more than demos because production AI must keep working "
            "under rate limits, timeouts, and bad output — which is what earns user trust."
        )
        return LLMResponse(answer, [], "end_turn", _toks(task), _toks(answer))
