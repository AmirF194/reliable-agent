"""Offline tests — no API key required, so CI stays green without secrets.
These cover the reliability guarantees, which is exactly what should never silently break.
"""
from __future__ import annotations

import pytest

from reliable_agent.reliability import with_retries
from reliable_agent.tools import default_registry
from reliable_agent.tracing import Tracer
from reliable_agent.backends.claude import ClaudeBackend
from reliable_agent.backends.base import ToolCall


class Transient(Exception):
    pass


def test_retries_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Transient()
        return "ok"

    tracer = Tracer(run_id="t")
    out = with_retries(flaky, retryable=(Transient,), max_retries=5, tracer=tracer, base_delay=0)
    assert out == "ok"
    assert calls["n"] == 3
    assert sum(1 for e in tracer.events if e["event"] == "retry") == 2


def test_non_retryable_propagates_immediately():
    def boom():
        raise ValueError("nope")

    tracer = Tracer(run_id="t")
    with pytest.raises(ValueError):
        with_retries(boom, retryable=(Transient,), max_retries=5, tracer=tracer, base_delay=0)


def test_retries_exhaust_and_raise():
    def always_fail():
        raise Transient()

    tracer = Tracer(run_id="t")
    with pytest.raises(Transient):
        with_retries(always_fail, retryable=(Transient,), max_retries=2, tracer=tracer, base_delay=0)
    assert any(e["event"] == "retry.exhausted" for e in tracer.events)


def test_tool_validation_rejects_bad_args():
    reg = default_registry()
    content, is_error = reg.dispatch("calculator", {"wrong_field": "1+1"})
    assert is_error
    assert "Invalid arguments" in content


def test_tool_runs_valid_args():
    reg = default_registry()
    content, is_error = reg.dispatch("calculator", {"expression": "6 * 7"})
    assert not is_error
    assert content == "42"


def test_unknown_tool_is_structured_error():
    reg = default_registry()
    content, is_error = reg.dispatch("nonexistent", {})
    assert is_error and "Unknown tool" in content


def test_claude_transcript_conversion():
    msgs = ClaudeBackend._to_messages(
        [
            {"role": "user", "text": "hi"},
            {"role": "assistant", "text": "calling", "tool_calls": [ToolCall("id1", "calculator", {"expression": "1+1"})]},
            {"role": "tool", "tool_call_id": "id1", "name": "calculator", "content": "2", "is_error": False},
        ]
    )
    assert msgs[0] == {"role": "user", "content": "hi"}
    assert msgs[1]["content"][1]["type"] == "tool_use"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "id1"
