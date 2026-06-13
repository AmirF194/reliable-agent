"""End-to-end tests on the offline FakeBackend — tool use + retries, no API key needed."""
from __future__ import annotations

from reliable_agent.agent import build_agent
from reliable_agent.backends.fake import FakeBackend
from reliable_agent.config import Config
from reliable_agent.reliability import with_retries
from reliable_agent.tools import default_registry
from reliable_agent.tracing import Tracer


def test_fake_agent_uses_calculator_and_answers():
    agent = build_agent(Config(backend="fake"))
    result = agent.run("What is 21 * 2? Give the final number.", run_id="t")
    assert result.stop_reason == "end_turn"
    assert "42" in result.answer
    assert result.trace["model_calls"] >= 2  # one turn to call the tool, one to answer


def test_fake_agent_recovers_from_tool_error():
    agent = build_agent(Config(backend="fake"))
    result = agent.run("Compute 10 divided by 0 with the calculator.", run_id="t")
    assert result.stop_reason == "end_turn"
    assert "zero" in result.answer.lower() or "division" in result.answer.lower()


def test_fake_retry_then_succeed_is_traced():
    backend = FakeBackend(flaky=2)
    tracer = Tracer(run_id="t")
    out = with_retries(
        lambda: backend.generate("sys", [{"role": "user", "text": "1 + 1"}], default_registry().specs()),
        retryable=backend.RETRYABLE,
        max_retries=5,
        tracer=tracer,
        base_delay=0,
    )
    assert out.tool_calls and out.tool_calls[0].name == "calculator"
    assert sum(1 for e in tracer.events if e["event"] == "retry") == 2
