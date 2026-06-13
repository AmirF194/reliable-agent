"""End-to-end demo — runs fully offline on the FakeBackend (no API key, no network).

    python examples/research_agent.py

It shows the two things this repo is about:
  1. The agent loop driving a tool call and returning a final answer.
  2. The reliability layer retrying a transient failure, captured in the trace.

Swap Config(backend="fake") for Config(backend="claude") (and `pip install -e .` already
pulls anthropic; set ANTHROPIC_API_KEY) to run the identical loop against a real model.
"""
from __future__ import annotations

from reliable_agent.agent import build_agent
from reliable_agent.backends.fake import FakeBackend
from reliable_agent.config import Config
from reliable_agent.reliability import with_retries
from reliable_agent.tools import default_registry
from reliable_agent.tracing import Tracer


def demo_agent_loop() -> None:
    print("1) Agent loop (tool use -> final answer), offline:\n")
    agent = build_agent(Config(backend="fake"))
    result = agent.run("What is 21 * 2? Give the final number.", run_id="example", echo=True)
    print(f"\n   answer      : {result.answer}")
    print(f"   stop_reason : {result.stop_reason}")
    print(f"   trace       : {result.trace}\n")


def demo_retry_layer() -> None:
    print("2) Reliability layer (retry transient errors with backoff), offline:\n")
    backend = FakeBackend(flaky=2)  # fail twice, then succeed
    tracer = Tracer(run_id="retry-demo")
    out = with_retries(
        lambda: backend.generate("sys", [{"role": "user", "text": "1 + 1"}], default_registry().specs()),
        retryable=backend.RETRYABLE,
        max_retries=5,
        tracer=tracer,
        base_delay=0,  # no real sleeping in the demo
    )
    retries = sum(1 for e in tracer.events if e["event"] == "retry")
    print(f"   succeeded after {retries} retries; the model wanted tool: "
          f"{out.tool_calls[0].name if out.tool_calls else None}")
    for e in tracer.events:
        print(f"   trace event: {e}")


if __name__ == "__main__":
    demo_agent_loop()
    demo_retry_layer()
