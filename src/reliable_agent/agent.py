"""The agent loop. Where step budgets, cost budgets, tracing, retries, and tool
dispatch all come together. Provider-neutral — works with any LLMBackend.
"""
from __future__ import annotations

from dataclasses import dataclass

from .backends.base import LLMBackend, Turn
from .config import Config
from .reliability import with_retries
from .tools import ToolRegistry
from .tracing import Tracer

SYSTEM_PROMPT = (
    "You are a reliable task-completing agent. Use tools when they improve accuracy. "
    "When the task is complete, state the final answer clearly and stop."
)


@dataclass
class AgentResult:
    answer: str
    stop_reason: str  # "end_turn" | "max_steps" | "token_budget" | "refusal"
    transcript: list[Turn]
    trace: dict


class Agent:
    def __init__(self, backend: LLMBackend, tools: ToolRegistry, config: Config):
        self.backend = backend
        self.tools = tools
        self.config = config

    def run(self, task: str, *, run_id: str = "run", echo: bool = False) -> AgentResult:
        tracer = Tracer(run_id=run_id, echo=echo)
        transcript: list[Turn] = [{"role": "user", "text": task}]
        retryable = getattr(self.backend, "RETRYABLE", ())
        spent_tokens = 0

        for step in range(self.config.max_steps):
            resp = with_retries(
                lambda: self.backend.generate(SYSTEM_PROMPT, transcript, self.tools.specs()),
                retryable=retryable,
                max_retries=self.config.max_retries,
                tracer=tracer,
            )
            spent_tokens += resp.input_tokens + resp.output_tokens
            tracer.event(
                "model.call", step=step, stop_reason=resp.stop_reason,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                tool_calls=len(resp.tool_calls),
            )

            if resp.stop_reason == "refusal":
                return AgentResult(resp.text or "[refused]", "refusal", transcript, tracer.summary())

            transcript.append(
                {"role": "assistant", "text": resp.text, "tool_calls": resp.tool_calls}
            )

            if not resp.tool_calls:
                return AgentResult(resp.text, "end_turn", transcript, tracer.summary())

            for call in resp.tool_calls:
                content, is_error = self.tools.dispatch(call.name, call.arguments)
                tracer.event("tool.call", tool=call.name, is_error=is_error)
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": content,
                        "is_error": is_error,
                    }
                )

            # Hard cost ceiling — stop early and say so, never burn budget silently.
            if self.config.token_budget and spent_tokens >= self.config.token_budget:
                tracer.event("budget.exceeded", spent_tokens=spent_tokens)
                return AgentResult(
                    resp.text or "[stopped: token budget exceeded]",
                    "token_budget", transcript, tracer.summary(),
                )

        tracer.event("budget.max_steps", steps=self.config.max_steps)
        return AgentResult("[stopped: max steps reached]", "max_steps", transcript, tracer.summary())


def build_agent(config: Config) -> Agent:
    from .tools import default_registry

    if config.backend == "claude":
        from .backends.claude import ClaudeBackend

        backend: LLMBackend = ClaudeBackend(config.claude_model, config.max_tokens, config.request_timeout)
    elif config.backend == "cohere":
        from .backends.cohere import CohereBackend

        backend = CohereBackend(config.cohere_model, config.max_tokens, config.request_timeout)
    elif config.backend == "fake":
        from .backends.fake import FakeBackend

        backend = FakeBackend()
    else:
        raise ValueError(
            f"Unknown backend: {config.backend!r} (expected 'claude', 'cohere', or 'fake')"
        )

    return Agent(backend, default_registry(), config)
