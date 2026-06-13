"""Claude (Anthropic) backend.

Uses the manual agentic-loop pattern: we drive the loop ourselves so retries,
validation, budgets, and tracing all live in our code, not the SDK's.
Model id and SDK usage verified against the Anthropic Python SDK (claude-opus-4-8).
"""
from __future__ import annotations

import anthropic

from .base import LLMBackend, LLMResponse, ToolCall, Tool, Turn


class ClaudeBackend(LLMBackend):
    name = "claude"

    # Transient/server errors worth retrying. Imported by reliability.py.
    RETRYABLE = (
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )

    def __init__(self, model: str, max_tokens: int, timeout: float):
        self.model = model
        self.max_tokens = max_tokens
        # The SDK retries some errors itself; we own retry policy, so disable it here.
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=0)

    def generate(self, system: str, transcript: list[Turn], tools: list[Tool]) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=self._to_messages(transcript),
            tools=[self._to_tool(t) for t in tools] or anthropic.NOT_GIVEN,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "end_turn",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw=resp,
        )

    # --- transcript -> Anthropic messages -----------------------------------
    @staticmethod
    def _to_tool(t: Tool) -> dict:
        return {"name": t.name, "description": t.description, "input_schema": t.input_schema}

    @staticmethod
    def _to_messages(transcript: list[Turn]) -> list[dict]:
        messages: list[dict] = []
        for turn in transcript:
            role = turn["role"]
            if role == "user":
                messages.append({"role": "user", "content": turn["text"]})
            elif role == "assistant":
                content: list[dict] = []
                if turn.get("text"):
                    content.append({"type": "text", "text": turn["text"]})
                for call in turn.get("tool_calls", []):
                    content.append(
                        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                    )
                messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Consecutive user messages are merged by the API, so one block per result is fine.
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": turn["tool_call_id"],
                                "content": turn["content"],
                                "is_error": turn.get("is_error", False),
                            }
                        ],
                    }
                )
        return messages
