"""Cohere Command backend — working starting point.

Demonstrates the agent loop is genuinely model-agnostic. Cohere's chat API uses an
OpenAI-style tool-call shape; we translate the neutral transcript to it here.

TODO (intentional, for the reader):
  - Verify the exact tool-call field names against your installed `cohere` version.
  - Map Cohere's finish_reason values onto the neutral stop_reason vocabulary.
  - Add Cohere-specific retryable exceptions to RETRYABLE.
"""
from __future__ import annotations

import json
import os

from .base import LLMBackend, LLMResponse, ToolCall, Tool, Turn


class CohereBackend(LLMBackend):
    name = "cohere"
    RETRYABLE: tuple = ()  # TODO: add cohere.errors transient types

    def __init__(self, model: str, max_tokens: int, timeout: float):
        import cohere  # imported lazily so Claude-only users don't need the dep

        self.model = model
        self.max_tokens = max_tokens
        self.client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"], timeout=timeout)

    def generate(self, system: str, transcript: list[Turn], tools: list[Tool]) -> LLMResponse:
        resp = self.client.chat(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self._to_messages(system, transcript),
            tools=[self._to_tool(t) for t in tools] or None,
        )
        msg = resp.message
        text = "".join(c.text for c in (msg.content or []) if getattr(c, "type", "") == "text")
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",  # TODO: map finish_reason
            input_tokens=int(getattr(resp.usage.tokens, "input_tokens", 0) or 0),
            output_tokens=int(getattr(resp.usage.tokens, "output_tokens", 0) or 0),
            raw=resp,
        )

    @staticmethod
    def _to_tool(t: Tool) -> dict:
        return {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
        }

    @staticmethod
    def _to_messages(system: str, transcript: list[Turn]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in transcript:
            role = turn["role"]
            if role == "user":
                messages.append({"role": "user", "content": turn["text"]})
            elif role == "assistant":
                m: dict = {"role": "assistant", "content": turn.get("text", "")}
                if turn.get("tool_calls"):
                    m["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in turn["tool_calls"]
                    ]
                messages.append(m)
            elif role == "tool":
                messages.append(
                    {"role": "tool", "tool_call_id": turn["tool_call_id"], "content": turn["content"]}
                )
        return messages
