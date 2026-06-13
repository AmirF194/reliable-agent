"""Provider-neutral types and the backend interface.

The agent loop speaks this neutral transcript; each backend translates it to its own
wire format. Adding a provider means implementing `LLMBackend` in one new file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "refusal" | ...
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


# A turn in the neutral transcript. One of:
#   {"role": "user", "text": str}
#   {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...]}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}
Turn = dict[str, Any]


@dataclass
class Tool:
    """Canonical tool definition (Claude-native shape; backends convert as needed)."""

    name: str
    description: str
    input_schema: dict[str, Any]


class LLMBackend(Protocol):
    name: str

    def generate(self, system: str, transcript: list[Turn], tools: list[Tool]) -> LLMResponse:
        """One model call. Must raise on transport errors so the retry layer can catch them."""
        ...
