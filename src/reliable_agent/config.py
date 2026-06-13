"""Runtime configuration. All limits are hard ceilings, read from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    backend: str = os.getenv("AGENT_BACKEND", "claude")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")
    cohere_model: str = os.getenv("COHERE_MODEL", "command-a-03-2025")

    max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "10"))
    max_retries: int = int(os.getenv("AGENT_MAX_RETRIES", "4"))
    request_timeout: float = float(os.getenv("AGENT_TIMEOUT", "60"))

    # Stop the loop if cumulative tokens exceed this. 0 = no cost ceiling.
    token_budget: int = int(os.getenv("AGENT_TOKEN_BUDGET", "50000"))


def load_config() -> Config:
    return Config()
