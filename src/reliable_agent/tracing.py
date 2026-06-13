"""Minimal structured tracing. Every step is a span/event you can later ship to
OpenTelemetry, Langfuse, or a file. Here we keep an in-memory list + JSONL stdout.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tracer:
    run_id: str
    events: list[dict] = field(default_factory=list)
    echo: bool = False  # print spans as they happen (CLI use)

    def event(self, name: str, **fields: Any) -> None:
        record = {"run_id": self.run_id, "event": name, **fields}
        self.events.append(record)
        if self.echo:
            print(json.dumps(record), file=sys.stderr)

    def summary(self) -> dict:
        steps = [e for e in self.events if e["event"] == "model.call"]
        retries = [e for e in self.events if e["event"] == "retry"]
        return {
            "run_id": self.run_id,
            "model_calls": len(steps),
            "retries": len(retries),
            "input_tokens": sum(e.get("input_tokens", 0) for e in steps),
            "output_tokens": sum(e.get("output_tokens", 0) for e in steps),
        }
