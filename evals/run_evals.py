"""Eval harness — the part that proves reliability, not just capability.

Runs each task, checks success against an expected regex, and records latency, tokens,
and cost. Writes evals/report.md and exits non-zero if success rate < THRESHOLD so CI
fails on regressions.

Usage:  python -m evals.run_evals
Skips gracefully (exit 0) when no API key is configured, so CI without secrets is green.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import time
from pathlib import Path

from reliable_agent.agent import build_agent
from reliable_agent.config import load_config

THRESHOLD = 0.75  # fail CI below this success rate
HERE = Path(__file__).parent

# Rough $/1M tokens for cost estimate (input+output blended; adjust per model).
COST_PER_MTOK = {"claude": 15.0, "cohere": 2.5, "fake": 0.0}


def _has_key(backend: str) -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") if backend == "claude" else os.getenv("COHERE_API_KEY"))


def main() -> int:
    config = load_config()
    # The fake backend needs no key — it's how CI runs the eval gate offline.
    if config.backend != "fake" and not _has_key(config.backend):
        print(f"[skip] no API key for backend '{config.backend}'; eval run skipped.")
        return 0

    agent = build_agent(config)
    tasks = [json.loads(line) for line in (HERE / "tasks.jsonl").read_text().splitlines() if line.strip()]

    rows, latencies = [], []
    for t in tasks:
        start = time.monotonic()
        result = agent.run(t["task"], run_id=t["id"])
        elapsed_ms = (time.monotonic() - start) * 1000
        latencies.append(elapsed_ms)

        ok = bool(re.search(t["expect_regex"], result.answer))
        tokens = result.trace["input_tokens"] + result.trace["output_tokens"]
        cost = tokens / 1_000_000 * COST_PER_MTOK.get(config.backend, 0)
        rows.append(
            {"id": t["id"], "ok": ok, "ms": elapsed_ms, "tokens": tokens,
             "cost": cost, "retries": result.trace["retries"]}
        )

    passed = sum(r["ok"] for r in rows)
    rate = passed / len(rows) if rows else 0.0
    _write_report(rows, rate, config.backend, latencies)

    print(f"\nsuccess rate: {passed}/{len(rows)} ({rate:.0%})   threshold: {THRESHOLD:.0%}")
    print(f"report → {HERE / 'report.md'}")
    if rate < THRESHOLD:
        print("FAIL: success rate below threshold")
        return 1
    return 0


def _write_report(rows, rate, backend, latencies) -> None:
    p50 = statistics.median(latencies) if latencies else 0
    p95 = (statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1 else latencies[0]) if latencies else 0
    lines = [
        "# Eval Report",
        "",
        f"- backend: `{backend}`",
        f"- success rate: **{sum(r['ok'] for r in rows)}/{len(rows)} ({rate:.0%})**",
        f"- latency p50/p95: {p50:.0f}ms / {p95:.0f}ms",
        f"- total cost: ${sum(r['cost'] for r in rows):.4f}",
        "",
        "| task | ok | latency_ms | tokens | retries | cost_usd |",
        "|---|:--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {'✓' if r['ok'] else '✗'} | {r['ms']:.0f} | "
            f"{r['tokens']} | {r['retries']} | ${r['cost']:.4f} |"
        )
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
