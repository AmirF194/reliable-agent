# reliable-agent

> **Most AI never makes it out of the demo. This is the part that gets it there.**
>
> A production-grade AI agent framework where the *reliability layer* is the product:
> retries with backoff + jitter, per-call timeouts, token budgets, schema-validated tool
> calls, structured tracing, and a real eval harness with a CI gate. Model-agnostic — runs
> on **Claude**, **Cohere Command**, or a fully offline **Fake** backend (zero keys — for
> tests, CI, and demos).

[![CI](https://github.com/AmirF194/reliable-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirF194/reliable-agent/actions)

---

## Why this exists

A chatbot demo is easy. An agent you can *depend on* — that retries transient failures, refuses to
emit malformed tool calls, stays within a token budget, traces every step, and fails a CI build when
task success regresses — is the actual engineering. This repo is that engineering, made small and readable.

It's deliberately **not** a framework you'd `pip install` and forget. It's a reference you can read in
ten minutes and see exactly how each reliability guarantee is implemented.

## What's in the box

| Capability | Where | What it proves |
|---|---|---|
| **Provider-agnostic backend** | [`backends/`](src/reliable_agent/backends/) | One agent loop, swappable models — Claude ⇄ Cohere ⇄ offline Fake |
| **Reliability layer** | [`reliability.py`](src/reliable_agent/reliability.py) | Exponential backoff, jitter, timeouts, typed-error handling |
| **Validated tool calls** | [`tools.py`](src/reliable_agent/tools.py) | JSON-schema validation before any tool runs; structured errors back to the model |
| **Cost & step budgeting** | [`agent.py`](src/reliable_agent/agent.py) | Hard ceilings on tokens and loop steps — no runaway agents |
| **Tracing** | [`tracing.py`](src/reliable_agent/tracing.py) | Every step (model call, tool call, retry) is a structured span |
| **Eval harness** | [`evals/`](evals/) | Task success rate, p50/p95 latency, cost/run — gated in CI |
| **Runs offline, no keys** | [`backends/fake.py`](src/reliable_agent/backends/fake.py) | Deterministic Fake backend → tests, eval gate, and the demo all run with zero setup |

## Architecture

```
                ┌──────────────────────────────────────────┐
   user task ──▶│  Agent loop (agent.py)                    │
                │   • step budget        • cost budget       │
                │   • tracing spans      • tool dispatch     │
                └───────┬───────────────────────┬───────────┘
                        │ generate()            │ validate + run
                        ▼                       ▼
              ┌──────────────────┐     ┌──────────────────┐
              │ LLMBackend       │     │ Tool registry    │
              │  • ClaudeBackend │     │  • JSON-schema    │
              │  • CohereBackend │     │    validation     │
              │  • FakeBackend   │     │                   │
              └────────┬─────────┘     └──────────────────┘
                       │ wrapped in
                       ▼
              ┌──────────────────┐
              │ reliability.py   │  retries · backoff · timeout
              └──────────────────┘
```

## Quickstart

```bash
git clone https://github.com/AmirF194/reliable-agent
cd reliable-agent
pip install -e ".[dev]"

# Everything below runs OFFLINE — no API key, no network (Fake backend):
python examples/research_agent.py             # agent loop + reliability layer, end to end
AGENT_BACKEND=fake python -m evals.run_evals  # eval suite + CI success-rate gate
pytest                                         # reliability guarantees, unit-tested

# Use a real model — same loop, one env var (needs a key):
cp .env.example .env                           # add ANTHROPIC_API_KEY and/or COHERE_API_KEY
python -m reliable_agent "What is 17 * 23, and is the result prime?"
AGENT_BACKEND=cohere python -m reliable_agent "..."
```

## The eval harness is the point

Anyone can demo an agent that works once. The eval harness measures whether it works
*reliably* — and the CI gate fails the build if the success rate drops below the threshold
in [`evals/run_evals.py`](evals/run_evals.py). It runs **offline on the Fake backend**, so CI
needs no secrets; point it at a real model with `AGENT_BACKEND=claude` for real latency and cost.

```
$ AGENT_BACKEND=fake python -m evals.run_evals

success rate: 4/4 (100%)   threshold: 75%
report → evals/report.md
```

The tasks (see [`evals/tasks.jsonl`](evals/tasks.jsonl)) exercise arithmetic, explicit
tool use, a tool-error recovery path (divide-by-zero), and a no-tool answer — each checked
against an expected pattern. That's the difference between "it worked on my machine" and
"it stays working."

## Design decisions & tradeoffs

- **Manual agent loop, not a black-box framework.** You can see exactly where retries, validation, and
  budgets are enforced. Readability over magic.
- **Backends own their wire format.** The loop speaks a neutral transcript; each backend translates. Adding
  a third provider is one file.
- **Validate tool inputs *before* executing.** A malformed tool call returns a structured error to the
  model (which can self-correct) instead of crashing the process.
- **Budgets are hard ceilings, not suggestions.** The loop stops at the token/step limit and says so —
  the failure mode is "stopped early, here's why," never "burned $40 silently."

## Status

Reference implementation. The Claude backend is complete; the Cohere backend is a working starting point
(see TODOs in [`backends/cohere.py`](src/reliable_agent/backends/cohere.py)). Built by
[Amir Fathi](https://www.linkedin.com/in/fathiamir) (FastInfer).

## License

MIT — see [LICENSE](LICENSE).
