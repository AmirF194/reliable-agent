"""CLI entry point:  python -m reliable_agent "your task here" """
from __future__ import annotations

import sys

from .agent import build_agent
from .config import load_config


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: python -m reliable_agent "your task here"', file=sys.stderr)
        return 2

    task = " ".join(sys.argv[1:])
    config = load_config()
    agent = build_agent(config)
    result = agent.run(task, run_id="cli", echo=True)

    print("\n" + "=" * 60)
    print(f"answer      : {result.answer}")
    print(f"stop_reason : {result.stop_reason}")
    print(f"trace       : {result.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
