"""The reliability layer: retry transient model errors with exponential backoff + jitter.

This is the star of the repo. A demo calls the API once and hopes. Production wraps every
model call so a rate limit or a blip doesn't take down the run.
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from .tracing import Tracer

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    *,
    retryable: tuple[type[Exception], ...],
    max_retries: int,
    tracer: Tracer,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Call `fn`, retrying only on `retryable` exceptions. Non-retryable errors propagate immediately."""
    attempt = 0
    while True:
        try:
            return fn()
        except retryable as exc:  # type: ignore[misc]
            attempt += 1
            if attempt > max_retries:
                tracer.event("retry.exhausted", attempts=attempt, error=type(exc).__name__)
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 0.5)
            tracer.event(
                "retry", attempt=attempt, delay_s=round(delay, 2), error=type(exc).__name__
            )
            time.sleep(delay)
