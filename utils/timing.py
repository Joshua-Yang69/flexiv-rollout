from __future__ import annotations

import time
from dataclasses import dataclass


def now_ms() -> float:
    return time.time() * 1000.0


@dataclass
class LoopStats:
    iterations: int = 0
    overruns: int = 0
    last_period_ms: float = 0.0
    last_work_ms: float = 0.0


class RateLimiter:
    """Fixed-rate loop helper based on time.perf_counter."""

    def __init__(self, hz: float) -> None:
        if hz <= 0:
            raise ValueError("Rate must be positive.")
        self.hz = float(hz)
        self.period_s = 1.0 / self.hz
        self._next_deadline = time.perf_counter()
        self._last_start = self._next_deadline
        self.stats = LoopStats()

    def mark_start(self) -> None:
        now = time.perf_counter()
        self.stats.last_period_ms = (now - self._last_start) * 1000.0
        self._last_start = now

    def sleep(self) -> None:
        now = time.perf_counter()
        self.stats.last_work_ms = (now - self._last_start) * 1000.0
        self._next_deadline += self.period_s
        sleep_s = self._next_deadline - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            self.stats.overruns += 1
            self._next_deadline = now
        self.stats.iterations += 1

