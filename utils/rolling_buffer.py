"""
utils/rolling_buffer.py
───────────────────────
Thread-safe rolling (fixed-capacity) deque for streaming data.

Unlike ``LatestBuffer`` which only holds the most recent value, a
``RollingBuffer`` retains the last *capacity* values and allows the consumer
to reconstruct temporal windows with arbitrary strides.

Typical use: perception thread appends every frame; model inference thread
reads a strided window for temporal feature extraction.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any


class RollingBuffer:
    """Fixed-capacity rolling buffer backed by a thread-safe deque.

    Parameters
    ----------
    capacity:
        Maximum number of items to retain.  Older items are silently dropped.

    Example
    -------
    Perception thread (50 Hz)::

        buf = RollingBuffer(capacity=100)
        # ...in perception loop:
        buf.put(observation)

    Inference thread (30 Hz)::

        # Get last 5 frames sampled every 5 perception frames
        frames = buf.sample_strided(history_len=5, stride=5)
        # frames[0] = oldest, frames[-1] = most recent
    """

    def __init__(self, capacity: int = 100) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be ≥ 1, got {capacity}")
        self._capacity = capacity
        self._dq: deque[Any] = deque(maxlen=capacity)
        self._lock = Lock()
        self._total_count: int = 0  # monotonic put counter

    # ── write ──────────────────────────────────────────────────────────────────

    def put(self, value: Any) -> None:
        """Append a value; drops the oldest item when at capacity."""
        with self._lock:
            self._dq.append(value)
            self._total_count += 1

    # ── read ───────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._dq)

    @property
    def total_count(self) -> int:
        """Monotonic count of all ``put`` calls since construction."""
        with self._lock:
            return self._total_count

    def latest(self) -> Any | None:
        """Return the most recently put value (or None if empty)."""
        with self._lock:
            return self._dq[-1] if self._dq else None

    def snapshot(self) -> list[Any]:
        """Return a shallow copy of all buffered items (oldest → newest)."""
        with self._lock:
            return list(self._dq)

    def sample_strided(self, history_len: int, stride: int) -> list[Any]:
        """Return ``history_len`` items sampled every ``stride`` positions.

        Sampling is anchored at the **newest** item (index -1) and steps
        backwards by ``stride`` each time, matching the ViTacDreamer training
        convention::

            result[0] = buf[-(history_len-1)*stride - 1]  (oldest slot)
            result[-1] = buf[-1]                           (most recent)

        If the buffer does not have enough items to fill a slot, the **oldest
        available item is repeated** (zero-padding at the start is avoided so
        the tensor always contains real observations).

        Parameters
        ----------
        history_len:
            Number of temporal slots to return (including the most-recent one).
        stride:
            Frame spacing between consecutive slots.

        Returns
        -------
        list of Any
            Length exactly ``history_len``, oldest first.
        """
        with self._lock:
            buf = list(self._dq)  # snapshot

        n = len(buf)
        if n == 0:
            return []

        frames: list[Any] = []
        # Walk from oldest-required to most-recent.
        # slot k (0-based from oldest) lives at index -(history_len - k) * stride
        for k in range(history_len):
            reverse_offset = (history_len - 1 - k) * stride
            raw_idx = n - 1 - reverse_offset
            # Clamp to the oldest available frame instead of returning None
            idx = max(0, raw_idx)
            frames.append(buf[idx])

        return frames  # length == history_len, oldest first
