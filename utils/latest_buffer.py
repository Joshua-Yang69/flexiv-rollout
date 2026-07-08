from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from threading import Condition, Lock
from typing import Any, Mapping

import numpy as np


class LatestBuffer:
    """Thread-safe latest-value buffer.

    The rollout pipeline should pass the most recent observation/action, not an
    unbounded queue. This keeps latency bounded when inference or hardware IO
    briefly falls behind.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._value: Any = None
        self._version = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def put(self, value: Any) -> int:
        with self._condition:
            self._value = value
            self._version += 1
            version = self._version
            self._condition.notify_all()
            return version

    def get(self) -> Any:
        with self._lock:
            return self._value

    def wait_next(self, last_version: int = 0, timeout: float | None = None) -> tuple[int, Any]:
        with self._condition:
            self._condition.wait_for(lambda: self._version > last_version, timeout=timeout)
            return self._version, self._value


@dataclass(frozen=True)
class ArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ArraySpec":
        return cls(
            name=str(data["name"]),
            shape=tuple(int(v) for v in data["shape"]),
            dtype=str(data["dtype"]),
        )


class SharedArrayBuffer:
    """Lock-free latest ndarray buffer using multiprocessing.shared_memory.

    The first uint64 is a sequence counter. Writers set it odd while copying and
    even after publishing. Readers retry until they observe a stable even
    sequence, which avoids returning torn frames without requiring a process
    shared lock.
    """

    _HEADER_DTYPE = np.dtype(np.uint64)

    def __init__(self, name: str, specs: list[ArraySpec], create: bool) -> None:
        self.name = name
        self.specs = specs
        self._arrays: dict[str, np.ndarray] = {}
        size = self._HEADER_DTYPE.itemsize
        for spec in specs:
            size += np.dtype(spec.dtype).itemsize * int(np.prod(spec.shape))

        self._shm = shared_memory.SharedMemory(name=name, create=create, size=size)
        self._seq = np.ndarray((1,), dtype=self._HEADER_DTYPE, buffer=self._shm.buf[: self._HEADER_DTYPE.itemsize])
        if create:
            self._seq[0] = 0

        offset = self._HEADER_DTYPE.itemsize
        for spec in specs:
            dtype = np.dtype(spec.dtype)
            nbytes = dtype.itemsize * int(np.prod(spec.shape))
            self._arrays[spec.name] = np.ndarray(
                spec.shape,
                dtype=dtype,
                buffer=self._shm.buf[offset : offset + nbytes],
            )
            offset += nbytes

    def write(self, values: Mapping[str, np.ndarray | float | int]) -> int:
        seq = int(self._seq[0])
        self._seq[0] = seq + 1
        for key, value in values.items():
            if key not in self._arrays:
                raise KeyError(f"Unknown shared buffer key: {key}")
            self._arrays[key][...] = value
        self._seq[0] = seq + 2
        return int(self._seq[0])

    def read(self, max_retries: int = 1000) -> tuple[int, dict[str, np.ndarray]]:
        for _ in range(max_retries):
            seq_start = int(self._seq[0])
            if seq_start % 2 == 1:
                continue
            data = {key: array.copy() for key, array in self._arrays.items()}
            seq_end = int(self._seq[0])
            if seq_start == seq_end and seq_end % 2 == 0:
                return seq_end, data
        raise TimeoutError("Could not read a stable shared buffer snapshot.")

    def close(self, unlink: bool = False) -> None:
        self._arrays.clear()
        self._shm.close()
        if unlink:
            self._shm.unlink()

