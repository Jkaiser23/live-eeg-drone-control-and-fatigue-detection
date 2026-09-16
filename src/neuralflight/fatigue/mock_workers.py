"""Mock EEG/vision worker threads for validating the threading + state
contract in shared_state.py without any real hardware attached.

Each mock worker runs on its own daemon thread and periodically calls into a
SharedFatigueState update method, following the exact same calling contract
a real EEGWorker/VisionWorker will use later. Two fault-injection knobs let
you exercise the failure paths the contract is designed to handle:

- fail_probability: chance per tick of publishing a NaN score (simulates a
  worker producing a bad reading while still running normally).
- silent_after_s: if set, the worker stops calling update_* entirely after
  this many seconds (simulates a hung/dead sensor thread).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable, Optional

from neuralflight.fatigue.shared_state import SharedFatigueState

logger = logging.getLogger(__name__)


class _MockWorker(threading.Thread):
    """Shared plumbing for the mock EEG/vision workers."""

    def __init__(
        self,
        update_fn: Callable[[Optional[float], float], None],
        *,
        interval_s: float = 0.5,
        name: str = "MockWorker",
        fail_probability: float = 0.0,
        silent_after_s: Optional[float] = None,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._update_fn = update_fn
        self._interval_s = interval_s
        self._fail_probability = fail_probability
        self._silent_after_s = silent_after_s
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None

    def stop(self) -> None:
        """Signal the run loop to exit on its next wake-up."""
        self._stop_event.set()

    def run(self) -> None:
        self._start_time = time.monotonic()
        while not self._stop_event.is_set():
            if self._gone_silent():
                # Simulate a hung/dead sensor: stop publishing entirely.
                # timestamp in SharedFatigueState will go stale from here on.
                self._stop_event.wait(self._interval_s)
                continue

            try:
                self._publish_one_reading()
            except Exception:  # noqa: BLE001 -- a worker thread must never die silently
                logger.exception("%s: unexpected error during tick", self.name)

            self._stop_event.wait(self._interval_s)

    def _gone_silent(self) -> bool:
        if self._silent_after_s is None:
            return False
        return (time.monotonic() - self._start_time) > self._silent_after_s

    def _publish_one_reading(self) -> None:
        if random.random() < self._fail_probability:
            # Simulate a bad reading reaching the contract boundary --
            # SharedFatigueState is responsible for turning this into
            # quality=0.0 rather than crashing this thread.
            self._update_fn(float("nan"), 0.9)
        else:
            score = random.uniform(0.0, 1.0)
            quality = random.uniform(0.7, 1.0)
            self._update_fn(score, quality)


class MockEEGWorker(_MockWorker):
    def __init__(self, state: SharedFatigueState, **kwargs) -> None:
        super().__init__(state.update_eeg, name="MockEEGWorker", **kwargs)


class MockVisionWorker(_MockWorker):
    def __init__(self, state: SharedFatigueState, **kwargs) -> None:
        super().__init__(state.update_vision, name="MockVisionWorker", **kwargs)
