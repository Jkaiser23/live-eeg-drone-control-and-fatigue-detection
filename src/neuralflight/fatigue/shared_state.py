"""Thread-safe fatigue state contract shared between sensor workers and the
flight control loop.

Contract
--------
fatigue_score : float, strictly clamped to [0.0, 1.0]
    0.0 = fully alert / fully awake signal
    1.0 = maximally fatigued signal (NOT "sensor lost" -- see `quality`)
quality : float, strictly clamped to [0.0, 1.0]
    Confidence/validity of the accompanying fatigue_score.
    0.0 means "ignore fatigue_score entirely -- treat as no data."
timestamp : float
    time.monotonic() at the moment this modality last attempted an update
    (including rejected/invalid attempts -- see _update()).

`fatigue_score` and `quality` are NEVER merged into one number. A worker that
cannot produce a valid reading must report quality=0.0, not fatigue_score=1.0.

Enforcement of the [0.0, 1.0] bound happens HERE, at the write boundary --
not in each worker -- so it is structurally impossible for an out-of-contract
value to reach the fusion engine, regardless of how a given worker is
implemented.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class FatigueScoreError(ValueError):
    """Raised internally when a value fails the [0.0, 1.0]-finite contract.

    Never escapes SharedFatigueState -- it is caught at the write boundary
    and converted into a quality=0.0 reading instead.
    """


def _clamp01(value) -> float:
    """Validate and clamp a value into [0.0, 1.0].

    Raises FatigueScoreError if the value is missing, non-numeric, NaN, or
    infinite. Values outside [0.0, 1.0] but otherwise valid are clamped, not
    rejected (e.g. a formula that slightly overshoots due to float error).
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FatigueScoreError(f"expected a finite number, got {value!r}")
    if math.isnan(value) or math.isinf(value):
        raise FatigueScoreError(f"expected a finite number, got {value!r}")
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ModalityReading:
    """Immutable snapshot of a single modality's last known reading."""

    score: Optional[float]  # None until the first successful update ever arrives
    quality: float  # 0.0 == treat score as unusable / no data
    timestamp: float  # time.monotonic() of last update attempt, 0.0 if never touched

    def is_fresh(self, now: float, max_staleness_s: float) -> bool:
        """True if this reading was touched recently enough to be trusted at all.

        Independent of `quality` -- this only measures whether the producing
        thread is still alive and updating, not whether its data is valid.
        """
        return self.timestamp > 0.0 and (now - self.timestamp) < max_staleness_s

    def is_usable(self, now: float, max_staleness_s: float, quality_floor: float) -> bool:
        """True if this reading is fresh, present, and above the quality floor."""
        return (
            self.score is not None
            and self.is_fresh(now, max_staleness_s)
            and self.quality >= quality_floor
        )


@dataclass(frozen=True)
class FatigueStateSnapshot:
    """Immutable point-in-time copy of both modality readings.

    Safe to read without holding any lock -- this is what `SharedFatigueState
    .snapshot()` returns, and what all downstream code (fusion, safety) is
    expected to operate on instead of touching SharedFatigueState directly.
    """

    eeg: ModalityReading
    vision: ModalityReading


_EMPTY_READING = ModalityReading(score=None, quality=0.0, timestamp=0.0)


class SharedFatigueState:
    """Thread-safe holder for the EEG and vision fatigue readings.

    Sensor worker threads call `update_eeg` / `update_vision`. The control
    loop calls `snapshot()` once per tick and works from the returned copy --
    it must never hold the lock while doing fusion or safety math.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._eeg: ModalityReading = _EMPTY_READING
        self._vision: ModalityReading = _EMPTY_READING

    def update_eeg(self, raw_score: Optional[float], raw_quality: float) -> None:
        """Publish a new EEG reading. Never raises."""
        self._update("_eeg", raw_score, raw_quality)

    def update_vision(self, raw_score: Optional[float], raw_quality: float) -> None:
        """Publish a new vision reading. Never raises."""
        self._update("_vision", raw_score, raw_quality)

    def _update(self, attr: str, raw_score: Optional[float], raw_quality: float) -> None:
        """Validate, clamp, and store a reading.

        On invalid input: log a warning, force quality to 0.0, and retain the
        previous score (history isn't erased, it's just marked unusable).
        timestamp always advances to "now", regardless of validity -- this is
        what lets staleness detection distinguish "thread died" (timestamp
        stalls) from "thread alive but producing bad data" (timestamp fresh,
        quality 0.0).
        """
        quality = 0.0
        score: Optional[float] = None
        valid = True

        try:
            quality = _clamp01(raw_quality)
        except FatigueScoreError as exc:
            logger.warning("%s: invalid quality discarded (%s); quality forced to 0.0", attr, exc)
            valid = False

        if valid and raw_score is not None:
            try:
                score = _clamp01(raw_score)
            except FatigueScoreError as exc:
                logger.warning("%s: invalid score discarded (%s); quality forced to 0.0", attr, exc)
                quality = 0.0
                score = None
                valid = False

        with self._lock:
            current: ModalityReading = getattr(self, attr)
            new_reading = ModalityReading(
                score=score if score is not None else current.score,
                quality=quality,
                timestamp=time.monotonic(),
            )
            setattr(self, attr, new_reading)

    def snapshot(self) -> FatigueStateSnapshot:
        """Return an immutable, lock-free-to-read copy of both readings."""
        with self._lock:
            return FatigueStateSnapshot(eeg=copy.copy(self._eeg), vision=copy.copy(self._vision))
