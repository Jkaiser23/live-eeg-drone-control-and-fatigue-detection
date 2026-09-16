#!/usr/bin/env python3
"""Phase 1 validation harness.

No drone, no fusion, no safety logic -- this exists only to prove out
SharedFatigueState under real concurrency before anything is built on top
of it:

  1. Two threads (mock EEG + mock vision) hammer the shared state
     concurrently -- confirms no corruption/deadlock under the lock.
  2. The EEG mock occasionally emits NaN -- confirms invalid input is
     caught at the write boundary and surfaces as quality=0.0, never a
     crash, never an out-of-range value in the snapshot.
  3. The vision mock goes silent after a fixed delay -- confirms staleness
     (timestamp) is detectable independently of quality.

Run: python demos/fatigue_phase1_harness.py
Stop: Ctrl+C
"""

import sys
import time
from pathlib import Path

# Allow running directly from a repo checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralflight.fatigue.mock_workers import MockEEGWorker, MockVisionWorker  # noqa: E402
from neuralflight.fatigue.shared_state import SharedFatigueState  # noqa: E402

TICK_HZ = 5.0  # slow tick is fine here -- this harness is for eyeballing, not real-time control
TICK_S = 1.0 / TICK_HZ

MAX_STALENESS_S = 1.5
QUALITY_FLOOR = 0.3

# Fault-injection knobs -- tune to exercise each contract guarantee:
EEG_FAIL_PROBABILITY = 0.15  # ~15% of EEG ticks publish NaN
VISION_SILENT_AFTER_S = 10.0  # vision worker "dies" 10s in


def describe(name: str, reading, now: float) -> str:
    if reading.score is None:
        score_str = "  None"
    else:
        score_str = f"{reading.score:5.2f}"

    fresh = reading.is_fresh(now, MAX_STALENESS_S)
    usable = reading.is_usable(now, MAX_STALENESS_S, QUALITY_FLOOR)

    return (
        f"{name:7s} score={score_str} quality={reading.quality:4.2f} "
        f"fresh={str(fresh):5s} usable={str(usable):5s}"
    )


def main() -> None:
    state = SharedFatigueState()

    eeg_worker = MockEEGWorker(state, interval_s=0.4, fail_probability=EEG_FAIL_PROBABILITY)
    vision_worker = MockVisionWorker(state, interval_s=0.2, silent_after_s=VISION_SILENT_AFTER_S)
    eeg_worker.start()
    vision_worker.start()

    print("Phase 1 harness starting.")
    print(f"  EEG worker: {EEG_FAIL_PROBABILITY:.0%} chance per tick of publishing NaN")
    print(f"  Vision worker: goes silent after {VISION_SILENT_AFTER_S:.0f}s (simulates a dead thread)")
    print("  Watch for: quality drops to 0.0 on NaN ticks, 'fresh' flips False after vision goes silent.\n")

    try:
        while True:
            tick_start = time.monotonic()
            snapshot = state.snapshot()

            print(
                f"[t={tick_start:8.2f}] "
                f"{describe('EEG', snapshot.eeg, tick_start)}  |  "
                f"{describe('VISION', snapshot.vision, tick_start)}"
            )

            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, TICK_S - elapsed))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        eeg_worker.stop()
        vision_worker.stop()
        eeg_worker.join(timeout=1.0)
        vision_worker.join(timeout=1.0)
        print("Done.")


if __name__ == "__main__":
    main()
