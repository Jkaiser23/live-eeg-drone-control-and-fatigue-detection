#!/usr/bin/env python3
"""Phase 2 validation harness.

Extends the Phase 1 harness by running the raw mock-worker readings through
compute_fatigue_index() every tick, so you can watch the fusion mode/index/
confidence react live as the mock workers agree, disagree, degrade, or go
silent -- on top of the scripted, exact-number scenarios in
tests/test_fusion.py.

Run: python demos/fatigue_phase2_harness.py
Stop: Ctrl+C
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralflight.fatigue.fusion import compute_fatigue_index  # noqa: E402
from neuralflight.fatigue.mock_workers import MockEEGWorker, MockVisionWorker  # noqa: E402
from neuralflight.fatigue.shared_state import SharedFatigueState  # noqa: E402

TICK_HZ = 5.0
TICK_S = 1.0 / TICK_HZ

# Fault-injection knobs -- same idea as Phase 1, now watched through fusion:
EEG_FAIL_PROBABILITY = 0.15
VISION_SILENT_AFTER_S = 10.0


def main() -> None:
    state = SharedFatigueState()

    eeg_worker = MockEEGWorker(state, interval_s=0.4, fail_probability=EEG_FAIL_PROBABILITY)
    vision_worker = MockVisionWorker(state, interval_s=0.2, silent_after_s=VISION_SILENT_AFTER_S)
    eeg_worker.start()
    vision_worker.start()

    print("Phase 2 harness starting.")
    print(f"  EEG worker: {EEG_FAIL_PROBABILITY:.0%} chance per tick of publishing NaN")
    print(f"  Vision worker: goes silent after {VISION_SILENT_AFTER_S:.0f}s")
    print("  Watch for: mode flips fused -> eeg_only/vision_only -> no_data as data degrades,")
    print("  and confidence dropping sharply whenever disagreement is high.\n")

    try:
        while True:
            tick_start = time.monotonic()
            snapshot = state.snapshot()
            result = compute_fatigue_index(snapshot, now=tick_start)

            disagreement_str = f"{result.disagreement:.2f}" if result.disagreement is not None else " n/a"
            print(
                f"[t={tick_start:8.2f}] mode={result.mode:11s} "
                f"index={result.fatigue_index:.2f} confidence={result.confidence:.2f} "
                f"disagreement={disagreement_str}"
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
