#!/usr/bin/env python3
"""Phase 4 EEG integration harness.

Runs the REAL EEGWorker (actual BoardShim.prepare_session/start_stream/
get_current_board_data calls, actual DataFilter Welch PSD) against
BrainFlow's built-in synthetic board -- so this proves the BrainFlow API
usage and the worker's threading/lifecycle are correct, without requiring
a physical Cyton+Daisy to be plugged in.

Swapping to real hardware later is a config change, not a code change:
set board_id=BoardIds.CYTON_DAISY_BOARD and serial_port to the actual
device path.

Run: python demos/fatigue_phase4_eeg_harness.py
Stop: Ctrl+C
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brainflow.board_shim import BoardIds, BoardShim  # noqa: E402

from neuralflight.fatigue.eeg_worker import EEGWorker, EEGWorkerConfig  # noqa: E402
from neuralflight.fatigue.shared_state import SharedFatigueState  # noqa: E402

RUN_SECONDS = 12.0
POLL_INTERVAL_S = 1.0


def main() -> None:
    board_id = BoardIds.SYNTHETIC_BOARD
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)[:4]  # pretend these are our motor-cortex subset

    config = EEGWorkerConfig(
        board_id=board_id,
        serial_port="",  # synthetic board ignores this
        eeg_channels=eeg_channels,
        sampling_rate=sampling_rate,
        window_seconds=3.0,
        update_interval_s=0.5,
        min_samples=128,
        baseline_low=0.5,
        baseline_high=2.0,
    )

    state = SharedFatigueState()
    worker = EEGWorker(state, config)

    print(f"Starting EEGWorker against BrainFlow SYNTHETIC_BOARD (sampling_rate={sampling_rate} Hz)")
    print(f"Channels used: {eeg_channels}")
    print(f"Running for {RUN_SECONDS:.0f}s, polling shared state every {POLL_INTERVAL_S:.1f}s.\n")

    worker.start()

    start = time.monotonic()
    try:
        while time.monotonic() - start < RUN_SECONDS:
            time.sleep(POLL_INTERVAL_S)
            snap = state.snapshot()
            elapsed = time.monotonic() - start
            reading = snap.eeg
            score_str = f"{reading.score:.3f}" if reading.score is not None else " None"
            print(
                f"[t={elapsed:5.1f}s] eeg_score={score_str} quality={reading.quality:.2f} "
                f"fresh={reading.is_fresh(time.monotonic(), 1.5)}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        worker.stop()
        worker.join(timeout=3.0)
        print("Done.")


if __name__ == "__main__":
    main()
