#!/usr/bin/env python3
"""Phase 3 validation harness.

Wires the full chain together against the real DroneSimulator (not a mock):

  MockEEGWorker/MockVisionWorker -> SharedFatigueState -> compute_fatigue_index
  -> SafetyMonitor -> DroneController (simulator)

No real drone commands beyond "run and print" are sent -- this is purely to
prove the control loop can read fatigue state, decide an action, and drive
the simulator each tick without threading issues, deadlocks, or the
simulator blocking sensor updates (or vice versa).

Two fault-injection phases are scripted so you can watch escalation live:
  - t=0-8s:  normal operation, occasional EEG NaN blips (from Phase 1/2)
  - t=8s:    drone "link" is forced down for a few ticks -> expect instant LAND
  - t=8s+:   vision worker goes silent -> expect HOVER then LAND (face-lost)

Run: python -m neuralflight.demos.fatigue_phase3_harness
Stop: Ctrl+C or close the simulator window
"""

import time

from neuralflight.controllers.drone_controller import DroneController
from neuralflight.fatigue.fusion import compute_fatigue_index
from neuralflight.fatigue.mock_workers import MockEEGWorker, MockVisionWorker
from neuralflight.fatigue.safety import SafetyAction, SafetyMonitor
from neuralflight.fatigue.shared_state import SharedFatigueState
from neuralflight.simulator.drone_sim import DroneSimulator
from neuralflight.utils.config_loader import load_config

TICK_HZ = 20.0
TICK_S = 1.0 / TICK_HZ

EEG_FAIL_PROBABILITY = 0.1
VISION_SILENT_AFTER_S = 8.0  # simulate face lost partway through the run
LINK_DOWN_START_S = 8.0  # simulate a dropped drone link at the same time, briefly
LINK_DOWN_DURATION_S = 0.3  # short glitch -- long enough to prove LAND is near-instant


def main() -> None:
    drone_config = load_config("drone_config")
    simulator = DroneSimulator(drone_config)
    controller = DroneController(simulator)
    controller.takeoff()

    state = SharedFatigueState()
    safety = SafetyMonitor()

    eeg_worker = MockEEGWorker(state, interval_s=0.4, fail_probability=EEG_FAIL_PROBABILITY)
    vision_worker = MockVisionWorker(state, interval_s=0.2, silent_after_s=VISION_SILENT_AFTER_S)
    eeg_worker.start()
    vision_worker.start()

    print("Phase 3 harness running against the real DroneSimulator.")
    print(f"  Vision worker goes silent at t={VISION_SILENT_AFTER_S:.0f}s (expect HOVER then LAND)")
    print(f"  Drone link forced down at t={LINK_DOWN_START_S:.0f}s for {LINK_DOWN_DURATION_S}s (expect instant LAND)")
    print("  Ctrl+C or close the window to stop.\n")

    start_time = time.monotonic()

    try:
        running = True
        while running:
            tick_start = time.monotonic()
            elapsed = tick_start - start_time

            drone_link_ok = not (
                LINK_DOWN_START_S <= elapsed < LINK_DOWN_START_S + LINK_DOWN_DURATION_S
            )

            snapshot = state.snapshot()
            fatigue_result = compute_fatigue_index(snapshot, now=tick_start)
            decision = safety.evaluate(snapshot, fatigue_result, now=tick_start, drone_link_ok=drone_link_ok)

            if decision.action == SafetyAction.LAND:
                controller.land()
            elif decision.action == SafetyAction.HOVER:
                controller.hover()
            elif decision.action == SafetyAction.REDUCE_SPEED:
                controller.move("forward", intensity=0.3)
            else:
                controller.move("forward", intensity=0.6)

            print(
                f"[t={elapsed:6.2f}s] link_ok={str(drone_link_ok):5s} "
                f"mode={fatigue_result.mode:11s} index={fatigue_result.fatigue_index:.2f} "
                f"-> {decision.action.value:12s} ({decision.reason})"
            )

            running = simulator.update()
            tick_elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, TICK_S - tick_elapsed))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        eeg_worker.stop()
        vision_worker.stop()
        eeg_worker.join(timeout=1.0)
        vision_worker.join(timeout=1.0)
        simulator.close()
        print("Done.")


if __name__ == "__main__":
    main()
