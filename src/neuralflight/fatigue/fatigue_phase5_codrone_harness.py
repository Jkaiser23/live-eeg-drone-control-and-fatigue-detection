#!/usr/bin/env python3
"""Phase 5 validation harness.

Wires the full chain against the REAL CoDroneWorker (genuine codrone_edu
Drone/pair() calls) in place of DroneSimulator:

  MockEEGWorker/MockVisionWorker -> SharedFatigueState -> compute_fatigue_index
  -> SafetyMonitor -> CoDroneWorker (real hardware link)

Unlike Phase 3's harness, `drone_link_ok` here comes from the REAL
CoDroneWorker.link_status(), not a scripted glitch -- if no CoDrone EDU
dongle is attached, this demonstrates the intended fail-safe behavior for
that exact situation: the link never comes up, so the safety chain should
issue an emergency stop from the very first tick and keep issuing it, rather than ever
allowing a takeoff/move command through.

Run: python -m neuralflight.fatigue.fatigue_phase5_codrone_harness
Stop: Ctrl+C
"""

import time

from neuralflight.controllers.codrone_adapter import CoDroneWorker, CoDroneWorkerConfig
from neuralflight.controllers.drone_controller import DroneController
from neuralflight.fatigue.fusion import compute_fatigue_index
from neuralflight.fatigue.mock_workers import MockEEGWorker, MockVisionWorker
from neuralflight.fatigue.safety import SafetyAction, SafetyMonitor
from neuralflight.fatigue.shared_state import SharedFatigueState

TICK_HZ = 10.0
TICK_S = 1.0 / TICK_HZ
MAX_LINK_STALENESS_S = 1.5


def main() -> None:
    codrone_worker = CoDroneWorker(CoDroneWorkerConfig())
    controller = DroneController(codrone_worker)
    codrone_worker.start()

    state = SharedFatigueState()
    safety = SafetyMonitor()

    eeg_worker = MockEEGWorker(state, interval_s=0.4, fail_probability=0.1)
    vision_worker = MockVisionWorker(state, interval_s=0.2)
    eeg_worker.start()
    vision_worker.start()

    print("Phase 5 harness running against the REAL CoDroneWorker.")
    print("If no CoDrone EDU dongle is attached, expect EMERGENCY_STOP from tick 1 onward --")
    print("that IS the correct fail-safe behavior for an unreachable drone.\n")

    controller.takeoff()  # queued -- CoDroneWorker will refuse to act meaningfully if never connected

    start_time = time.monotonic()
    try:
        running = True
        while running:
            tick_start = time.monotonic()
            elapsed = tick_start - start_time

            link_status = codrone_worker.link_status()
            drone_link_ok = link_status.is_ok(tick_start, MAX_LINK_STALENESS_S)

            snapshot = state.snapshot()
            fatigue_result = compute_fatigue_index(snapshot, now=tick_start)
            decision = safety.evaluate(snapshot, fatigue_result, now=tick_start, drone_link_ok=drone_link_ok)

            if decision.action == SafetyAction.EMERGENCY_STOP:
                controller.emergency_stop()
            elif decision.action == SafetyAction.LAND:
                controller.land()
            elif decision.action == SafetyAction.HOVER:
                controller.hover()
            elif decision.action == SafetyAction.REDUCE_SPEED:
                controller.move("forward", intensity=0.3)
            else:
                controller.move("forward", intensity=0.6)

            print(
                f"[t={elapsed:5.2f}s] link_ok={str(drone_link_ok):5s} "
                f"connected={str(link_status.connected):5s} "
                f"worker_alive={str(codrone_worker.is_alive()):5s} "
                f"-> {decision.action.value:12s} ({decision.reason})"
            )

            # Stop after a short bounded run in this harness -- see the module
            # docstring: a permanently-dead CoDroneWorker thread combined with
            # an unbounded loop would queue LAND commands forever with nothing
            # draining them. A production control loop should detect a dead
            # worker thread and stop issuing new commands entirely, not just
            # rely on this harness's timeout.
            running = elapsed < 8.0

            elapsed_tick = time.monotonic() - tick_start
            time.sleep(max(0.0, TICK_S - elapsed_tick))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        eeg_worker.stop()
        vision_worker.stop()
        codrone_worker.stop()
        eeg_worker.join(timeout=1.0)
        vision_worker.join(timeout=1.0)
        codrone_worker.join(timeout=3.0)
        print("Done.")


if __name__ == "__main__":
    main()
