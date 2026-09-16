#!/usr/bin/env python3
"""Deterministic validation of CoDroneWorker's dispatch and queue logic.

Uses a fake `Drone`-shaped class (records calls, never touches hardware) so
these tests exercise OUR queue/dispatch/thread-safety logic, not the real
SDK's I/O. The real SDK's actual failure behavior (SystemExit on a missing
dongle) is validated separately in demos/fatigue_phase5_codrone_harness.py,
against the genuine codrone_edu package.

Run: python tests/test_codrone_adapter.py
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralflight.controllers.codrone_adapter import (  # noqa: E402
    CoDroneWorker,
    CoDroneWorkerConfig,
)


class FakeDrone:
    """Records every call instead of touching hardware. Thread-unsafe by
    design (like the real SDK) -- CoDroneWorker's job is to ensure only one
    thread ever calls into it."""

    def __init__(self):
        self.calls = []
        self.battery = 80.0
        self.connected = True
        self.fail_next_go = False

    def pair(self, port=None):
        self.calls.append(("pair", port))
        return True

    def isConnected(self):
        return self.connected

    def get_battery(self):
        return self.battery

    def takeoff(self):
        self.calls.append(("takeoff",))

    def land(self):
        self.calls.append(("land",))

    def emergency_stop(self):
        self.calls.append(("emergency_stop",))

    def hover(self, duration):
        self.calls.append(("hover", duration))

    def go(self, direction, power, duration):
        if self.fail_next_go:
            self.fail_next_go = False
            raise RuntimeError("simulated serial write failure")
        self.calls.append(("go", direction, power, duration))

    def turn_left(self):
        self.calls.append(("turn_left",))

    def turn_right(self):
        self.calls.append(("turn_right",))

    def close(self):
        self.calls.append(("close",))


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def make_worker(fake_drone, config=None):
    return CoDroneWorker(config=config or CoDroneWorkerConfig(heartbeat_interval_s=0.05), drone_factory=lambda: fake_drone)


def wait_for(predicate, timeout_s=2.0, interval_s=0.02):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def scenario_takeoff_and_move_dispatch_correctly():
    """Basic dispatch: takeoff, then a forward move, land up with expected SDK calls."""
    fake = FakeDrone()
    worker = make_worker(fake)
    worker.start()
    try:
        worker.takeoff()
        worker.send_command("forward", intensity=1.0)
        wait_for(lambda: any(c[0] == "go" for c in fake.calls))

        check("takeoff: called", ("takeoff",) in fake.calls)
        go_calls = [c for c in fake.calls if c[0] == "go"]
        check("move: exactly one go() call", len(go_calls) == 1, f"got {go_calls}")
        check("move: direction mapped forward->forward", go_calls[0][1] == "forward")
        check("move: full intensity -> full configured power", go_calls[0][2] == 50)
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def scenario_intensity_clamped_to_power_range():
    """Out-of-range intensity (e.g. from a bad upstream calc) must clamp, not crash or over-drive."""
    fake = FakeDrone()
    worker = make_worker(fake)
    worker.start()
    try:
        worker.send_command("forward", intensity=5.0)  # way out of [0,1]
        wait_for(lambda: any(c[0] == "go" for c in fake.calls))
        go_calls = [c for c in fake.calls if c[0] == "go"]
        check("clamp: power does not exceed configured max (50)", go_calls[0][2] == 50, f"got {go_calls}")

        worker.send_command("forward", intensity=-3.0)
        wait_for(lambda: len([c for c in fake.calls if c[0] == "go"]) >= 2)
        go_calls = [c for c in fake.calls if c[0] == "go"]
        check("clamp: negative intensity floors to 0 power", go_calls[1][2] == 0, f"got {go_calls}")
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def scenario_rotate_uses_turn_not_go():
    """rotate_left/rotate_right have no go() equivalent -- must route to turn_left/turn_right."""
    fake = FakeDrone()
    worker = make_worker(fake)
    worker.start()
    try:
        worker.send_command("rotate_left", intensity=1.0)
        wait_for(lambda: ("turn_left",) in fake.calls)
        check("rotate_left: routed to turn_left()", ("turn_left",) in fake.calls)
        check("rotate_left: did NOT go through go()", not any(c[0] == "go" for c in fake.calls))
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def scenario_land_preempts_queued_moves():
    """Queuing several moves then calling land() must drop the queued moves --
    land() must be the only thing that executes next, not a leftover move."""
    fake = FakeDrone()
    worker = make_worker(fake)
    # Don't start the thread yet -- queue commands while nothing is draining them,
    # to deterministically prove land() clears what's pending before anything runs.
    worker.send_command("forward", intensity=1.0)
    worker.send_command("strafe_right", intensity=1.0)
    worker.send_command("forward", intensity=1.0)
    worker.land()  # should drain the 3 queued moves and become the only queued item

    check("preempt: queue holds exactly 1 item (the land)", worker._queue.qsize() == 1,
          f"got qsize={worker._queue.qsize()}")

    worker.start()
    try:
        wait_for(lambda: ("land",) in fake.calls)
        go_calls = [c for c in fake.calls if c[0] == "go"]
        check("preempt: land() was called", ("land",) in fake.calls)
        check("preempt: none of the pre-queued moves executed", len(go_calls) == 0, f"got {go_calls}")
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def scenario_emergency_stop_preempts_pending_land():
    """emergency_stop() must preempt even an already-queued land()."""
    fake = FakeDrone()
    worker = make_worker(fake)
    worker.land()
    worker.emergency_stop()

    check("preempt_land: queue holds exactly 1 item (the emergency stop)", worker._queue.qsize() == 1)

    worker.start()
    try:
        wait_for(lambda: ("emergency_stop",) in fake.calls)
        check("preempt_land: emergency_stop() was called", ("emergency_stop",) in fake.calls)
        # land() is also called once more at shutdown as an extra fail-safe -- that's expected
        # and separate from whether the *queued* land ever executed pre-emptively.
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def scenario_command_failure_reports_link_down():
    """A raised exception from the SDK during dispatch must flip link_status to disconnected,
    not crash the worker thread."""
    fake = FakeDrone()
    fake.fail_next_go = True
    worker = make_worker(fake)
    worker.start()
    try:
        worker.send_command("forward", intensity=1.0)
        wait_for(lambda: worker.link_status().connected is False)
        status = worker.link_status()
        check("failure: link_status reports connected=False", status.connected is False)
        check("failure: worker thread stays alive (one bad command != dead worker)", worker.is_alive())
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def scenario_shutdown_always_lands_first():
    """Stopping the worker must attempt land() before close(), regardless of prior state."""
    fake = FakeDrone()
    worker = make_worker(fake)
    worker.start()
    wait_for(lambda: ("pair", None) in fake.calls)
    worker.stop()
    worker.join(timeout=2.0)

    check("shutdown: land() called during shutdown", ("land",) in fake.calls)
    check("shutdown: close() called after land()",
          fake.calls.index(("close",)) > fake.calls.index(("land",)))


def scenario_link_status_freshness():
    """link_status().is_ok() must require BOTH connected=True and a fresh timestamp."""
    fake = FakeDrone()
    worker = make_worker(fake)
    worker.start()
    try:
        wait_for(lambda: worker.link_status().connected is True)
        status = worker.link_status()
        now = time.monotonic()
        check("freshness: fresh + connected -> is_ok() True", status.is_ok(now, max_staleness_s=1.5))
        check("freshness: same status considered stale far in the future",
              not status.is_ok(now + 10.0, max_staleness_s=1.5))
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def main() -> None:
    scenarios = [
        scenario_takeoff_and_move_dispatch_correctly,
        scenario_intensity_clamped_to_power_range,
        scenario_rotate_uses_turn_not_go,
        scenario_land_preempts_queued_moves,
        scenario_emergency_stop_preempts_pending_land,
        scenario_command_failure_reports_link_down,
        scenario_shutdown_always_lands_first,
        scenario_link_status_freshness,
    ]
    print(f"Running {len(scenarios)} CoDroneWorker scenarios...\n")
    for scenario in scenarios:
        scenario()
    print("\nAll CoDroneWorker scenarios passed.")


if __name__ == "__main__":
    main()
