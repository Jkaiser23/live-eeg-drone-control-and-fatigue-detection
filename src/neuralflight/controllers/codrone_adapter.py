"""CoDroneWorker: sole owner of the CoDrone EDU link, running on its own thread.

Real drone I/O (pairing, serial writes) must never happen on the control
loop's thread -- a blocked or slow write to the drone must not stall fatigue
computation or safety evaluation. This module is the only place in the
codebase allowed to import/touch `codrone_edu.drone.Drone` directly.

IMPORTANT, found by testing directly against the real SDK with no dongle
attached: codrone_edu's `Drone.open()` (called by `pair()`/`connect()`) raises
`SystemExit`, not a normal exception, when it can't find the USB dongle. A
bare `except Exception` around a connection attempt will NOT catch this --
the thread dies silently and `link_status()` freezes at its last value
forever, which looks identical to "still connected" to anything reading it.
Every call into the SDK in this module is therefore wrapped in
`except (Exception, SystemExit)`. This is deliberate and specific to this
SDK's behavior, not a general practice -- catching SystemExit elsewhere in
the codebase would be wrong.

Command dispatch uses one queue with a "drain-then-push" urgent path: LAND
and EMERGENCY_STOP are never appended normally. `land()` / `emergency_stop()`
first clear whatever's pending in the queue, then push the urgent command --
so a stale "move_forward" enqueued before a safety decision can never
execute after LAND has been requested.

Known hardware latency bound: `codrone_edu`'s `move(duration)` blocks
synchronously for the full duration (confirmed by reading `_move_desktop`'s
`sendControlWhile` call). That means an urgent LAND can't preempt a command
that's already mid-flight on the wire -- only the queue, not the in-flight
write. Worst-case latency from `land()` to the drone actually receiving the
land command is therefore bounded by `default_move_duration_s`, which is why
that value should be kept short (a few hundred ms), not by anything in this
module's queue logic.

`link_status()` follows the same timestamped-freshness pattern as
SharedFatigueState: `connected=True` alone is not proof of a live link -- if
this worker's thread dies, that flag freezes at its last value, so callers
must also check `timestamp` freshness, exactly like ModalityReading.is_fresh().
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from codrone_edu.drone import Drone

    _CODRONE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when codrone-edu isn't installed
    _CODRONE_AVAILABLE = False

# codrone_edu's go() only accepts forward/backward/left/right/up/down (case-insensitive).
# rotate_left/rotate_right have no go() equivalent -- handled via turn_left()/turn_right().
_GO_DIRECTION_MAP = {
    "forward": "forward",
    "backward": "backward",
    "strafe_left": "left",
    "strafe_right": "right",
}


@dataclass(frozen=True)
class LinkStatus:
    """Timestamped connection health -- mirrors ModalityReading's contract.

    `connected` alone is not enough: a dead worker thread leaves `connected`
    frozen at whatever it last was. Callers must check `is_ok()`, which also
    requires the timestamp to be fresh.
    """

    connected: bool
    battery_percent: Optional[float]
    timestamp: float  # time.monotonic() of the last successful heartbeat/command ack, 0.0 if never set

    def is_ok(self, now: float, max_staleness_s: float) -> bool:
        return self.connected and self.timestamp > 0.0 and (now - self.timestamp) < max_staleness_s


_EMPTY_STATUS = LinkStatus(connected=False, battery_percent=None, timestamp=0.0)


@dataclass
class CoDroneWorkerConfig:
    port: Optional[str] = None  # None lets the SDK auto-detect the USB dongle
    heartbeat_interval_s: float = 0.3  # how often to poll isConnected()/battery between commands
    default_move_power: int = 50  # 0-100, codrone_edu's power scale
    default_move_duration_s: float = 0.3  # kept short deliberately -- see module docstring on LAND latency
    low_battery_percent: float = 15.0  # informational only -- surfaced via LinkStatus, not auto-landed here


class _CommandKind(Enum):
    MOVE = "move"
    TAKEOFF = "takeoff"
    LAND = "land"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class _Command:
    kind: _CommandKind
    direction: Optional[str] = None  # e.g. "forward", "hover", "rotate_left" -- only used for MOVE
    intensity: float = 1.0


class CoDroneWorker(threading.Thread):
    """Owns the CoDrone EDU connection; all hardware I/O happens on this thread.

    Public surface mirrors DroneSimulator so DroneController can wrap either
    interchangeably: `send_command(name, intensity)`, `takeoff()`, `land()`.
    `link_status()` is the addition the control loop needs for the
    `drone_link_ok` input to SafetyMonitor.evaluate().
    """

    def __init__(
        self,
        config: Optional[CoDroneWorkerConfig] = None,
        drone_factory: Optional[type] = None,
    ) -> None:
        super().__init__(name="CoDroneWorker", daemon=True)
        if not _CODRONE_AVAILABLE and drone_factory is None:
            raise ImportError("codrone-edu is not installed -- pip install codrone-edu")

        self._config = config or CoDroneWorkerConfig()
        # Defaults to the real SDK class; tests inject a fake stand-in here
        # instead of monkeypatching the module, so production code is untouched.
        self._drone_factory = drone_factory or Drone
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[_Command]" = queue.Queue()
        self._queue_lock = threading.Lock()  # guards the drain-then-push urgent sequence
        self._drone = None

        self._status_lock = threading.Lock()
        self._status: LinkStatus = _EMPTY_STATUS

    # ------------------------------------------------------------------
    # Public API -- safe to call from the control loop thread, never blocks on hardware I/O
    # ------------------------------------------------------------------

    def send_command(self, command: str, intensity: float = 1.0) -> None:
        """Enqueue a normal flight command. Never blocks."""
        self._queue.put(_Command(kind=_CommandKind.MOVE, direction=command, intensity=intensity))

    def takeoff(self) -> None:
        self._queue.put(_Command(kind=_CommandKind.TAKEOFF))

    def land(self) -> None:
        """Request a landing. Preempts: clears any pending commands first."""
        self._push_urgent(_Command(kind=_CommandKind.LAND))

    def emergency_stop(self) -> None:
        """Cut motors immediately. Preempts everything, including a pending LAND."""
        self._push_urgent(_Command(kind=_CommandKind.EMERGENCY_STOP))

    def link_status(self) -> LinkStatus:
        with self._status_lock:
            return self._status

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internals -- everything below touches `self._drone` and runs ONLY on this thread
    # ------------------------------------------------------------------

    def _push_urgent(self, command: _Command) -> None:
        """Atomically drop whatever's queued and make `command` next."""
        with self._queue_lock:
            self._drain_queue()
            self._queue.put(command)

    def _drain_queue(self) -> List[_Command]:
        drained = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return drained

    def run(self) -> None:
        cfg = self._config
        try:
            self._drone = self._drone_factory()
            connected = self._drone.pair(cfg.port)
        except (Exception, SystemExit):
            # See module docstring: pair()/open() raises SystemExit on a missing
            # dongle, not a normal Exception. Both must be caught here.
            logger.exception("CoDroneWorker: failed to connect; reporting link down")
            self._publish_status(connected=False, battery_percent=None)
            return

        if not connected:
            logger.error("CoDroneWorker: pair() returned False; reporting link down")
            self._publish_status(connected=False, battery_percent=None)
            return

        logger.info("CoDroneWorker: connected")
        self._publish_status(connected=True, battery_percent=self._safe_get_battery())

        last_heartbeat = time.monotonic()
        try:
            while not self._stop_event.is_set():
                try:
                    command = self._queue.get(timeout=cfg.heartbeat_interval_s)
                    self._dispatch(command)
                except queue.Empty:
                    pass  # nothing queued -- fall through to the heartbeat check below

                now = time.monotonic()
                if now - last_heartbeat >= cfg.heartbeat_interval_s:
                    self._heartbeat()
                    last_heartbeat = now
        finally:
            self._shutdown_drone()

    def _dispatch(self, command: _Command) -> None:
        cfg = self._config
        try:
            if command.kind == _CommandKind.TAKEOFF:
                self._drone.takeoff()
            elif command.kind == _CommandKind.LAND:
                self._drone.land()
            elif command.kind == _CommandKind.EMERGENCY_STOP:
                self._drone.emergency_stop()
            elif command.kind == _CommandKind.MOVE:
                self._dispatch_move(command)
            self._publish_status(connected=True, battery_percent=self._safe_get_battery())
        except (Exception, SystemExit):
            logger.exception("CoDroneWorker: command %s failed; reporting link down", command.kind)
            self._publish_status(connected=False, battery_percent=None)

    def _dispatch_move(self, command: _Command) -> None:
        cfg = self._config
        clamped_intensity = max(0.0, min(1.0, command.intensity))
        power = int(round(cfg.default_move_power * clamped_intensity))

        if command.direction == "hover":
            self._drone.hover(cfg.default_move_duration_s)
        elif command.direction == "rotate_left":
            self._drone.turn_left()
        elif command.direction == "rotate_right":
            self._drone.turn_right()
        elif command.direction in _GO_DIRECTION_MAP:
            self._drone.go(_GO_DIRECTION_MAP[command.direction], power, cfg.default_move_duration_s)
        else:
            logger.warning("CoDroneWorker: unrecognized direction %r, ignoring", command.direction)

    def _heartbeat(self) -> None:
        try:
            connected = bool(self._drone.isConnected())
            battery = self._safe_get_battery() if connected else None
            self._publish_status(connected=connected, battery_percent=battery)
        except (Exception, SystemExit):
            logger.exception("CoDroneWorker: heartbeat check failed; reporting link down")
            self._publish_status(connected=False, battery_percent=None)

    def _safe_get_battery(self) -> Optional[float]:
        try:
            return float(self._drone.get_battery())
        except (Exception, SystemExit):
            return None

    def _publish_status(self, connected: bool, battery_percent: Optional[float]) -> None:
        with self._status_lock:
            self._status = LinkStatus(
                connected=connected, battery_percent=battery_percent, timestamp=time.monotonic()
            )

    def _shutdown_drone(self) -> None:
        if self._drone is None:
            return
        try:
            # Extra fail-safe: always attempt a landing before disconnecting,
            # regardless of why the run loop is exiting.
            self._drone.land()
        except (Exception, SystemExit):
            logger.exception("CoDroneWorker: land-on-shutdown failed")
        try:
            self._drone.close()
        except (Exception, SystemExit):
            logger.exception("CoDroneWorker: error while closing connection")
