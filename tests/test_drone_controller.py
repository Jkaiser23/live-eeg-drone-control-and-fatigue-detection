"""Regression tests for controller-level emergency-stop dispatch."""

from neuralflight.controllers.drone_controller import DroneController
from neuralflight.simulator.drone_sim import DroneSimulator, DroneState


class FakeDrone:
    def __init__(self) -> None:
        self.calls = []

    def send_command(self, command: str, intensity: float) -> None:
        self.calls.append(("send_command", command, intensity))

    def emergency_stop(self) -> None:
        self.calls.append(("emergency_stop",))


class LegacyFakeDrone:
    def __init__(self) -> None:
        self.calls = []

    def send_command(self, command: str, intensity: float) -> None:
        self.calls.append(("send_command", command, intensity))


def scenario_emergency_stop_reaches_backend_when_flight_state_is_stale() -> None:
    """A hard stop must not be gated by DroneController._is_flying."""
    drone = FakeDrone()
    controller = DroneController(drone)

    controller.emergency_stop()

    assert drone.calls == [("emergency_stop",)]
    assert controller.is_flying is False


def scenario_emergency_stop_has_a_safe_legacy_backend_fallback() -> None:
    """Backends without a hard-stop API still receive an unconditional land command."""
    drone = LegacyFakeDrone()
    controller = DroneController(drone)

    controller.emergency_stop()

    assert drone.calls == [("send_command", "land", 1.0)]
    assert controller.is_flying is False


def scenario_simulator_emergency_stop_zeros_motion_and_lands() -> None:
    """The simulator implements the same hard-stop interface as CoDroneWorker."""
    simulator = object.__new__(DroneSimulator)
    simulator.state = DroneState(x=1.0, y=2.0, velocity_x=3.0, velocity_y=-4.0, is_flying=True)
    simulator.current_command = ("forward", 1.0)

    simulator.emergency_stop()

    assert simulator.state.is_flying is False
    assert simulator.state.velocity_x == 0.0
    assert simulator.state.velocity_y == 0.0
    assert simulator.current_command == ("hover", 0.0)
