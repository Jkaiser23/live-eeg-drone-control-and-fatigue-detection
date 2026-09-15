#!/usr/bin/env python3
"""Deterministic validation of SafetyMonitor.

Unlike fusion (a pure function), SafetyMonitor is stateful -- it tracks
consecutive-violation streaks across ticks. So these scenarios drive
`evaluate()` in a loop, tick by tick, and assert the exact action at each
step: HOVER only appears after `hover_after_ticks`, LAND only after
`land_after_ticks`, and a single good tick resets the streak to zero.

Run: python tests/test_safety.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralflight.fatigue.fusion import FatigueResult  # noqa: E402
from neuralflight.fatigue.safety import (  # noqa: E402
    SafetyAction,
    SafetyMonitor,
    SafetyThresholds,
)
from neuralflight.fatigue.shared_state import (  # noqa: E402
    FatigueStateSnapshot,
    ModalityReading,
)

NOW = 1000.0
FRESH = NOW - 0.1
STALE = NOW - 100.0

# Small thresholds so tests run fast and are easy to eyeball: hover after 2
# bad ticks, land after 4. Real deployment would use larger tick counts.
THRESHOLDS = SafetyThresholds(hover_after_ticks=2, land_after_ticks=4, link_lost_land_ticks=1)


def reading(score, quality, timestamp=FRESH):
    return ModalityReading(score=score, quality=quality, timestamp=timestamp)


def snap(eeg_score=0.2, eeg_quality=0.9, vision_score=0.2, vision_quality=0.9,
         eeg_ts=FRESH, vision_ts=FRESH):
    return FatigueStateSnapshot(
        eeg=reading(eeg_score, eeg_quality, eeg_ts),
        vision=reading(vision_score, vision_quality, vision_ts),
    )


def result(index=0.2, confidence=0.9, disagreement=0.0, mode="fused"):
    return FatigueResult(fatigue_index=index, confidence=confidence, disagreement=disagreement, mode=mode)


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def scenario_normal_stays_normal():
    """Healthy readings across several ticks -> always NORMAL."""
    monitor = SafetyMonitor(THRESHOLDS)
    for i in range(5):
        decision = monitor.evaluate(snap(), result(), now=NOW, drone_link_ok=True)
        check(f"normal[{i}]: action == NORMAL", decision.action == SafetyAction.NORMAL)


def scenario_link_lost_is_near_instant():
    """Drone link loss -> LAND on the very first bad tick (link_lost_land_ticks=1)."""
    monitor = SafetyMonitor(THRESHOLDS)
    decision = monitor.evaluate(snap(), result(), now=NOW, drone_link_ok=False)
    check("link_lost: LAND on first bad tick", decision.action == SafetyAction.LAND)
    check("link_lost: reason is drone_link_lost", decision.reason == "drone_link_lost")


def scenario_link_lost_preempts_everything_else():
    """Link lost wins even if fatigue/disagreement would also fire -- priority order matters."""
    monitor = SafetyMonitor(THRESHOLDS)
    bad_result = result(index=0.95, disagreement=0.9)  # would trigger high-fatigue AND disagreement
    decision = monitor.evaluate(snap(), bad_result, now=NOW, drone_link_ok=False)
    check("preempt: LAND (link) not disagreement/fatigue", decision.action == SafetyAction.LAND)
    check("preempt: reason is drone_link_lost, not fatigue/disagreement", decision.reason == "drone_link_lost")


def scenario_face_lost_debounces_then_escalates():
    """Vision unusable -> NORMAL, NORMAL, HOVER, HOVER, LAND (thresholds: hover=2, land=4)."""
    monitor = SafetyMonitor(THRESHOLDS)
    lost_snap = snap(vision_quality=0.0)  # vision present but below quality floor
    expected = [
        SafetyAction.NORMAL,  # streak=1
        SafetyAction.HOVER,   # streak=2 -> hover_after_ticks
        SafetyAction.HOVER,   # streak=3
        SafetyAction.LAND,    # streak=4 -> land_after_ticks
    ]
    for i, expected_action in enumerate(expected):
        decision = monitor.evaluate(lost_snap, result(), now=NOW, drone_link_ok=True)
        check(f"face_lost[tick {i+1}]: action == {expected_action.value}", decision.action == expected_action,
              f"got {decision.action.value}")


def scenario_face_lost_recovers_resets_streak():
    """A single good tick resets the streak -- no partial credit toward LAND."""
    monitor = SafetyMonitor(THRESHOLDS)
    lost_snap = snap(vision_quality=0.0)
    good_snap = snap()

    monitor.evaluate(lost_snap, result(), now=NOW, drone_link_ok=True)  # streak=1
    monitor.evaluate(lost_snap, result(), now=NOW, drone_link_ok=True)  # streak=2 -> HOVER
    monitor.evaluate(good_snap, result(), now=NOW, drone_link_ok=True)  # recovers, streak=0
    decision = monitor.evaluate(lost_snap, result(), now=NOW, drone_link_ok=True)  # streak=1 again
    check("recover: back to NORMAL after one bad tick post-recovery", decision.action == SafetyAction.NORMAL,
          f"got {decision.action.value}")


def scenario_sustained_high_fatigue_lands():
    """fatigue_index above hard limit, sustained -> escalates to LAND."""
    monitor = SafetyMonitor(THRESHOLDS)
    high_fatigue = result(index=0.9)
    for _ in range(3):  # streak 1,2,3 -> HOVER by tick 2
        decision = monitor.evaluate(snap(), high_fatigue, now=NOW, drone_link_ok=True)
    check("high_fatigue: HOVER by 3rd consecutive tick", decision.action == SafetyAction.HOVER)
    decision = monitor.evaluate(snap(), high_fatigue, now=NOW, drone_link_ok=True)  # streak=4
    check("high_fatigue: LAND by 4th consecutive tick", decision.action == SafetyAction.LAND)


def scenario_mild_fatigue_reduces_speed_no_debounce():
    """Mild fatigue (soft limit) reduces speed immediately -- no debounce, this is a graceful degrade."""
    monitor = SafetyMonitor(THRESHOLDS)
    mild = result(index=0.6)
    decision = monitor.evaluate(snap(), mild, now=NOW, drone_link_ok=True)
    check("mild_fatigue: REDUCE_SPEED on first tick", decision.action == SafetyAction.REDUCE_SPEED)


def scenario_disagreement_hovers_immediately():
    """Strong modality disagreement -> HOVER on the very first tick, no debounce."""
    monitor = SafetyMonitor(THRESHOLDS)
    conflicting = result(index=0.5, disagreement=0.8)
    decision = monitor.evaluate(snap(), conflicting, now=NOW, drone_link_ok=True)
    check("disagreement: HOVER on first tick", decision.action == SafetyAction.HOVER)
    check("disagreement: reason is modality_disagreement", decision.reason == "modality_disagreement")


def scenario_both_stale_uses_fusion_no_data_mode():
    """Both modalities stale -> fusion already reports mode='no_data'; safety escalates on that."""
    monitor = SafetyMonitor(THRESHOLDS)
    stale_snap = snap(eeg_ts=STALE, vision_ts=STALE)
    no_data_result = result(index=1.0, confidence=0.0, disagreement=None, mode="no_data")
    for _ in range(3):
        decision = monitor.evaluate(stale_snap, no_data_result, now=NOW, drone_link_ok=True)
    check("both_stale: HOVER by 3rd tick", decision.action == SafetyAction.HOVER)
    decision = monitor.evaluate(stale_snap, no_data_result, now=NOW, drone_link_ok=True)
    check("both_stale: LAND by 4th tick", decision.action == SafetyAction.LAND)


def scenario_face_lost_fires_even_in_eeg_only_mode():
    """Vision lost but EEG alone still 'usable' per fusion (mode=eeg_only) -- face-lost
    check must still fire, since it reads the raw snapshot, not fusion's mode."""
    monitor = SafetyMonitor(THRESHOLDS)
    vision_lost_snap = snap(vision_quality=0.0)
    eeg_only_result = result(index=0.3, confidence=0.5, disagreement=None, mode="eeg_only")
    for _ in range(3):
        decision = monitor.evaluate(vision_lost_snap, eeg_only_result, now=NOW, drone_link_ok=True)
    check("eeg_only_but_face_lost: HOVER by 3rd tick", decision.action == SafetyAction.HOVER)
    decision = monitor.evaluate(vision_lost_snap, eeg_only_result, now=NOW, drone_link_ok=True)
    check("eeg_only_but_face_lost: LAND by 4th tick", decision.action == SafetyAction.LAND)


def main() -> None:
    scenarios = [
        scenario_normal_stays_normal,
        scenario_link_lost_is_near_instant,
        scenario_link_lost_preempts_everything_else,
        scenario_face_lost_debounces_then_escalates,
        scenario_face_lost_recovers_resets_streak,
        scenario_sustained_high_fatigue_lands,
        scenario_mild_fatigue_reduces_speed_no_debounce,
        scenario_disagreement_hovers_immediately,
        scenario_both_stale_uses_fusion_no_data_mode,
        scenario_face_lost_fires_even_in_eeg_only_mode,
    ]
    print(f"Running {len(scenarios)} safety scenarios...\n")
    for scenario in scenarios:
        scenario()
    print("\nAll safety scenarios passed.")


if __name__ == "__main__":
    main()
