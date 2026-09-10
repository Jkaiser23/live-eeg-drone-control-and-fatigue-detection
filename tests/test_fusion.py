#!/usr/bin/env python3
"""Deterministic validation of compute_fatigue_index().

Unlike the Phase 1 harness (random mock workers, eyeballed over time),
fusion is a pure function -- so it's validated with exact, scripted
scenarios instead. Each scenario builds a FatigueStateSnapshot by hand and
asserts the exact branch and numbers the fusion policy should produce.

Run: python tests/test_fusion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralflight.fatigue.fusion import compute_fatigue_index  # noqa: E402
from neuralflight.fatigue.shared_state import (  # noqa: E402
    FatigueStateSnapshot,
    ModalityReading,
)

NOW = 1000.0  # arbitrary fixed "current time" for all scenarios
FRESH = NOW - 0.1  # well within any reasonable max_staleness_s
STALE = NOW - 100.0  # far outside any reasonable max_staleness_s


def reading(score, quality, timestamp=FRESH):
    return ModalityReading(score=score, quality=quality, timestamp=timestamp)


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def scenario_both_agree_high_quality():
    """Both modalities usable and closely agree -> fused, high confidence."""
    snap = FatigueStateSnapshot(eeg=reading(0.8, 0.9), vision=reading(0.82, 0.9))
    result = compute_fatigue_index(snap, now=NOW)

    check("both_agree: mode == fused", result.mode == "fused")
    check("both_agree: index close to inputs", abs(result.fatigue_index - 0.81) < 1e-6)
    check("both_agree: disagreement is small", result.disagreement < 0.05)
    check("both_agree: confidence is high (>0.8)", result.confidence > 0.8, f"got {result.confidence:.3f}")


def scenario_both_disagree_strongly():
    """Both modalities usable but wildly disagree -> fused, but confidence tanks."""
    snap = FatigueStateSnapshot(eeg=reading(0.05, 0.95), vision=reading(0.95, 0.95))
    result = compute_fatigue_index(snap, now=NOW)

    check("disagree: mode == fused", result.mode == "fused")
    check("disagree: disagreement is large", result.disagreement > 0.85, f"got {result.disagreement:.3f}")
    check(
        "disagree: confidence collapses despite high individual quality",
        result.confidence < 0.15,
        f"got {result.confidence:.3f}",
    )


def scenario_eeg_only():
    """Vision unusable (low quality) -> falls back to EEG alone, penalized confidence."""
    snap = FatigueStateSnapshot(eeg=reading(0.7, 0.9), vision=reading(0.3, 0.1))
    result = compute_fatigue_index(snap, now=NOW, quality_floor=0.3)

    check("eeg_only: mode == eeg_only", result.mode == "eeg_only")
    check("eeg_only: index equals eeg score", result.fatigue_index == 0.7)
    check("eeg_only: disagreement is None (single modality)", result.disagreement is None)
    check(
        "eeg_only: confidence is penalized vs raw eeg quality",
        result.confidence < 0.9,
        f"got {result.confidence:.3f}",
    )


def scenario_vision_only_face_lost():
    """EEG stale (worker died) -> falls back to vision alone."""
    snap = FatigueStateSnapshot(eeg=reading(0.6, 0.9, timestamp=STALE), vision=reading(0.4, 0.85))
    result = compute_fatigue_index(snap, now=NOW)

    check("vision_only: mode == vision_only", result.mode == "vision_only")
    check("vision_only: index equals vision score", result.fatigue_index == 0.4)


def scenario_both_lost():
    """Both stale/low-quality -> fail toward max fatigue index, zero confidence."""
    snap = FatigueStateSnapshot(eeg=reading(0.1, 0.9, timestamp=STALE), vision=reading(0.2, 0.05))
    result = compute_fatigue_index(snap, now=NOW)

    check("both_lost: mode == no_data", result.mode == "no_data")
    check("both_lost: fatigue_index pinned to worst case (1.0)", result.fatigue_index == 1.0)
    check("both_lost: confidence pinned to 0.0", result.confidence == 0.0)


def scenario_never_updated():
    """A modality that has literally never published (score=None) is never usable."""
    empty = ModalityReading(score=None, quality=0.0, timestamp=0.0)
    snap = FatigueStateSnapshot(eeg=empty, vision=reading(0.5, 0.9))
    result = compute_fatigue_index(snap, now=NOW)

    check("never_updated: mode == vision_only", result.mode == "vision_only")


def scenario_weight_skew():
    """Weighting EEG much higher than vision should pull the fused index toward EEG."""
    snap = FatigueStateSnapshot(eeg=reading(0.9, 0.9), vision=reading(0.1, 0.9))
    result = compute_fatigue_index(snap, now=NOW, weight_eeg=0.9, weight_vision=0.1)

    check("weight_skew: mode == fused", result.mode == "fused")
    check(
        "weight_skew: index pulled toward the heavily-weighted modality",
        result.fatigue_index > 0.75,
        f"got {result.fatigue_index:.3f}",
    )


def scenario_degenerate_zero_weights():
    """Both weights zero is a bad config, not a crash -- falls back to an even split."""
    snap = FatigueStateSnapshot(eeg=reading(0.4, 0.9), vision=reading(0.8, 0.9))
    result = compute_fatigue_index(snap, now=NOW, weight_eeg=0.0, weight_vision=0.0)

    check("zero_weights: does not raise / index is a valid blend", 0.0 <= result.fatigue_index <= 1.0)
    check("zero_weights: falls back to even split (~0.6)", abs(result.fatigue_index - 0.6) < 1e-6)


def main() -> None:
    scenarios = [
        scenario_both_agree_high_quality,
        scenario_both_disagree_strongly,
        scenario_eeg_only,
        scenario_vision_only_face_lost,
        scenario_both_lost,
        scenario_never_updated,
        scenario_weight_skew,
        scenario_degenerate_zero_weights,
    ]
    print(f"Running {len(scenarios)} fusion scenarios...\n")
    for scenario in scenarios:
        scenario()
    print("\nAll fusion scenarios passed.")


if __name__ == "__main__":
    main()
