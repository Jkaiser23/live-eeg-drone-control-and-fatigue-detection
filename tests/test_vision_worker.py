#!/usr/bin/env python3
"""Deterministic validation of the pure vision scoring functions.

These don't need a camera or a real face -- they feed hand-built landmark
coordinates into eye_aspect_ratio() and hand-built (timestamp, ear) windows
into summarize_window(), and assert the math behaves as documented.

Run: python tests/test_vision_worker.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralflight.fatigue.vision_worker import (  # noqa: E402
    eye_aspect_ratio,
    summarize_window,
)


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def scenario_open_eye_has_high_ear():
    """A tall, open-eye-shaped hexagon should have a large EAR."""
    # p1, p4 are the horizontal corners (0.0, 0.5) and (1.0, 0.5).
    # p2, p3, p5, p6 form a tall lid gap (0.3 units above/below center).
    points = [(0.0, 0.5), (0.3, 0.2), (0.7, 0.2), (1.0, 0.5), (0.7, 0.8), (0.3, 0.8)]
    ear = eye_aspect_ratio(points)
    check("open_eye: EAR is well above the closed threshold (~0.3)", ear > 0.25, f"got {ear:.3f}")


def scenario_closed_eye_has_low_ear():
    """A nearly-flat hexagon (lids touching) should have EAR near zero."""
    points = [(0.0, 0.5), (0.3, 0.49), (0.7, 0.49), (1.0, 0.5), (0.7, 0.51), (0.3, 0.51)]
    ear = eye_aspect_ratio(points)
    check("closed_eye: EAR is near zero", ear < 0.05, f"got {ear:.3f}")


def scenario_degenerate_horizontal_distance_is_safe():
    """p1 == p4 (zero horizontal distance) must not raise a ZeroDivisionError."""
    points = [(0.5, 0.5), (0.3, 0.2), (0.7, 0.2), (0.5, 0.5), (0.7, 0.8), (0.3, 0.8)]
    ear = eye_aspect_ratio(points)
    check("degenerate: returns 0.0 instead of raising", ear == 0.0, f"got {ear}")


def scenario_all_eyes_open_gives_zero_perclos():
    """A window where every frame has EAR well above threshold -> PERCLOS 0.0."""
    history = [(float(i), 0.35) for i in range(20)]
    score, quality = summarize_window(history, ear_closed_threshold=0.21)
    check("all_open: score == 0.0", score == 0.0, f"got {score}")
    check("all_open: quality == 1.0 (every frame had a detection)", quality == 1.0, f"got {quality}")


def scenario_all_eyes_closed_gives_full_perclos():
    """A window where every frame has EAR below threshold -> PERCLOS 1.0."""
    history = [(float(i), 0.05) for i in range(20)]
    score, quality = summarize_window(history, ear_closed_threshold=0.21)
    check("all_closed: score == 1.0", score == 1.0, f"got {score}")


def scenario_mixed_perclos_and_partial_quality():
    """Some frames closed, some open, some no-detection -> PERCLOS over valid frames only,
    quality reflects detection rate over ALL frames (valid + missing)."""
    history = [
        (0.0, 0.35), (1.0, 0.35), (2.0, 0.05), (3.0, 0.05),  # 4 valid: 2 open, 2 closed
        (4.0, None), (5.0, None),  # 2 frames with no face detected
    ]
    score, quality = summarize_window(history, ear_closed_threshold=0.21)
    check("mixed: PERCLOS is 0.5 (2 of 4 valid frames closed)", score == 0.5, f"got {score}")
    check("mixed: quality is 4/6 (detection rate over all frames)", abs(quality - (4 / 6)) < 1e-9, f"got {quality}")


def scenario_no_face_ever_detected_gives_none_and_zero_quality():
    """Every frame has ear=None (face never found) -> (None, 0.0), never a crash."""
    history = [(float(i), None) for i in range(10)]
    score, quality = summarize_window(history, ear_closed_threshold=0.21)
    check("no_face: score is None (never claim a fatigue reading with no data)", score is None)
    check("no_face: quality is 0.0", quality == 0.0, f"got {quality}")


def scenario_empty_window_is_safe():
    """An empty history (worker just started) -> (None, 0.0), not an IndexError."""
    score, quality = summarize_window([], ear_closed_threshold=0.21)
    check("empty_window: score is None", score is None)
    check("empty_window: quality is 0.0", quality == 0.0)


def main() -> None:
    scenarios = [
        scenario_open_eye_has_high_ear,
        scenario_closed_eye_has_low_ear,
        scenario_degenerate_horizontal_distance_is_safe,
        scenario_all_eyes_open_gives_zero_perclos,
        scenario_all_eyes_closed_gives_full_perclos,
        scenario_mixed_perclos_and_partial_quality,
        scenario_no_face_ever_detected_gives_none_and_zero_quality,
        scenario_empty_window_is_safe,
    ]
    print(f"Running {len(scenarios)} vision scoring scenarios...\n")
    for scenario in scenarios:
        scenario()
    print("\nAll vision scoring scenarios passed.")


if __name__ == "__main__":
    main()
