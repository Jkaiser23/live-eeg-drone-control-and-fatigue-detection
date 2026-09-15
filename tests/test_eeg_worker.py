#!/usr/bin/env python3
"""Deterministic validation of the pure EEG scoring functions.

These don't need a board -- they feed hand-built sine waves into
compute_theta_alpha_ratio / normalize_fatigue_score / check_channel_quality
and assert the math does what the docstring in eeg_worker.py claims.

Run: python tests/test_eeg_worker.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralflight.fatigue.eeg_worker import (  # noqa: E402
    check_channel_quality,
    compute_theta_alpha_ratio,
    normalize_fatigue_score,
)

SAMPLING_RATE = 250.0
DURATION_S = 4.0


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def sine_wave(freq_hz: float, amplitude: float = 20.0, noise_std: float = 0.5) -> np.ndarray:
    t = np.arange(0, DURATION_S, 1.0 / SAMPLING_RATE)
    signal = amplitude * np.sin(2 * np.pi * freq_hz * t)
    signal += np.random.default_rng(42).normal(0, noise_std, size=signal.shape)
    return signal


def scenario_alpha_dominant_gives_low_ratio():
    """A clean 10 Hz (alpha) signal should have theta/alpha << 1 -- 'alert' pattern."""
    window = sine_wave(10.0)
    ratio = compute_theta_alpha_ratio(window, SAMPLING_RATE)
    check("alpha_dominant: ratio is not None", ratio is not None)
    check("alpha_dominant: ratio < 0.5 (alpha power dominates)", ratio < 0.5, f"got {ratio:.3f}")


def scenario_theta_dominant_gives_high_ratio():
    """A clean 6 Hz (theta) signal should have theta/alpha >> 1 -- 'drowsy' pattern."""
    window = sine_wave(6.0)
    ratio = compute_theta_alpha_ratio(window, SAMPLING_RATE)
    check("theta_dominant: ratio is not None", ratio is not None)
    check("theta_dominant: ratio > 2.0 (theta power dominates)", ratio > 2.0, f"got {ratio:.3f}")


def scenario_too_short_window_returns_none():
    """A window with too few samples can't produce a usable PSD -- must return None, not crash."""
    window = np.random.default_rng(0).normal(0, 5, size=32)
    ratio = compute_theta_alpha_ratio(window, SAMPLING_RATE)
    check("too_short: returns None instead of raising", ratio is None)


def scenario_flat_zero_window_returns_none():
    """An all-zero window (dead channel) has zero alpha power -- must return None, not divide by zero."""
    window = np.zeros(int(SAMPLING_RATE * DURATION_S))
    ratio = compute_theta_alpha_ratio(window, SAMPLING_RATE)
    check("flat_zero: returns None instead of raising ZeroDivisionError", ratio is None)


def scenario_normalize_clamps_to_bounds():
    """Raw ratios far outside the calibrated baseline span must still clamp into [0, 1]."""
    below = normalize_fatigue_score(raw_ratio=0.0, baseline_low=1.0, baseline_high=3.0)
    above = normalize_fatigue_score(raw_ratio=10.0, baseline_low=1.0, baseline_high=3.0)
    mid = normalize_fatigue_score(raw_ratio=2.0, baseline_low=1.0, baseline_high=3.0)

    check("normalize: below-baseline clamps to 0.0", below == 0.0, f"got {below}")
    check("normalize: above-baseline clamps to 1.0", above == 1.0, f"got {above}")
    check("normalize: midpoint maps to 0.5", abs(mid - 0.5) < 1e-9, f"got {mid}")


def scenario_normalize_degenerate_baseline_is_safe():
    """baseline_high <= baseline_low is a bad calibration, not a crash -- falls back to 0.5."""
    result = normalize_fatigue_score(raw_ratio=2.0, baseline_low=3.0, baseline_high=1.0)
    check("degenerate_baseline: falls back to neutral 0.5", result == 0.5, f"got {result}")


def scenario_flatline_channel_fails_quality_check():
    """A near-constant channel (disconnected/railed electrode) must fail the quality check."""
    flat = np.full(1000, 512.3) + np.random.default_rng(1).normal(0, 0.01, size=1000)
    ok = check_channel_quality(flat, flatline_std_floor=1.0)
    check("flatline: fails quality check", ok is False)


def scenario_healthy_channel_passes_quality_check():
    """Normal-amplitude EEG noise should pass the quality check."""
    healthy = np.random.default_rng(2).normal(0, 15.0, size=1000)
    ok = check_channel_quality(healthy, flatline_std_floor=1.0)
    check("healthy: passes quality check", ok is True)


def scenario_too_short_window_config_raises_at_construction():
    """A window_seconds too short for the sampling rate must fail loudly at
    construction time, not silently return quality=0.0 forever at runtime."""
    from neuralflight.fatigue.eeg_worker import EEGWorkerConfig, _validate_window_length

    # 250 Hz needs nfft=512 samples => >= 2.048s. 2.0s is 12 samples short.
    bad_config = EEGWorkerConfig(
        board_id=0,
        serial_port="",
        eeg_channels=[1, 2],
        sampling_rate=250.0,
        window_seconds=2.0,
    )
    raised = False
    try:
        _validate_window_length(bad_config)
    except ValueError:
        raised = True
    check("too_short_window_config: raises ValueError at construction", raised)

    good_config = EEGWorkerConfig(
        board_id=0,
        serial_port="",
        eeg_channels=[1, 2],
        sampling_rate=250.0,
        window_seconds=3.0,
    )
    raised = False
    try:
        _validate_window_length(good_config)
    except ValueError:
        raised = True
    check("adequate_window_config: does not raise", not raised)


def main() -> None:
    scenarios = [
        scenario_alpha_dominant_gives_low_ratio,
        scenario_theta_dominant_gives_high_ratio,
        scenario_too_short_window_returns_none,
        scenario_flat_zero_window_returns_none,
        scenario_normalize_clamps_to_bounds,
        scenario_normalize_degenerate_baseline_is_safe,
        scenario_flatline_channel_fails_quality_check,
        scenario_healthy_channel_passes_quality_check,
        scenario_too_short_window_config_raises_at_construction,
    ]
    print(f"Running {len(scenarios)} EEG scoring scenarios...\n")
    for scenario in scenarios:
        scenario()
    print("\nAll EEG scoring scenarios passed.")


if __name__ == "__main__":
    main()
