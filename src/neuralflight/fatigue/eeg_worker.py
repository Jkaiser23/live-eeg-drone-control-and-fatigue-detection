"""Real EEG fatigue worker using BrainFlow (targets OpenBCI Cyton + Daisy).

Runs on its own thread, polls the board at a fixed interval, computes a
theta/alpha band-power ratio over a sliding window per channel, normalizes
the average across channels onto the project's [0.0, 1.0] fatigue_score
contract, and publishes it (with a quality flag) into SharedFatigueState.

Scoring formula
----------------
    raw_ratio(channel) = theta_power(4-8 Hz) / alpha_power(8-13 Hz), via Welch PSD
    raw_ratio           = mean over configured channels
    fatigue_score        = clamp((raw_ratio - baseline_low) / (baseline_high - baseline_low), 0, 1)

    0.0 anchor: raw_ratio at a rested/alert calibration baseline
    1.0 anchor: raw_ratio at the calibration session's max observed drowsiness

`baseline_low`/`baseline_high` are per-user calibration values (see
EEGWorkerConfig) -- theta/alpha ratio at "alert" varies enough between
people that a fixed global constant would misclassify most users.

Quality is forced to 0.0 whenever:
    - fewer than `min_samples` are available in the current window (board
      just started, or an acquisition hiccup)
    - any configured channel looks railed/flatlined (std below a floor --
      a strong indicator of a disconnected or saturated electrode)
    - the BrainFlow call itself raises (bad serial link, board disconnect)

The pure scoring functions (`compute_theta_alpha_ratio`, `normalize_fatigue_score`,
`check_channel_quality`) are deliberately free of BrainFlow I/O so they can be
unit-tested with synthetic arrays, independent of any board being attached.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from neuralflight.fatigue.shared_state import SharedFatigueState

logger = logging.getLogger(__name__)

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations

    _BRAINFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when brainflow isn't installed
    _BRAINFLOW_AVAILABLE = False


@dataclass
class EEGWorkerConfig:
    """Everything needed to poll a board and score its output.

    `eeg_channels` are indices into the board's data array (NOT raw pin
    numbers) -- use BoardShim.get_eeg_channels(board_id) to discover them
    rather than hardcoding, since channel layout differs across boards.
    """

    board_id: int
    serial_port: str
    eeg_channels: List[int]
    sampling_rate: float
    window_seconds: float = 3.0
    update_interval_s: float = 0.5
    min_samples: int = 256
    flatline_std_floor: float = 1.0  # uV; below this, channel treated as railed/disconnected
    baseline_low: float = 1.0  # theta/alpha ratio at rested baseline (calibrate per user)
    baseline_high: float = 3.0  # theta/alpha ratio at max observed drowsiness (calibrate per user)
    board_params: "BrainFlowInputParams" = field(default=None)  # type: ignore[assignment]


def compute_theta_alpha_ratio(window: np.ndarray, sampling_rate: float) -> Optional[float]:
    """Pure function: theta/alpha band-power ratio for one channel's window.

    Returns None if the window is too short/degenerate to compute a Welch
    PSD from, so callers can treat that as "no valid reading" instead of
    crashing the worker thread.
    """
    if window.size < 64:
        return None

    data = window.copy().astype(np.float64)  # DataFilter.detrend mutates in place
    DataFilter.detrend(data, DetrendOperations.LINEAR.value)

    nfft = DataFilter.get_nearest_power_of_two(int(sampling_rate * 2))
    if data.size < nfft:
        return None

    try:
        psd = DataFilter.get_psd_welch(
            data, nfft, nfft // 2, int(sampling_rate), WindowOperations.HANNING.value
        )
        theta_power = DataFilter.get_band_power(psd, 4.0, 8.0)
        alpha_power = DataFilter.get_band_power(psd, 8.0, 13.0)
    except Exception:
        logger.exception("theta/alpha computation failed on this window")
        return None

    if alpha_power <= 1e-9:
        return None
    return theta_power / alpha_power


def normalize_fatigue_score(raw_ratio: float, baseline_low: float, baseline_high: float) -> float:
    """Map a raw theta/alpha ratio onto the project's [0.0, 1.0] fatigue_score scale.

    Note: SharedFatigueState clamps again at the write boundary regardless --
    this clamp exists so the worker's own logging/debugging reflects the
    same bounded value that ultimately gets published.
    """
    span = baseline_high - baseline_low
    if span <= 0:
        # Degenerate calibration (misconfigured baselines) -- fall back to a
        # neutral midpoint rather than dividing by zero or crashing.
        return 0.5
    normalized = (raw_ratio - baseline_low) / span
    return max(0.0, min(1.0, normalized))


def check_channel_quality(window: np.ndarray, flatline_std_floor: float) -> bool:
    """True if the channel does not look railed/flatlined."""
    if window.size == 0:
        return False
    return bool(np.std(window) >= flatline_std_floor)


def _validate_window_length(config: "EEGWorkerConfig") -> None:
    """Fail loudly at construction if the window can't produce a valid PSD.

    compute_theta_alpha_ratio() needs at least `nfft` samples, where nfft is
    BrainFlow's nearest-power-of-two above `sampling_rate * 2`. If
    `window_seconds * sampling_rate` comes in under that, every single tick
    will silently return None/quality=0.0 forever -- which is indistinguishable
    from "board disconnected" at the SharedFatigueState level. That failure
    mode is a misconfiguration, not a runtime data problem, so it should
    raise here rather than surface as a mysterious permanent quality=0.0.
    """
    nfft = DataFilter.get_nearest_power_of_two(int(config.sampling_rate * 2))
    window_samples = int(config.window_seconds * config.sampling_rate)
    if window_samples < nfft:
        min_seconds = nfft / config.sampling_rate
        raise ValueError(
            f"EEGWorkerConfig.window_seconds={config.window_seconds} is too short for "
            f"sampling_rate={config.sampling_rate}: needs >= {min_seconds:.3f}s "
            f"({nfft} samples) to compute a Welch PSD. Every tick would otherwise "
            f"silently report quality=0.0, indistinguishable from a disconnected board."
        )


class EEGWorker(threading.Thread):
    """Polls a BrainFlow board on its own thread and publishes a fatigue score."""

    def __init__(self, state: SharedFatigueState, config: EEGWorkerConfig) -> None:
        super().__init__(name="EEGWorker", daemon=True)
        if not _BRAINFLOW_AVAILABLE:
            raise ImportError("brainflow is not installed -- pip install brainflow")

        _validate_window_length(config)

        self._state = state
        self._config = config
        self._stop_event = threading.Event()
        self._board: Optional["BoardShim"] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        cfg = self._config
        params = cfg.board_params or BrainFlowInputParams()
        params.serial_port = cfg.serial_port

        try:
            self._board = BoardShim(cfg.board_id, params)
            self._board.prepare_session()
            self._board.start_stream()
            logger.info("EEGWorker: board session started (board_id=%s)", cfg.board_id)
        except Exception:
            logger.exception("EEGWorker: failed to start board session; publishing quality=0.0 and stopping")
            self._state.update_eeg(None, 0.0)
            return

        try:
            while not self._stop_event.is_set():
                self._tick()
                self._stop_event.wait(cfg.update_interval_s)
        finally:
            self._shutdown_board()

    def _tick(self) -> None:
        cfg = self._config
        try:
            n_samples = int(cfg.window_seconds * cfg.sampling_rate)
            data = self._board.get_current_board_data(n_samples)
        except Exception:
            logger.exception("EEGWorker: get_current_board_data failed")
            self._state.update_eeg(None, 0.0)
            return

        if data.shape[1] < cfg.min_samples:
            # Board just started or acquisition hiccup -- not an error, just not enough data yet.
            self._state.update_eeg(None, 0.0)
            return

        ratios = []
        for ch in cfg.eeg_channels:
            if ch >= data.shape[0]:
                continue
            channel_window = data[ch]

            if not check_channel_quality(channel_window, cfg.flatline_std_floor):
                continue  # this channel looks railed/disconnected -- skip it, don't fail the whole tick

            ratio = compute_theta_alpha_ratio(channel_window, cfg.sampling_rate)
            if ratio is not None:
                ratios.append(ratio)

        if not ratios:
            # Every configured channel was either railed or produced no valid PSD.
            self._state.update_eeg(None, 0.0)
            return

        raw_ratio = float(np.mean(ratios))
        score = normalize_fatigue_score(raw_ratio, cfg.baseline_low, cfg.baseline_high)

        # Quality scales with how many of the configured channels were usable --
        # partial channel loss degrades confidence rather than an all-or-nothing flag.
        quality = len(ratios) / max(1, len(cfg.eeg_channels))
        self._state.update_eeg(score, quality)

    def _shutdown_board(self) -> None:
        if self._board is None:
            return
        try:
            self._board.stop_stream()
            self._board.release_session()
            logger.info("EEGWorker: board session released")
        except Exception:
            logger.exception("EEGWorker: error while releasing board session")
