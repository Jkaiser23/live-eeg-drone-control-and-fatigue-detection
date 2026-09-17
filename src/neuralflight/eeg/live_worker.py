"""Live BrainFlow worker for EEG motor-imagery inference."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from neuralflight.eeg.prediction import EEGPredictor

logger = logging.getLogger(__name__)

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    _BRAINFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BRAINFLOW_AVAILABLE = False


@dataclass(frozen=True)
class LiveEEGConfig:
    """BrainFlow acquisition settings for motor-imagery inference."""
    board_id: int
    serial_port: str
    eeg_channels: tuple[int, ...]
    sampling_rate: int
    window_samples: int
    update_interval_s: float = 0.25


class LiveEEGWorker(threading.Thread):
    """Acquire EEG on a background thread and publish the latest prediction."""

    def __init__(self, predictor: EEGPredictor, config: LiveEEGConfig) -> None:
        super().__init__(name="LiveEEGWorker", daemon=True)
        if not _BRAINFLOW_AVAILABLE:
            raise ImportError("brainflow is not installed -- pip install brainflow")
        if len(config.eeg_channels) != len(predictor.config.channels):
            raise ValueError("BrainFlow channel count must match checkpoint channel count")
        if config.sampling_rate != predictor.config.sampling_rate:
            raise ValueError("BrainFlow sampling rate must match checkpoint sampling rate")
        if config.window_samples != predictor.config.n_samples:
            raise ValueError("BrainFlow window_samples must match checkpoint n_samples")
        if config.update_interval_s <= 0:
            raise ValueError("update_interval_s must be greater than 0")
        self.predictor = predictor
        self.config = config
        self._stop_event = threading.Event()
        self._board: Optional["BoardShim"] = None
        self._lock = threading.Lock()
        self._latest = None
        self._last_error: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()

    def get_latest(self):
        with self._lock:
            return self._latest

    def get_last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def run(self) -> None:
        params = BrainFlowInputParams()
        params.serial_port = self.config.serial_port
        try:
            self._board = BoardShim(self.config.board_id, params)
            self._board.prepare_session()
            self._board.start_stream()
            logger.info("LiveEEGWorker: board session started")
        except Exception as exc:
            self._record_error(exc)
            logger.exception("LiveEEGWorker: failed to start board session")
            return
        try:
            while not self._stop_event.is_set():
                self._tick()
                self._stop_event.wait(self.config.update_interval_s)
        finally:
            self._shutdown_board()

    def _tick(self) -> None:
        try:
            data = self._board.get_current_board_data(self.config.window_samples)
            if data.shape[1] < self.config.window_samples:
                return
            raw_window = data[list(self.config.eeg_channels), -self.config.window_samples:]
            prediction = self.predictor.predict(raw_window)
            with self._lock:
                self._latest = prediction
                self._last_error = None
        except Exception as exc:
            self._record_error(exc)
            logger.exception("LiveEEGWorker: inference tick failed")

    def _record_error(self, exc: Exception) -> None:
        with self._lock:
            self._last_error = str(exc)

    def _shutdown_board(self) -> None:
        if self._board is None:
            return
        try:
            self._board.stop_stream()
            self._board.release_session()
            logger.info("LiveEEGWorker: board session released")
        except Exception:
            logger.exception("LiveEEGWorker: error while releasing board session")
