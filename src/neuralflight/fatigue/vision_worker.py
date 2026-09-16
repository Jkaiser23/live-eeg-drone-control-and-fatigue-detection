"""Real webcam-based fatigue worker using OpenCV + MediaPipe FaceMesh.

Runs on its own thread, samples webcam frames, tracks the Eye Aspect Ratio
(EAR; Soukupova & Cech, 2016) per frame, and publishes PERCLOS -- Percentage
of Eye Closure (Wierwille & Ellsworth, 1994) -- over a rolling time window
as the fatigue_score.

Why PERCLOS specifically: it is *defined* as "fraction of the last N seconds
with eyes closed," so unlike the EEG theta/alpha ratio it is naturally
bounded to [0.0, 1.0] with no separate per-user baseline calibration needed.
The only tunable is what EAR counts as "closed."

    fatigue_score = fraction of frames in the rolling window with EAR < ear_closed_threshold
    quality       = fraction of frames in the rolling window where a face was actually detected

If no face is detected anywhere in the current window, quality = 0.0 and the
last known score is retained -- consistent with the fatigue_score contract:
score alone never signals "no data," only quality does.

The pure math (`eye_aspect_ratio`, `summarize_window`) is independent of
OpenCV/MediaPipe so it can be unit-tested with hand-built landmark
coordinates, without a camera or a real face in frame.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

from neuralflight.fatigue.shared_state import SharedFatigueState

logger = logging.getLogger(__name__)

try:
    import cv2
    import mediapipe as mp

    _VISION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when cv2/mediapipe aren't installed
    _VISION_DEPS_AVAILABLE = False

# MediaPipe FaceMesh landmark indices for the 6-point EAR formula, in the
# (p1, p2, p3, p4, p5, p6) order Soukupova & Cech define: p1/p4 are the
# horizontal corners, p2/p3/p5/p6 are the upper/lower lid points.
# This is a widely-used community mapping for the 468-point mesh -- verify
# against your MediaPipe version if EAR values look unexpectedly off.
LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]


@dataclass
class VisionWorkerConfig:
    camera_index: int = 0
    ear_closed_threshold: float = 0.21  # below this, eye counted as closed (Soukupova & Cech default)
    window_seconds: float = 10.0  # PERCLOS rolling window
    update_interval_s: float = 0.5
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_consecutive_failures: int = 10
    max_history_len: int = 600


def _euclidean(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def eye_aspect_ratio(points: Sequence[Tuple[float, float]]) -> float:
    """Pure function: EAR from 6 (x, y) points in (p1..p6) order.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Higher EAR = eye more open. Degenerate (near-zero horizontal distance,
    e.g. a bad landmark read) returns 0.0 (treated as "closed") rather than
    raising a division error.
    """
    p1, p2, p3, p4, p5, p6 = points
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal <= 1e-9:
        return 0.0
    return vertical / (2.0 * horizontal)


def summarize_window(
    history: Sequence[Tuple[float, Optional[float]]], ear_closed_threshold: float
) -> Tuple[Optional[float], float]:
    """Pure function: (fatigue_score, quality) from a window of (timestamp, ear|None) samples.

    `ear` is None for frames where no face/landmarks were detected. Returns
    (None, 0.0) if the window is empty or no frame in it had a detected face.
    """
    if not history:
        return None, 0.0

    valid_ears = [ear for _, ear in history if ear is not None]
    quality = len(valid_ears) / len(history)

    if not valid_ears:
        return None, 0.0

    closed_count = sum(1 for ear in valid_ears if ear < ear_closed_threshold)
    perclos = closed_count / len(valid_ears)
    return perclos, quality


class VisionWorker(threading.Thread):
    """Polls a webcam on its own thread and publishes a PERCLOS fatigue score."""

    def __init__(self, state: SharedFatigueState, config: VisionWorkerConfig) -> None:
        super().__init__(name="VisionWorker", daemon=True)
        if not _VISION_DEPS_AVAILABLE:
            raise ImportError("opencv-python and mediapipe are required -- pip install opencv-python mediapipe")
        if config.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least 1")
        if config.max_history_len < 1:
            raise ValueError("max_history_len must be at least 1")

        self._state = state
        self._config = config
        self._stop_event = threading.Event()
        self._history: Deque[Tuple[float, Optional[float]]] = deque()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        cfg = self._config
        cap = cv2.VideoCapture(cfg.camera_index)
        if not cap.isOpened():
            logger.error("VisionWorker: could not open camera index %s; publishing quality=0.0", cfg.camera_index)
            self._state.update_vision(None, 0.0)
            return

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

        last_publish = time.monotonic()
        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                now = time.monotonic()

                if not ret or frame is None:
                    consecutive_failures += 1
                    self._record_sample(now, None)
                    last_publish = self._publish_if_due(now, last_publish)

                    if consecutive_failures == cfg.max_consecutive_failures:
                        logger.error(
                            "VisionWorker: camera unresponsive after %d reads; publishing quality=0.0",
                            consecutive_failures,
                        )
                        self._state.update_vision(None, 0.0)

                    # Event.wait, unlike sleep, lets stop() interrupt a backoff promptly.
                    backoff_s = min(0.5, 0.05 * consecutive_failures)
                    self._stop_event.wait(backoff_s)
                    continue

                consecutive_failures = 0
                self._record_sample(now, self._safe_process_frame(frame, face_mesh))
                last_publish = self._publish_if_due(now, last_publish)
        finally:
            cap.release()
            face_mesh.close()

    def _record_sample(self, now: float, ear: Optional[float]) -> None:
        """Store one frame result while enforcing both time and count bounds."""
        self._history.append((now, ear))
        self._trim_window(now)
        while len(self._history) > self._config.max_history_len:
            self._history.popleft()

    def _publish_if_due(self, now: float, last_publish: float) -> float:
        """Publish the current rolling-window summary at the configured cadence."""
        if now - last_publish < self._config.update_interval_s:
            return last_publish
        score, quality = summarize_window(list(self._history), self._config.ear_closed_threshold)
        self._state.update_vision(score, quality)
        return now

    def _trim_window(self, now: float) -> None:
        cutoff = now - self._config.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _safe_process_frame(self, frame, face_mesh) -> Optional[float]:
        """Never lets a single bad frame crash the worker thread."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            left_points = [(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_EAR_INDICES]
            right_points = [(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_EAR_INDICES]

            left_ear = eye_aspect_ratio(left_points)
            right_ear = eye_aspect_ratio(right_points)
            return (left_ear + right_ear) / 2.0
        except Exception:
            logger.exception("VisionWorker: frame processing failed; treating as no-detection")
            return None
