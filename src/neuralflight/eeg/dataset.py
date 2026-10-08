"""PhysioNet Motor Imagery dataset loading and EEG preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mne
import numpy as np
from scipy.signal import butter, filtfilt


DEFAULT_RUNS = (4, 8, 12)
DEFAULT_CHANNELS = (
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
)


class PhysioNetDataset:
    """Load PhysioNet hand-motor-imagery EEG into the model-ready contract.

    The contract returned by :meth:`load_subject` is:

        X: ``(epochs, channels, samples)`` float32 EEG data
        y: ``(epochs,)`` int64 labels

    Labels are normalized to:

        0 -> T1 -> left-hand motor imagery
        1 -> T2 -> right-hand motor imagery

    Only runs 4, 8, and 12 are used by default. Each run is loaded and
    epoched independently so event labels cannot be accidentally mixed with
    T1/T2 events from a different task.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/physionet",
        sampling_rate: int = 160,
        channels: Sequence[str] = DEFAULT_CHANNELS,
        tmin: float = 0.0,
        window_seconds: float = 3.0,
        lowcut: float = 8.0,
        highcut: float = 30.0,
        filter_order: int = 5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sampling_rate = int(sampling_rate)
        self.channels = list(channels)
        self.tmin = float(tmin)
        self.window_seconds = float(window_seconds)
        self.lowcut = float(lowcut)
        self.highcut = float(highcut)
        self.filter_order = int(filter_order)

        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be greater than 0")
        if not self.channels:
            raise ValueError("channels must not be empty")
        if len(self.channels) != len(set(self.channels)):
            raise ValueError("channels must be unique")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0")

        self.n_samples = round(self.window_seconds * self.sampling_rate)
        if self.n_samples <= 0:
            raise ValueError("window_seconds produces zero samples")

        self.event_id = {
            "T1": 1,
            "T2": 2,
        }

    def download_subject(
        self,
        subject_id: int,
        runs: Sequence[int] | None = None,
    ) -> bool:
        """Download the requested runs for one subject."""
        runs = list(DEFAULT_RUNS if runs is None else runs)
        if not runs:
            raise ValueError("runs must not be empty")

        print(f"Downloading subject {subject_id:03d}...")

        try:
            for run in runs:
                mne.datasets.eegbci.load_data(
                    subject_id,
                    runs=[run],
                    path=str(self.data_dir.parent),
                )
            return True
        except Exception as exc:
            print(f"Error downloading subject {subject_id}: {exc}")
            return False

    def load_subject(
        self,
        subject_id: int,
        runs: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load, epoch, filter, and validate one subject.

        Args:
            subject_id: PhysioNet subject number.
            runs: Motor-imagery runs. Defaults to ``[4, 8, 12]``.
            channels: EEG channels. Defaults to the project's 17-channel
                motor-imagery configuration.

        Returns:
            ``(X, y)`` where X has shape ``(epochs, channels, samples)`` and
            y contains only labels 0 and 1.
        """
        runs = list(DEFAULT_RUNS if runs is None else runs)
        channels = list(self.channels if channels is None else channels)

        if not runs:
            raise ValueError("runs must not be empty")
        if not channels:
            raise ValueError("channels must not be empty")
        if len(channels) != len(set(channels)):
            raise ValueError("channels must be unique")

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        # MNE's tmax is inclusive. Use exactly n_samples / fs seconds of
        # samples by ending one sample before the nominal window boundary.
        epoch_tmax = (
            self.tmin
            + self.n_samples / self.sampling_rate
            - 1.0 / self.sampling_rate
        )

        for run in runs:
            raw_files = mne.datasets.eegbci.load_data(
                subject_id,
                runs=[run],
                path=str(self.data_dir.parent),
            )

            if len(raw_files) != 1:
                raise ValueError(
                    f"Expected one EDF for subject {subject_id}, run {run}; "
                    f"received {len(raw_files)}"
                )

            raw = mne.io.read_raw_edf(
                raw_files[0],
                preload=True,
                verbose=False,
            )
            mne.datasets.eegbci.standardize(raw)

            missing_channels = [
                channel for channel in channels if channel not in raw.ch_names
            ]
            if missing_channels:
                raise ValueError(
                    f"Missing requested channels in run {run}: "
                    f"{missing_channels}"
                )

            raw.pick_channels(channels, ordered=True)

            # Process each requested run independently. This preserves the
            # run/task boundary and prevents T1/T2 codes from unrelated runs
            # being interpreted as hand imagery.
            events, event_id = mne.events_from_annotations(
                raw,
                verbose=False,
            )

            selected_events = {
                name: event_id[name]
                for name in ("T1", "T2")
                if name in event_id
            }

            if set(selected_events) != {"T1", "T2"}:
                raise ValueError(
                    f"Run {run} must contain both T1 and T2 motor-imagery "
                    f"events; found {sorted(selected_events)}"
                )

            epochs = mne.Epochs(
                raw,
                events,
                event_id=selected_events,
                tmin=self.tmin,
                tmax=epoch_tmax,
                baseline=None,
                preload=True,
                verbose=False,
            )

            X_run = epochs.get_data()
            y_run = epochs.events[:, -1].astype(np.int64) - 1

            expected_run_shape = (len(channels), self.n_samples)
            if X_run.ndim != 3 or X_run.shape[1:] != expected_run_shape:
                raise ValueError(
                    f"Unexpected epoch shape for run {run}: {X_run.shape}; "
                    f"expected (epochs, {expected_run_shape[0]}, "
                    f"{expected_run_shape[1]})"
                )

            if not set(np.unique(y_run).tolist()).issubset({0, 1}):
                raise ValueError(f"Run {run} produced labels outside 0/1")

            X_parts.append(X_run)
            y_parts.append(y_run)

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)

        X = preprocess_eeg(
            X,
            lowcut=self.lowcut,
            highcut=self.highcut,
            fs=self.sampling_rate,
            filter_order=self.filter_order,
        ).astype(np.float32, copy=False)

        self._validate_contract(X, y, channels)
        return X, y

    def _validate_contract(
        self,
        X: np.ndarray,
        y: np.ndarray,
        channels: Sequence[str],
    ) -> None:
        """Validate the final model-facing EEG contract."""
        expected_shape = (len(channels), self.n_samples)

        if X.ndim != 3 or X.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected EEG shape (epochs, {expected_shape[0]}, "
                f"{expected_shape[1]}), received {X.shape}"
            )

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                f"X/y length mismatch: X has {len(X)} epochs, "
                f"y has shape {y.shape}"
            )

        labels = set(np.unique(y).tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(f"Labels must be 0/1, received {sorted(labels)}")
        if labels != {0, 1}:
            raise ValueError(
                f"Dataset must contain both motor-imagery classes; "
                f"received {sorted(labels)}"
            )

    def download_multiple_subjects(
        self,
        subject_ids: Sequence[int],
        runs: Sequence[int] | None = None,
    ) -> int:
        """Download requested runs for multiple subjects."""
        success_count = 0
        for subject_id in subject_ids:
            if self.download_subject(subject_id, runs):
                success_count += 1
        return success_count


def preprocess_eeg(
    X: np.ndarray,
    lowcut: float = 8.0,
    highcut: float = 30.0,
    fs: float = 160.0,
    filter_order: int = 5,
) -> np.ndarray:
    """Apply the project's zero-phase EEG bandpass filter.

    Args:
        X: EEG data with shape ``(epochs, channels, samples)``.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        filter_order: Butterworth filter order.

    Returns:
        Filtered EEG data with the same shape as the input.
    """
    X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(
            "X must have shape (epochs, channels, samples); "
            f"received {X.shape}"
        )

    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError(
            f"Invalid bandpass frequencies: lowcut={lowcut}, "
            f"highcut={highcut}, fs={fs}"
        )

    if filter_order <= 0:
        raise ValueError("filter_order must be greater than 0")

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(filter_order, [low, high], btype="bandpass")

    return filtfilt(b, a, X, axis=-1)
