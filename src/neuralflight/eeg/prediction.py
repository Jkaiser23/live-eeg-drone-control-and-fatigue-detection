"""Live EEG motor-imagery preprocessing and checkpoint inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from neuralflight.models.eegnet import EEGClassifier, EEGNet


@dataclass(frozen=True)
class EEGInferenceConfig:
    """Inference settings captured from the trained checkpoint."""

    sampling_rate: int
    channels: tuple[str, ...]
    lowcut: float
    highcut: float
    filter_order: int
    n_samples: int


class EEGPredictor:
    """Load one EEGNet checkpoint and convert raw windows into predictions."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = checkpoint.get("model_config")
        if not model_config:
            raise ValueError("Checkpoint is missing model_config")
        required = {"architecture", "n_channels", "n_classes", "n_samples", "dropout", "kernel_length", "F1", "D", "F2", "use_attention"}
        missing = required.difference(model_config)
        if missing:
            raise ValueError(f"Checkpoint model_config is missing fields: {sorted(missing)}")
        if model_config["architecture"] != "EEGNet":
            raise ValueError(f"Unsupported checkpoint architecture: {model_config['architecture']!r}")
        if int(model_config["n_classes"]) != 2:
            raise ValueError("Live motor-imagery inference requires exactly 2 classes")

        data_metadata = checkpoint.get("data_metadata")
        if not data_metadata:
            raise ValueError("Checkpoint is missing data_metadata")
        self.config = EEGInferenceConfig(
            sampling_rate=int(data_metadata["sampling_rate"]),
            channels=tuple(data_metadata["channels"]),
            lowcut=float(data_metadata["lowcut"]),
            highcut=float(data_metadata["highcut"]),
            filter_order=int(data_metadata["filter_order"]),
            n_samples=int(model_config["n_samples"]),
        )
        if len(self.config.channels) != int(model_config["n_channels"]):
            raise ValueError("Checkpoint channel metadata does not match model n_channels")

        self.model = EEGNet(
            n_channels=int(model_config["n_channels"]), n_classes=int(model_config["n_classes"]),
            n_samples=int(model_config["n_samples"]), dropout=float(model_config["dropout"]),
            kernel_length=int(model_config["kernel_length"]), F1=int(model_config["F1"]),
            D=int(model_config["D"]), F2=int(model_config["F2"]), use_attention=bool(model_config["use_attention"]),
        )
        self.classifier = EEGClassifier(self.model, device)
        self.classifier.load(checkpoint_path)

        raw_mapping = checkpoint.get("label_metadata", {}).get("class_to_command", {})
        self.class_to_command = {int(k): str(v) for k, v in raw_mapping.items()}
        if self.class_to_command != {0: "strafe_left", 1: "strafe_right"}:
            raise ValueError("Checkpoint command mapping must be {0: 'strafe_left', 1: 'strafe_right'}")

    def preprocess(self, raw_window: np.ndarray) -> np.ndarray:
        """Band-pass filter a raw window and return float32 model input."""
        data = np.asarray(raw_window, dtype=np.float64)
        expected_shape = (len(self.config.channels), self.config.n_samples)
        if data.shape != expected_shape:
            raise ValueError(f"Expected raw EEG shape {expected_shape}, received {data.shape}")
        nyquist = 0.5 * self.config.sampling_rate
        if not 0 < self.config.lowcut < self.config.highcut < nyquist:
            raise ValueError("Invalid band-pass frequencies for the sampling rate")
        b, a = butter(self.config.filter_order, [self.config.lowcut / nyquist, self.config.highcut / nyquist], btype="band")
        return filtfilt(b, a, data, axis=-1).astype(np.float32)

    def predict(self, raw_window: np.ndarray) -> tuple[int, float, str]:
        """Return ``(class_id, confidence, command)`` for one raw window."""
        processed = self.preprocess(raw_window)
        predicted, probabilities = self.classifier.predict(torch.from_numpy(processed))
        class_id = int(predicted[0])
        confidence = float(probabilities[0, class_id])
        return class_id, confidence, self.class_to_command[class_id]
