"""EEGNet model for motor imagery EEG classification.

Based on Lawhern et al. (2018), "EEGNet: A Compact Convolutional Network
for EEG-based Brain-Computer Interfaces".

This module contains the single EEGNet architecture used by the project.
It includes a residual connection around the separable-convolution block and
an optional channel-attention mechanism.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """Compact EEGNet for motor-imagery classification.

    Expected input shape is ``(batch, channels, time_samples)``.

    Architecture:
        1. Temporal convolution.
        2. Depthwise spatial convolution.
        3. Separable temporal convolution with a residual projection.
        4. Optional channel attention.
        5. Linear classification head.

    The default configuration matches the project's current EEG contract:
    17 channels, 480 samples, and 2 motor-imagery classes.
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_classes: int = 2,
        n_samples: int = 480,
        dropout: float = 0.5,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        use_attention: bool = False,
    ) -> None:
        """Initialize EEGNet.

        Args:
            n_channels: Number of EEG channels.
            n_classes: Number of output classes.
            n_samples: Number of time samples in each input window.
            dropout: Dropout probability used after convolutional blocks.
            kernel_length: Temporal-convolution kernel length.
            F1: Number of temporal filters.
            D: Depth multiplier for the spatial depthwise convolution.
            F2: Number of output filters in the separable-convolution block.
            use_attention: Enable channel attention before classification.
        """
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be greater than 0")
        if n_classes <= 0:
            raise ValueError("n_classes must be greater than 0")
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than 0")
        if kernel_length <= 0:
            raise ValueError("kernel_length must be greater than 0")
        if F1 <= 0 or D <= 0 or F2 <= 0:
            raise ValueError("F1, D, and F2 must be greater than 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dropout = dropout
        self.kernel_length = kernel_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.use_attention = use_attention

        # Block 1: temporal convolution.
        self.conv1 = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Block 2: depthwise spatial convolution.
        self.conv2 = nn.Conv2d(
            F1,
            F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Block 3: depthwise + pointwise separable convolution.
        # Padding of 7 keeps the temporal dimension unchanged for kernel 16.
        self.conv3 = nn.Conv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, 16),
            padding=(0, 7),
            groups=F1 * D,
            bias=False,
        )
        self.conv4 = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # Residual projection matches the separable block's channel count.
        self.residual_proj = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
        )

        # Optional channel attention.
        if use_attention:
            attention_hidden = max(1, F2 // 4)
            self.attention = nn.Sequential(
                nn.Linear(F2, attention_hidden),
                nn.ReLU(),
                nn.Linear(attention_hidden, F2),
                nn.Sigmoid(),
            )

        self._to_linear = self._get_conv_output_size()
        self.fc = nn.Linear(self._to_linear, n_classes)

    def _feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolutional feature extractor."""
        if x.ndim != 3:
            raise ValueError(
                "EEGNet expects input shape (batch, channels, time_samples); "
                f"received {tuple(x.shape)}"
            )

        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} EEG channels, received {x.shape[1]}"
            )

        if x.shape[2] != self.n_samples:
            raise ValueError(
                f"Expected {self.n_samples} time samples, received {x.shape[2]}"
            )

        # (batch, channels, time) -> (batch, 1, channels, time)
        x = x.unsqueeze(1)

        # Block 1: temporal filtering.
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Block 2: spatial filtering.
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Block 3: separable convolution with residual connection.
        identity = self.residual_proj(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)

        # Even-sized temporal kernels can introduce a one-sample mismatch
        # in some configurations. Crop only if needed.
        if x.shape[-1] != identity.shape[-1]:
            min_length = min(x.shape[-1], identity.shape[-1])
            x = x[..., :min_length]
            identity = identity[..., :min_length]

        x = F.elu(x + identity)
        x = self.pooling2(x)
        x = self.dropout2(x)

        if self.use_attention:
            # Global context -> one weight per F2 feature channel.
            context = x.mean(dim=(2, 3))
            attention_weights = self.attention(context)
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
            x = x * attention_weights

        return x

    def _get_conv_output_size(self) -> int:
        """Calculate flattened feature size for the configured input shape."""
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                x = torch.zeros(
                    1,
                    self.n_channels,
                    self.n_samples,
                )
                return self._feature_maps(x).numel()
        finally:
            self.train(was_training)

    def get_config(self) -> dict[str, Any]:
        """Return the complete architecture configuration for checkpoints."""
        return {
            "architecture": "EEGNet",
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "n_samples": self.n_samples,
            "dropout": self.dropout,
            "kernel_length": self.kernel_length,
            "F1": self.F1,
            "D": self.D,
            "F2": self.F2,
            "use_attention": self.use_attention,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for an EEG batch."""
        x = self._feature_maps(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class EEGClassifier:
    """Training, evaluation, checkpoint, and inference wrapper for EEGNet."""

    def __init__(self, model: EEGNet, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device

    def train_step(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Run one optimization step and return ``(loss, accuracy)``."""
        self.model.train()

        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device).long()

        optimizer.zero_grad()

        outputs = self.model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        predicted = outputs.argmax(dim=1)
        accuracy = (predicted == y_batch).float().mean().item()

        return loss.item(), accuracy

    def eval_step(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[float, float, Any]:
        """Evaluate one batch and return ``(loss, accuracy, predictions)``."""
        self.model.eval()

        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device).long()

        with torch.no_grad():
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)

            predicted = outputs.argmax(dim=1)
            accuracy = (predicted == y_batch).float().mean().item()

        return loss.item(), accuracy, predicted.cpu().numpy()

    def predict(self, X: torch.Tensor) -> tuple[Any, Any]:
        """Predict class indices and probabilities.

        Args:
            X: Tensor with shape ``(channels, time_samples)`` or
                ``(batch, channels, time_samples)``.

        Returns:
            Tuple ``(predicted_classes, probabilities)`` as NumPy arrays.
        """
        self.model.eval()

        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(
                "predict expects shape (channels, time_samples) or "
                "(batch, channels, time_samples)"
            )

        X = X.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            probabilities = F.softmax(outputs, dim=1)
            predicted = probabilities.argmax(dim=1)

        return predicted.cpu().numpy(), probabilities.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights and the complete architecture configuration."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.get_config(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load weights into an already-constructed model.

        The caller is responsible for constructing EEGNet with the
        checkpoint's ``model_config`` before calling this method.
        """
        checkpoint = torch.load(path, map_location=self.device)
        model_config = checkpoint.get("model_config", {})

        if model_config.get("architecture") not in (None, "EEGNet"):
            raise ValueError(
                f"Checkpoint architecture is "
                f"{model_config['architecture']!r}; "
                "expected 'EEGNet'"
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
