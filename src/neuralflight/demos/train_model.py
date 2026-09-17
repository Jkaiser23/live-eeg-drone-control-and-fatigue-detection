#!/usr/bin/env python3
"""Train the canonical two-class EEGNet motor-imagery classifier.

The training contract is:
    class 0 -> left-hand motor imagery -> strafe_left
    class 1 -> right-hand motor imagery -> strafe_right

The saved checkpoint contains the complete EEGNet architecture configuration
plus training and dataset metadata needed to reproduce the model.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neuralflight.eeg.dataset import PhysioNetDataset, preprocess_eeg
from neuralflight.models.eegnet import EEGClassifier, EEGNet
from neuralflight.utils.config_loader import get_project_root, load_config


CLASS_TO_COMMAND = {
    0: "strafe_left",
    1: "strafe_right",
}


def _validate_two_class_data(X, y, split_name: str, expected_channels: int):
    """Validate the model-facing EEG data contract."""
    if X is None or y is None or len(X) == 0:
        raise ValueError(f"{split_name}: no EEG data was loaded")

    if X.ndim != 3:
        raise ValueError(
            f"{split_name}: expected X shape (epochs, channels, samples), "
            f"received {X.shape}"
        )

    if X.shape[1] != expected_channels:
        raise ValueError(
            f"{split_name}: expected {expected_channels} EEG channels, "
            f"received {X.shape[1]}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{split_name}: X/y length mismatch ({len(X)} != {len(y)})"
        )

    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{split_name}: labels must be only 0 and 1; received {labels}"
        )


def _load_split(dataset, subject_ids, runs, channels, preprocess_config, split_name):
    """Load one subject-level split and map its two events to 0/1."""
    X_parts, y_parts = [], []

    for subject_id in tqdm(subject_ids, desc=f"{split_name} subjects"):
        try:
            dataset.download_subject(subject_id, runs)
            X, y = dataset.load_subject(subject_id, runs, channels)

            if X is None or len(X) == 0:
                print(f"  Subject {subject_id}: no data returned")
                continue

            unique_events = np.unique(y)
            if len(unique_events) != 2:
                print(
                    f"  Subject {subject_id}: expected exactly 2 events, "
                    f"got {unique_events}; skipping"
                )
                continue

            event_map = {1: 0, 2: 1}
            if not set(unique_events).issubset(event_map):
                print(
                    f"  Subject {subject_id}: unexpected event IDs "
                    f"{unique_events}; skipping"
                )
                continue

            y = np.asarray([event_map[int(label)] for label in y], dtype=np.int64)

            X = preprocess_eeg(
                X,
                lowcut=preprocess_config["lowcut"],
                highcut=preprocess_config["highcut"],
                fs=preprocess_config["sampling_rate"],
            )

            X_parts.append(X)
            y_parts.append(y)
            print(f"  Subject {subject_id}: ✓ {len(X)} epochs")

        except Exception as exc:
            print(f"  Subject {subject_id}: ✗ {exc}")

    if not X_parts:
        raise ValueError(f"No usable {split_name.lower()} subjects were loaded")

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0), len(X_parts)


def prepare_data(config: dict):
    """Download and prepare subject-level train/validation data."""
    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)

    dataset_config = config["dataset"]
    preprocess_config = config["preprocessing"]
    model_config = config["model"]

    if model_config["num_classes"] != 2:
        raise ValueError(
            "eeg_config.yaml must define exactly 2 model classes for motor imagery"
        )

    train_subjects = dataset_config["train_subjects"]
    val_subjects = dataset_config["val_subjects"]
    runs = dataset_config["runs"]
    channels = preprocess_config["channels"]

    print("\nSubject split:")
    print(f"  Training subjects: {train_subjects}")
    print(f"  Validation subjects: {val_subjects}")
    print(f"  Runs per subject: {runs}")
    print(f"  EEG channels: {len(channels)}")
    print("  Classes: 0=left hand, 1=right hand")

    data_dir = get_project_root() / "data" / "raw" / "physionet"
    dataset = PhysioNetDataset(str(data_dir))

    print(f"\n{'=' * 60}")
    print("LOADING TRAINING SUBJECTS")
    print("=" * 60)
    X_train, y_train, train_count = _load_split(
        dataset, train_subjects, runs, channels, preprocess_config, "Training"
    )

    print(f"\n{'=' * 60}")
    print("LOADING VALIDATION SUBJECTS")
    print("=" * 60)
    X_val, y_val, val_count = _load_split(
        dataset, val_subjects, runs, channels, preprocess_config, "Validation"
    )

    expected_channels = model_config["input_channels"]
    _validate_two_class_data(X_train, y_train, "Training", expected_channels)
    _validate_two_class_data(X_val, y_val, "Validation", expected_channels)

    expected_samples = int(
        round(
            (config["epochs"]["tmax"] - config["epochs"]["tmin"])
            * preprocess_config["sampling_rate"]
        )
    )

    if X_train.shape[2] != expected_samples or X_val.shape[2] != expected_samples:
        raise ValueError(
            f"Epoch length mismatch: expected {expected_samples} samples; "
            f"train={X_train.shape[2]}, val={X_val.shape[2]}"
        )

    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise ValueError("Training data must contain both classes 0 and 1")
    if set(np.unique(y_val).tolist()) != {0, 1}:
        raise ValueError("Validation data must contain both classes 0 and 1")

    print(f"\n{'=' * 60}")
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Training subjects loaded: {train_count}")
    print(f"Training epochs: {len(X_train)}")
    print(f"Training shape: {X_train.shape}")
    print(f"Training class distribution: {np.bincount(y_train, minlength=2)}")
    print(f"Validation subjects loaded: {val_count}")
    print(f"Validation epochs: {len(X_val)}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Validation class distribution: {np.bincount(y_val, minlength=2)}")

    return X_train, y_train, X_val, y_val


def _build_checkpoint_metadata(config, epoch, val_loss, val_acc, device):
    """Build reproducibility metadata for the saved checkpoint."""
    dataset_config = config["dataset"]
    preprocess_config = config["preprocessing"]
    model_config = config["model"]
    training_config = config["training"]
    epochs_config = config["epochs"]

    return {
        "training_metadata": {
            "epoch": epoch,
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_acc),
            "device": str(device),
            "batch_size": int(training_config["batch_size"]),
            "learning_rate": float(training_config["learning_rate"]),
            "num_epochs": int(training_config["num_epochs"]),
            "early_stopping_patience": int(training_config["early_stopping_patience"]),
        },
        "data_metadata": {
            "dataset": dataset_config["name"],
            "runs": list(dataset_config["runs"]),
            "train_subjects": list(dataset_config["train_subjects"]),
            "val_subjects": list(dataset_config["val_subjects"]),
            "channels": list(preprocess_config["channels"]),
            "sampling_rate": int(preprocess_config["sampling_rate"]),
            "lowcut": float(preprocess_config["lowcut"]),
            "highcut": float(preprocess_config["highcut"]),
            "filter_order": int(preprocess_config["filter_order"]),
            "notch_freq": float(preprocess_config["notch_freq"]),
            "tmin": float(epochs_config["tmin"]),
            "tmax": float(epochs_config["tmax"]),
        },
        "label_metadata": {
            "num_classes": 2,
            "class_to_command": {str(k): v for k, v in CLASS_TO_COMMAND.items()},
            "class_to_meaning": {
                "0": "left-hand motor imagery",
                "1": "right-hand motor imagery",
            },
        },
        "config_metadata": {
            "model": dict(model_config),
            "config_file": "config/eeg_config.yaml",
        },
    }


def train_model(config: dict, X_train, y_train, X_val, y_val):
    """Train EEGNet and save the best validation checkpoint."""
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    model_config = config["model"]
    training_config = config["training"]

    if model_config["architecture"] != "EEGNet":
        raise ValueError(
            f"Unsupported architecture: {model_config['architecture']!r}. "
            "The project now uses only the consolidated EEGNet."
        )
    if model_config["num_classes"] != 2:
        raise ValueError("EEGNet training requires exactly 2 classes")

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=training_config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=training_config["batch_size"],
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    print("Using consolidated EEGNet")

    model = EEGNet(
        n_channels=model_config["input_channels"],
        n_classes=2,
        n_samples=X_train.shape[2],
        dropout=model_config["dropout"],
        kernel_length=model_config["kernel_length"],
        F1=model_config.get("F1", 8),
        D=model_config.get("D", 2),
        F2=model_config.get("F2", 16),
        use_attention=model_config.get("use_attention", False),
    )
    classifier = EEGClassifier(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    patience = training_config["early_stopping_patience"]

    save_path = get_project_root() / training_config["savedir"] / training_config["savename"]
    os.makedirs(save_path.parent, exist_ok=True)

    for epoch in range(training_config["num_epochs"]):
        train_losses, train_accs = [], []
        for X_batch, y_batch in train_loader:
            loss, acc = classifier.train_step(X_batch, y_batch, optimizer, criterion)
            train_losses.append(loss)
            train_accs.append(acc)

        val_losses, val_accs = [], []
        for X_batch, y_batch in val_loader:
            loss, acc, _ = classifier.eval_step(X_batch, y_batch, criterion)
            val_losses.append(loss)
            val_accs.append(acc)

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))
        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))

        print(
            f"Epoch {epoch + 1:04d}/{training_config['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            classifier.save(str(save_path))
            checkpoint = torch.load(save_path, map_location="cpu")
            checkpoint.update(
                _build_checkpoint_metadata(
                    config, epoch + 1, val_loss, val_acc, device
                )
            )
            torch.save(checkpoint, save_path)

            print(
                f"  → Saved best checkpoint "
                f"(val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})"
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break

    if not save_path.exists():
        raise RuntimeError("Training completed without producing a checkpoint")

    print(f"\n✓ Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"✓ Checkpoint: {save_path}")
    return classifier


def main():
    print("\n🧠 EEG MOTOR IMAGERY CLASSIFIER TRAINING\n")
    config = load_config("eeg_config")
    X_train, y_train, X_val, y_val = prepare_data(config)
    train_model(config, X_train, y_train, X_val, y_val)

    save_path = get_project_root() / config["training"]["savedir"] / config["training"]["savename"]
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
