#!/usr/bin/env python3
"""Train the canonical two-class EEGNet motor-imagery classifier."""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neuralflight.eeg.dataset import PhysioNetDataset
from neuralflight.models.eegnet import EEGClassifier, EEGNet
from neuralflight.utils.config_loader import get_project_root, load_config

CLASS_TO_COMMAND = {0: "strafe_left", 1: "strafe_right"}


def _validate(X, y, name, channels, samples):
    if X is None or y is None or len(X) == 0:
        raise ValueError(f"{name}: no EEG data was loaded")
    if X.ndim != 3 or X.shape[1:] != (channels, samples):
        raise ValueError(f"{name}: expected (epochs, {channels}, {samples}), received {X.shape}")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{name}: invalid X/y shapes: X={X.shape}, y={y.shape}")
    labels = set(np.unique(y).tolist())
    if labels != {0, 1}:
        raise ValueError(f"{name}: expected both classes 0 and 1; received {labels}")


def _load_split(dataset, subjects, runs, channels, name):
    X_parts, y_parts = [], []
    for subject_id in tqdm(subjects, desc=f"{name} subjects"):
        try:
            dataset.download_subject(subject_id, runs)
            X, y = dataset.load_subject(subject_id, runs, channels)
            if len(X) == 0:
                continue
            if set(np.unique(y).tolist()) != {0, 1}:
                print(f"Subject {subject_id}: missing one class; skipping")
                continue
            X_parts.append(X)
            y_parts.append(np.asarray(y, dtype=np.int64))
        except Exception as exc:
            print(f"Subject {subject_id}: {exc}")
    if not X_parts:
        raise ValueError(f"No usable {name.lower()} subjects were loaded")
    return np.concatenate(X_parts), np.concatenate(y_parts), len(X_parts)


def prepare_data(config):
    dataset_cfg = config["dataset"]
    prep_cfg = config["preprocessing"]
    model_cfg = config["model"]
    epoch_cfg = config["epochs"]
    if model_cfg["num_classes"] != 2:
        raise ValueError("The model must have exactly 2 classes")
    if model_cfg["input_channels"] != len(prep_cfg["channels"]):
        raise ValueError("Model input_channels must match preprocessing channels")

    fs = int(prep_cfg["sampling_rate"])
    samples = int(round((float(epoch_cfg["tmax"]) - float(epoch_cfg["tmin"])) * fs))
    channels = prep_cfg["channels"]
    dataset = PhysioNetDataset(
        str(get_project_root() / "data" / "raw" / "physionet"),
        sampling_rate=fs,
        channels=channels,
        tmin=float(epoch_cfg["tmin"]),
        window_seconds=samples / fs,
        lowcut=float(prep_cfg["lowcut"]),
        highcut=float(prep_cfg["highcut"]),
        filter_order=int(prep_cfg["filter_order"]),
    )
    X_train, y_train, train_count = _load_split(dataset, dataset_cfg["train_subjects"], dataset_cfg["runs"], channels, "Training")
    X_val, y_val, val_count = _load_split(dataset, dataset_cfg["val_subjects"], dataset_cfg["runs"], channels, "Validation")
    _validate(X_train, y_train, "Training", int(model_cfg["input_channels"]), samples)
    _validate(X_val, y_val, "Validation", int(model_cfg["input_channels"]), samples)
    print(f"Training subjects={train_count}, shape={X_train.shape}")
    print(f"Validation subjects={val_count}, shape={X_val.shape}")
    return X_train, y_train, X_val, y_val


def _checkpoint_metadata(config, epoch, val_loss, val_acc, device):
    d, p, m, t, e = config["dataset"], config["preprocessing"], config["model"], config["training"], config["epochs"]
    return {
        "training_metadata": {"epoch": epoch, "validation_loss": float(val_loss), "validation_accuracy": float(val_acc), "device": str(device), "batch_size": int(t["batch_size"]), "learning_rate": float(t["learning_rate"]), "num_epochs": int(t["num_epochs"]), "early_stopping_patience": int(t["early_stopping_patience"])},
        "data_metadata": {"dataset": d["name"], "runs": list(d["runs"]), "train_subjects": list(d["train_subjects"]), "val_subjects": list(d["val_subjects"]), "channels": list(p["channels"]), "sampling_rate": int(p["sampling_rate"]), "lowcut": float(p["lowcut"]), "highcut": float(p["highcut"]), "filter_order": int(p["filter_order"]), "notch_freq": float(p["notch_freq"]), "tmin": float(e["tmin"]), "tmax": float(e["tmax"])},
        "label_metadata": {"num_classes": 2, "class_to_command": {str(k): v for k, v in CLASS_TO_COMMAND.items()}, "class_to_meaning": {"0": "left-hand motor imagery", "1": "right-hand motor imagery"}},
        "config_metadata": {"model": dict(m), "config_file": "config/eeg_config.yaml"},
    }


def train_model(config, X_train, y_train, X_val, y_val):
    model_cfg, train_cfg = config["model"], config["training"]
    if model_cfg["architecture"] != "EEGNet" or model_cfg["num_classes"] != 2:
        raise ValueError("Training requires the canonical two-class EEGNet")
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.LongTensor(y_val)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=train_cfg["batch_size"], shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EEGNet(n_channels=model_cfg["input_channels"], n_classes=2, n_samples=X_train.shape[2], dropout=model_cfg["dropout"], kernel_length=model_cfg["kernel_length"], F1=model_cfg.get("F1", 8), D=model_cfg.get("D", 2), F2=model_cfg.get("F2", 16), use_attention=model_cfg.get("use_attention", False))
    classifier = EEGClassifier(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    best_loss, best_acc, patience = float("inf"), 0.0, 0
    save_path = get_project_root() / train_cfg["savedir"] / train_cfg["savename"]
    os.makedirs(save_path.parent, exist_ok=True)
    for epoch in range(train_cfg["num_epochs"]):
        for X_batch, y_batch in train_loader:
            classifier.train_step(X_batch, y_batch, optimizer, criterion)
        val_losses, val_accs = [], []
        for X_batch, y_batch in val_loader:
            loss, acc, _ = classifier.eval_step(X_batch, y_batch, criterion)
            val_losses.append(loss); val_accs.append(acc)
        val_loss, val_acc = float(np.mean(val_losses)), float(np.mean(val_accs))
        if val_loss < best_loss:
            best_loss, best_acc, patience = val_loss, val_acc, 0
            classifier.save(str(save_path))
            checkpoint = torch.load(save_path, map_location="cpu")
            checkpoint.update(_checkpoint_metadata(config, epoch + 1, val_loss, val_acc, device))
            torch.save(checkpoint, save_path)
        else:
            patience += 1
        if patience >= train_cfg["early_stopping_patience"]:
            break
    if not save_path.exists():
        raise RuntimeError("Training completed without producing a checkpoint")
    return classifier


def main():
    config = load_config("eeg_config")
    train_model(config, *prepare_data(config))


if __name__ == "__main__":
    main()
