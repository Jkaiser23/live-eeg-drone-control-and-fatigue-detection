#!/usr/bin/env python3
"""Live EEG motor-imagery control demo using the drone simulator."""

from __future__ import annotations

import time

import pygame
import torch

from neuralflight.controllers.drone_controller import DroneController
from neuralflight.eeg.live_worker import LiveEEGConfig, LiveEEGWorker
from neuralflight.eeg.prediction import EEGPredictor
from neuralflight.simulator.drone_sim import DroneSimulator
from neuralflight.utils.config_loader import get_project_root, load_config


def main() -> None:
    eeg_config = load_config("eeg_config")
    drone_config = load_config("drone_config")
    model_path = get_project_root() / eeg_config["training"]["savedir"] / eeg_config["training"]["savename"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Run neuralflight-train first.")

    predictor = EEGPredictor(str(model_path), device="cuda" if torch.cuda.is_available() else "cpu")
    preprocessing = eeg_config["preprocessing"]
    board_config = eeg_config.get("live_board")
    if not board_config:
        raise ValueError("eeg_config.yaml must define live_board for the live EEG demo")
    live_config = LiveEEGConfig(
        board_id=int(board_config["board_id"]),
        serial_port=str(board_config["serial_port"]),
        eeg_channels=tuple(int(i) for i in board_config["eeg_channels"]),
        sampling_rate=int(preprocessing["sampling_rate"]),
        window_samples=predictor.config.n_samples,
        update_interval_s=float(board_config.get("update_interval_s", 0.25)),
    )
    worker = LiveEEGWorker(predictor, live_config)
    simulator = DroneSimulator(drone_config)
    controller = DroneController(simulator)
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    live_control = eeg_config.get("live_control", {})
    confidence_threshold = float(live_control.get("confidence_threshold", 0.65))
    command_interval_s = float(live_control.get("command_interval_s", 1.0))
    last_command_time = 0.0
    latest = None
    running = True
    worker.start()
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if controller.is_flying:
                            controller.land()
                        else:
                            controller.takeoff()

            prediction = worker.get_latest()
            if prediction is not None:
                latest = prediction
                class_id, confidence, command = prediction
                now = time.monotonic()
                if controller.is_flying and confidence >= confidence_threshold and now - last_command_time >= command_interval_s:
                    controller.move(command, intensity=0.6)
                    last_command_time = now
            if controller.is_flying and (latest is None or latest[1] < confidence_threshold):
                controller.hover()

            running = simulator.update()
            if latest is not None:
                class_id, confidence, command = latest
                class_name = "Left Hand" if class_id == 0 else "Right Hand"
                status = f"Command: {command}" if confidence >= confidence_threshold else "Command: HOVER (low confidence)"
                panel = pygame.Surface((430, 105))
                panel.set_alpha(200)
                panel.fill((20, 20, 40))
                simulator.screen.blit(panel, (5, 115))
                for i, line in enumerate((f"Prediction: {class_name}", f"Confidence: {confidence:.1%}", status)):
                    simulator.screen.blit(font.render(line, True, (230, 230, 230)), (15, 125 + i * 28))
            error = worker.get_last_error()
            if error:
                simulator.screen.blit(small_font.render(f"EEG error: {error[:70]}", True, (255, 180, 120)), (10, simulator.height - 55))
            simulator.screen.blit(small_font.render("SPACE takeoff/land | ESC exit | low confidence -> hover", True, (200, 200, 200)), (10, simulator.height - 30))
            pygame.display.flip()
            clock.tick(60)
    finally:
        worker.stop()
        worker.join(timeout=3.0)
        if controller.is_flying:
            controller.land()
        simulator.close()


if __name__ == "__main__":
    main()
