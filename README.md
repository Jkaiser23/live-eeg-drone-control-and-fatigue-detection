# NeuralFlight

NeuralFlight is a Python framework for controlling a simulated or physical
drone with hand gestures, head pose, and EEG motor-imagery signals.

## Installation

Python 3.10 or newer is required. Create and activate a virtual environment,
then install the application and development tools:

```bash
python -m pip install -e ".[dev]"
```

Optional hardware integrations are installed only when needed:

```bash
# BrainFlow EEG worker and synthetic-board harness
python -m pip install -e ".[eeg]"

# CoDrone EDU hardware adapter
python -m pip install -e ".[codrone]"
```

To install every optional dependency for local development:

```bash
python -m pip install -e ".[dev,eeg,codrone]"
```

## Commands

After installation, launch the demos with:

```bash
neuralflight-hand
neuralflight-head
neuralflight-eeg
neuralflight-train
```

Run the test suite with:

```bash
python -m pytest -q
```

The detailed project documentation is available in [docs/](docs/), and the
full project overview is maintained in [.github/README.md](.github/README.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
