"""Late-fusion of EEG and vision fatigue readings into a single fatigue index.

This module is a pure function of a FatigueStateSnapshot -- it holds no
state of its own and is meant to be called once per control-loop tick from
the main thread, not run on its own thread. Phase 1's SharedFatigueState
already enforces the [0.0, 1.0] contract at the write boundary, so this
module can trust that any `score`/`quality` it sees is already valid; its
job is purely the fusion policy, not re-validation.

Fusion policy (mirrors the "Detected State -> System Response" table from
the project brief):
  - both modalities usable  -> weighted blend, confidence penalized by
                                how much the two modalities disagree
  - one modality usable     -> use it alone, at a reduced confidence
                                (single-modality readings are inherently
                                less trustworthy than a cross-checked one)
  - neither usable          -> fail toward the worst case: fatigue_index
                                pinned to 1.0, confidence pinned to 0.0.
                                The 0.0 confidence is what the safety layer
                                actually keys off of -- the index value
                                itself is never meant to be "trusted" in
                                this branch, it's a conservative default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from neuralflight.fatigue.shared_state import FatigueStateSnapshot

# Default fusion parameters. These are intentionally kept as function
# defaults (not hardcoded in the body) so a config-driven loader can
# override them later (Phase 5+) without touching this module's logic.
DEFAULT_MAX_STALENESS_S = 1.5
DEFAULT_QUALITY_FLOOR = 0.3
DEFAULT_WEIGHT_EEG = 0.5
DEFAULT_WEIGHT_VISION = 0.5
SINGLE_MODALITY_CONFIDENCE_PENALTY = 0.6  # applied when only one modality is usable


@dataclass(frozen=True)
class FatigueResult:
    """Output of one fusion pass. Immutable -- produced fresh every tick."""

    fatigue_index: float  # 0.0-1.0, higher = more fatigued (same contract as raw scores)
    confidence: float  # 0.0-1.0, how much the safety layer should trust fatigue_index
    disagreement: Optional[float]  # |eeg.score - vision.score| when both used, else None
    mode: str  # "fused" | "eeg_only" | "vision_only" | "no_data"


def compute_fatigue_index(
    snapshot: FatigueStateSnapshot,
    now: float,
    max_staleness_s: float = DEFAULT_MAX_STALENESS_S,
    quality_floor: float = DEFAULT_QUALITY_FLOOR,
    weight_eeg: float = DEFAULT_WEIGHT_EEG,
    weight_vision: float = DEFAULT_WEIGHT_VISION,
) -> FatigueResult:
    """Fuse the two modality readings in `snapshot` into one FatigueResult.

    Pure function: same inputs always produce the same output, no side
    effects, no I/O. Safe to call as often as needed from the control loop.
    """
    eeg_ok = snapshot.eeg.is_usable(now, max_staleness_s, quality_floor)
    vision_ok = snapshot.vision.is_usable(now, max_staleness_s, quality_floor)

    if not eeg_ok and not vision_ok:
        return FatigueResult(fatigue_index=1.0, confidence=0.0, disagreement=None, mode="no_data")

    if eeg_ok and vision_ok:
        return _fuse_both(snapshot, weight_eeg, weight_vision)

    if eeg_ok:
        return FatigueResult(
            fatigue_index=snapshot.eeg.score,
            confidence=snapshot.eeg.quality * SINGLE_MODALITY_CONFIDENCE_PENALTY,
            disagreement=None,
            mode="eeg_only",
        )

    return FatigueResult(
        fatigue_index=snapshot.vision.score,
        confidence=snapshot.vision.quality * SINGLE_MODALITY_CONFIDENCE_PENALTY,
        disagreement=None,
        mode="vision_only",
    )


def _fuse_both(snapshot: FatigueStateSnapshot, weight_eeg: float, weight_vision: float) -> FatigueResult:
    eeg_score = snapshot.eeg.score
    vision_score = snapshot.vision.score
    disagreement = abs(eeg_score - vision_score)

    w_total = weight_eeg + weight_vision
    if w_total <= 0:
        # Degenerate config (both weights zero) -- fall back to an even split
        # rather than dividing by zero. This should never happen with the
        # module defaults; it's a guard against a bad config value.
        weight_eeg = weight_vision = 0.5
        w_total = 1.0

    fused_index = (weight_eeg * eeg_score + weight_vision * vision_score) / w_total

    # Confidence rewards agreement between modalities and penalizes disagreement.
    # A full 1.0 disagreement (e.g. eeg=0.0, vision=1.0) drives confidence to 0,
    # even if both individual qualities are high -- two confident-but-conflicting
    # sensors are less trustworthy together than either one alone, which is why
    # this can end up LOWER than either single-modality branch would produce.
    avg_quality = (snapshot.eeg.quality + snapshot.vision.quality) / 2
    confidence = max(0.0, avg_quality * (1.0 - disagreement))

    return FatigueResult(
        fatigue_index=fused_index,
        confidence=confidence,
        disagreement=disagreement,
        mode="fused",
    )
