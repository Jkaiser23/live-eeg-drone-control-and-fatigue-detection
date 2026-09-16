"""Priority-ordered safety decision chain for the flight control loop.

Evaluated once per control-loop tick, synchronously, from a FatigueResult
(Phase 2) plus the raw FatigueStateSnapshot (Phase 1) and a drone-link health
flag. Checks run in STRICT priority order and the first one that fires wins
-- results are never blended. This mirrors the "Detected State -> System
Response" table from the project brief:

  1. drone link lost         -> EMERGENCY_STOP immediately (near-instant, no grace)
  2. both modalities stale   -> LAND (after brief grace) -- total data loss
  3. face lost (vision only) -> LAND (after grace) -- vision specifically gone
  4. sustained high fatigue  -> LAND (after grace) -- fatigue_index >= hard limit
  5. modality disagreement   -> HOVER -- conflicting signals, don't average away
  6. mild fatigue            -> REDUCE_SPEED -- degrade gracefully, keep flying
  7. otherwise               -> NORMAL

`SafetyMonitor` is intentionally the ONLY stateful piece in the fatigue
pipeline (Phase 1's SharedFatigueState is state too, but it's a passive
container; SafetyMonitor actively tracks consecutive-violation streaks so a
single bad tick doesn't trigger a landing). One SafetyMonitor instance per
flight session -- it is not safe to share across concurrent flights.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from neuralflight.fatigue.fusion import FatigueResult
from neuralflight.fatigue.shared_state import FatigueStateSnapshot


class SafetyAction(Enum):
    NORMAL = "normal"  # execute the requested command as-is
    REDUCE_SPEED = "reduce_speed"  # execute, but the caller should throttle intensity
    HOVER = "hover"  # suppress new commands, hold position
    LAND = "land"  # force immediate landing
    EMERGENCY_STOP = "emergency_stop"  # bypass controller state and hard-stop backend


@dataclass
class SafetyThresholds:
    """All tunable knobs for the safety chain, in one place.

    Grace periods are expressed in TICKS, not seconds, because the safety
    chain is only ever evaluated from a fixed-rate control loop -- ticks are
    the natural unit for debounce logic and avoid needing a control-loop
    frequency parameter threaded through this module.
    """

    max_staleness_s: float = 1.5
    quality_floor: float = 0.3

    fatigue_soft_limit: float = 0.5  # REDUCE_SPEED at/above this
    fatigue_hard_limit: float = 0.8  # eligible for LAND (after grace) at/above this
    disagreement_limit: float = 0.4  # HOVER at/above this (eeg vs vision split)

    hover_after_ticks: int = 3  # ~0.15s at 20 Hz -- short, absorbs one blink/flicker
    land_after_ticks: int = 15  # ~0.75s at 20 Hz -- sustained, not a single bad tick
    link_lost_land_ticks: int = 1  # near-instant -- no grace for a dead drone link


@dataclass
class _DebounceCounter:
    """Tracks a consecutive-violation streak. Resets to 0 on any good tick."""

    streak: int = 0

    def update(self, is_bad_tick: bool) -> int:
        self.streak = self.streak + 1 if is_bad_tick else 0
        return self.streak


@dataclass(frozen=True)
class SafetyDecision:
    """Output of one safety evaluation. Immutable -- produced fresh every tick."""

    action: SafetyAction
    reason: str  # short machine-readable tag, e.g. "sustained_high_fatigue"
    fatigue_result: FatigueResult  # carried through for logging/telemetry


class SafetyMonitor:
    """Stateful (per-tick debounce counters) safety policy evaluator.

    Call `evaluate()` once per control-loop tick, in order, with the latest
    snapshot/fatigue result/link status. Do not call out of tick order and
    do not skip ticks -- the debounce counters assume one call == one tick.
    """

    def __init__(self, thresholds: SafetyThresholds | None = None) -> None:
        self.thresholds = thresholds or SafetyThresholds()
        self._link_lost_counter = _DebounceCounter()
        self._data_loss_counter = _DebounceCounter()
        self._face_lost_counter = _DebounceCounter()
        self._fatigue_counter = _DebounceCounter()

    def evaluate(
        self,
        snapshot: FatigueStateSnapshot,
        fatigue_result: FatigueResult,
        now: float,
        drone_link_ok: bool,
    ) -> SafetyDecision:
        t = self.thresholds

        # 1. Hardware link check -- highest priority, effectively no grace.
        # A dead link invalidates controller flight-state telemetry, so this
        # deliberately uses the backend's unconditional hard-stop path.
        link_streak = self._link_lost_counter.update(not drone_link_ok)
        if link_streak >= t.link_lost_land_ticks:
            return SafetyDecision(SafetyAction.EMERGENCY_STOP, "drone_link_lost", fatigue_result)

        # 2. Total data staleness -- both modalities gone (fusion already says so).
        both_stale = fatigue_result.mode == "no_data"
        data_streak = self._data_loss_counter.update(both_stale)
        if data_streak >= t.land_after_ticks:
            return SafetyDecision(SafetyAction.LAND, "both_modalities_stale", fatigue_result)
        if data_streak >= t.hover_after_ticks:
            return SafetyDecision(SafetyAction.HOVER, "both_modalities_stale_grace", fatigue_result)

        # 3. Face-lost check -- vision specifically unusable, even if EEG alone is fine.
        # Checked against the raw snapshot (not fatigue_result.mode) so this fires
        # even while fusion is happily reporting "eeg_only" as a valid mode.
        vision_lost = not snapshot.vision.is_usable(now, t.max_staleness_s, t.quality_floor)
        face_streak = self._face_lost_counter.update(vision_lost)
        if face_streak >= t.land_after_ticks:
            return SafetyDecision(SafetyAction.LAND, "face_lost_sustained", fatigue_result)
        if face_streak >= t.hover_after_ticks:
            return SafetyDecision(SafetyAction.HOVER, "face_lost_grace", fatigue_result)

        # 4. High fatigue -- hard limit, sustained.
        fatigue_streak = self._fatigue_counter.update(fatigue_result.fatigue_index >= t.fatigue_hard_limit)
        if fatigue_streak >= t.land_after_ticks:
            return SafetyDecision(SafetyAction.LAND, "sustained_high_fatigue", fatigue_result)
        if fatigue_streak >= t.hover_after_ticks:
            return SafetyDecision(SafetyAction.HOVER, "high_fatigue_grace", fatigue_result)

        # 5. Modality disagreement -- suppress new commands rather than average it away.
        # No debounce here: a single wildly-disagreeing tick is reason enough to hover,
        # since (unlike staleness/fatigue) this is about not trusting THIS tick's blend.
        if fatigue_result.disagreement is not None and fatigue_result.disagreement >= t.disagreement_limit:
            return SafetyDecision(SafetyAction.HOVER, "modality_disagreement", fatigue_result)

        # 6. Mild fatigue -- soft limit, degrade gracefully rather than stop flying.
        if fatigue_result.fatigue_index >= t.fatigue_soft_limit:
            return SafetyDecision(SafetyAction.REDUCE_SPEED, "mild_fatigue", fatigue_result)

        # 7. Normal.
        return SafetyDecision(SafetyAction.NORMAL, "ok", fatigue_result)
