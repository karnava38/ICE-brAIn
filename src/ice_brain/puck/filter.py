"""Temporal single-puck candidate filter."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot


@dataclass(frozen=True, slots=True)
class PuckCandidate:
    frame_index: int
    track_id: int
    confidence: float
    x: float
    y: float
    bbox: tuple[float, float, float, float]


class SinglePuckFilter:
    """Select at most one puck candidate per frame using temporal continuity.

    This is a baseline filter, not a trained puck model. It deliberately avoids
    aggressive physics assumptions until rink calibration and puck ground truth
    are available.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        confidence_weight: float = 0.70,
        continuity_weight: float = 0.30,
        continuity_sigma_ratio: float = 0.12,
        same_track_bonus: float = 0.10,
    ) -> None:
        self.diagonal = hypot(frame_width, frame_height)
        self.confidence_weight = confidence_weight
        self.continuity_weight = continuity_weight
        self.continuity_sigma = max(self.diagonal * continuity_sigma_ratio, 1.0)
        self.same_track_bonus = same_track_bonus
        self.previous: PuckCandidate | None = None

    def update(self, candidates: list[PuckCandidate]) -> PuckCandidate | None:
        """Select one candidate for the current frame, or None when absent."""
        if not candidates:
            return None

        if self.previous is None:
            selected = max(candidates, key=lambda c: c.confidence)
            self.previous = selected
            return selected

        def score(candidate: PuckCandidate) -> float:
            distance = hypot(
                candidate.x - self.previous.x,
                candidate.y - self.previous.y,
            )
            continuity = exp(-distance / self.continuity_sigma)
            same_track = (
                self.same_track_bonus if candidate.track_id == self.previous.track_id else 0.0
            )
            return (
                self.confidence_weight * candidate.confidence
                + self.continuity_weight * continuity
                + same_track
            )

        selected = max(candidates, key=score)
        self.previous = selected
        return selected
