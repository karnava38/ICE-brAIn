"""Common tracking data structures."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Track:
    """A tracked object for one frame."""

    frame_index: int
    track_id: int
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
