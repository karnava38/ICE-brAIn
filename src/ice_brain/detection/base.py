"""Common detection data structures and interfaces."""

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class Detection:
    """One object detection in image coordinates."""

    frame_index: int
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class Detector(Protocol):
    """Interface implemented by detector backends such as HockeyAI."""

    def detect(self, frame, frame_index: int) -> Sequence[Detection]:
        """Return detections for one video frame."""
        ...
