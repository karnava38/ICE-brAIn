"""HockeyAI + ByteTrack adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Track


class HockeyAIByteTracker:
    """Run HockeyAI YOLO inference with Ultralytics ByteTrack."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        tracker_config: str = "bytetrack.yaml",
        device: str | int | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"HockeyAI weights not found: {self.model_path}. "
                "Run scripts/download_hockeyai.py first."
            )

        self.model = YOLO(str(self.model_path))
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config
        self.device = device

    def track_frame(self, frame: Any, frame_index: int) -> list[Track]:
        """Track one frame while retaining ByteTrack state between calls."""
        results = self.model.track(
            source=frame,
            conf=self.confidence_threshold,
            tracker=self.tracker_config,
            persist=True,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        track_ids = boxes.id.detach().cpu().numpy().astype(int)

        return [
            Track(
                frame_index=frame_index,
                track_id=int(track_id),
                class_id=int(class_id),
                confidence=float(score),
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
            )
            for box, score, class_id, track_id in zip(xyxy, conf, cls, track_ids)
        ]
