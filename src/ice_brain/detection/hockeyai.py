"""HockeyAI YOLO detector backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Detection


DEFAULT_CLASS_NAMES = {
    0: "center_ice",
    1: "faceoff",
    2: "goal",
    3: "goalie",
    4: "player",
    5: "puck",
    6: "referee",
}


def _normalize_names(names: Any) -> dict[int, str]:
    """Normalize Ultralytics class names to {class_id: name}."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


class HockeyAIDetector:
    """Thin adapter around the published HockeyAI YOLO weights."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
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
        self.device = device
        self.class_names = _normalize_names(self.model.names)

    def detect(self, frame: Any, frame_index: int) -> list[Detection]:
        """Run one-frame detection and return normalized detections."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)

        return [
            Detection(
                frame_index=frame_index,
                class_id=int(class_id),
                confidence=float(score),
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
            )
            for box, score, class_id in zip(xyxy, conf, cls)
        ]

    def class_name(self, class_id: int) -> str:
        """Return a stable class name, falling back to the model's metadata."""
        return self.class_names.get(class_id, DEFAULT_CLASS_NAMES.get(class_id, str(class_id)))
