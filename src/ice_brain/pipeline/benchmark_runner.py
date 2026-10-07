"""Headless benchmark runner for real hockey footage."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import cv2

from ice_brain.metrics.benchmark import BenchmarkStats
from ice_brain.tracking.hockeyai_tracker import HockeyAIByteTracker


def run_benchmark(
    video_path: str | Path,
    tracker: HockeyAIByteTracker,
    output_dir: str | Path,
    sample_stride: int = 1,
    class_name: Callable[[int], str] | None = None,
) -> dict:
    """Run tracking and save one JSONL record per sampled frame."""
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_source = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    records_path = output_dir / "tracks.jsonl"
    stats = BenchmarkStats()
    started = time.perf_counter()

    try:
        with records_path.open("w", encoding="utf-8") as records:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % sample_stride == 0:
                    tracks = tracker.track_frame(frame, frame_index)
                    stats.frames_processed += 1
                    stats.tracks += len(tracks)

                    counts: dict[str, int] = {}
                    for track in tracks:
                        name = class_name(track.class_id) if class_name else tracker.class_name(track.class_id)
                        counts[name] = counts.get(name, 0) + 1
                        records.write(
                            json.dumps(
                                {
                                    "frame_index": track.frame_index,
                                    "track_id": track.track_id,
                                    "class_id": track.class_id,
                                    "class_name": name,
                                    "confidence": round(track.confidence, 6),
                                    "bbox": [
                                        round(track.x1, 3),
                                        round(track.y1, 3),
                                        round(track.x2, 3),
                                        round(track.y2, 3),
                                    ],
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    stats.add_class_counts(counts)

                frame_index += 1
    finally:
        capture.release()

    stats.elapsed_seconds = time.perf_counter() - started

    summary = {
        "video": str(video_path),
        "video_fps": fps_source,
        "video_frames": total_frames,
        "resolution": {"width": width, "height": height},
        "sample_stride": sample_stride,
        "frames_processed": stats.frames_processed,
        "elapsed_seconds": round(stats.elapsed_seconds, 3),
        "processing_fps": round(stats.fps, 3),
        "track_observations": stats.tracks,
        "class_observations": stats.class_counts,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
