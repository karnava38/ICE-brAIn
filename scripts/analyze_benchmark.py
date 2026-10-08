"""Analyze a completed ICE-brAIn benchmark JSONL output."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_total_frames(summary_path: Path) -> int | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    value = summary.get("video_frames")
    return int(value) if value is not None else None


def analyze(path: str | Path, total_frames: int | None = None) -> dict:
    """Analyze tracked observations and distinguish missing vs duplicate detections."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if total_frames is None:
        total_frames = _load_total_frames(path.parent / "summary.json")

    frame_counts_by_class: dict[str, Counter[int]] = defaultdict(Counter)
    track_frames: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    confidence_by_class: dict[str, list[float]] = defaultdict(list)
    total_observations = 0

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            class_name = str(row["class_name"])
            frame_index = int(row["frame_index"])
            track_id = int(row["track_id"])
            confidence = float(row["confidence"])

            total_observations += 1
            frame_counts_by_class[class_name][frame_index] += 1
            track_frames[class_name][track_id].append(frame_index)
            confidence_by_class[class_name].append(confidence)

    result: dict = {
        "input": str(path),
        "total_observations": total_observations,
        "total_video_frames": total_frames,
        "classes": {},
    }

    for class_name in sorted(frame_counts_by_class):
        counts = frame_counts_by_class[class_name]
        frames_with_observations = len(counts)
        unique_tracks = len(track_frames[class_name])
        observations = sum(counts.values())
        distribution = Counter(counts.values())

        tracks = track_frames[class_name]
        lifetimes = [len(frames_seen) for frames_seen in tracks.values()]

        confidences = confidence_by_class[class_name]
        missing_frames = (
            max((total_frames or 0) - frames_with_observations, 0)
            if total_frames is not None
            else None
        )
        multi_frames = sum(v for k, v in distribution.items() if k >= 2)

        result["classes"][class_name] = {
            "observations": observations,
            "frames_with_observations": frames_with_observations,
            "frames_without_observations": missing_frames,
            "unique_track_ids": unique_tracks,
            "mean_observations_per_observed_frame": round(
                observations / frames_with_observations, 4
            ) if frames_with_observations else 0.0,
            "mean_observations_per_video_frame": round(
                observations / total_frames, 4
            ) if total_frames else None,
            "frames_with_1_observation": distribution.get(1, 0),
            "frames_with_2plus_observations": multi_frames,
            "multi_observation_rate_all_frames": round(
                multi_frames / total_frames, 4
            ) if total_frames else None,
            "mean_track_lifetime_frames": round(
                sum(lifetimes) / len(lifetimes), 2
            ) if lifetimes else 0.0,
            "max_track_lifetime_frames": max(lifetimes, default=0),
            "confidence_mean": round(
                sum(confidences) / len(confidences), 4
            ) if confidences else 0.0,
            "confidence_min": round(min(confidences), 4) if confidences else 0.0,
            "confidence_max": round(max(confidences), 4) if confidences else 0.0,
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/benchmark/tracks.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="data/processed/benchmark/analysis.json",
        type=Path,
    )
    parser.add_argument(
        "--total-frames",
        default=None,
        type=int,
        help="Override total source frames when summary.json is unavailable.",
    )
    args = parser.parse_args()

    report = analyze(args.input, total_frames=args.total_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
