"""Analyze a completed ICE-brAIn benchmark JSONL output."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

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
        "classes": {},
    }

    for class_name in sorted(frame_counts_by_class):
        counts = frame_counts_by_class[class_name]
        frames = len(counts)
        unique_tracks = len(track_frames[class_name])
        observations = sum(counts.values())
        distribution = Counter(counts.values())

        tracks = track_frames[class_name]
        lifetimes = [len(frames_seen) for frames_seen in tracks.values()]

        confidences = confidence_by_class[class_name]
        result["classes"][class_name] = {
            "observations": observations,
            "frames_with_observations": frames,
            "unique_track_ids": unique_tracks,
            "mean_observations_per_frame": round(observations / frames, 4) if frames else 0.0,
            "frames_with_0_observations": 0,
            "frames_with_1_observation": distribution.get(1, 0),
            "frames_with_2plus_observations": sum(v for k, v in distribution.items() if k >= 2),
            "mean_track_lifetime_frames": round(sum(lifetimes) / len(lifetimes), 2) if lifetimes else 0.0,
            "max_track_lifetime_frames": max(lifetimes, default=0),
            "confidence_mean": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
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
    args = parser.parse_args()

    report = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
