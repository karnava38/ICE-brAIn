"""Apply the baseline single-puck temporal filter to benchmark tracks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ice_brain.puck.filter import PuckCandidate, SinglePuckFilter


def filter_puck(
    input_path: str | Path,
    output_path: str | Path,
    frame_width: int,
    frame_height: int,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)

    by_frame: dict[int, list[PuckCandidate]] = {}
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("class_name") != "puck":
                continue
            x1, y1, x2, y2 = row["bbox"]
            by_frame.setdefault(int(row["frame_index"]), []).append(
                PuckCandidate(
                    frame_index=int(row["frame_index"]),
                    track_id=int(row["track_id"]),
                    confidence=float(row["confidence"]),
                    x=(float(x1) + float(x2)) / 2.0,
                    y=(float(y1) + float(y2)) / 2.0,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

    selector = SinglePuckFilter(frame_width, frame_height)
    selected = 0
    candidate_frames = len(by_frame)
    raw_multi_frames = sum(1 for values in by_frame.values() if len(values) >= 2)
    selected_frames: list[int] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for frame_index in sorted(by_frame):
            puck = selector.update(by_frame[frame_index])
            if puck is None:
                continue
            selected += 1
            selected_frames.append(frame_index)
            out.write(
                json.dumps(
                    {
                        "frame_index": puck.frame_index,
                        "track_id": puck.track_id,
                        "confidence": round(puck.confidence, 6),
                        "bbox": [round(v, 3) for v in puck.bbox],
                        "center": [round(puck.x, 3), round(puck.y, 3)],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    gaps = [
        b - a - 1
        for a, b in zip(selected_frames, selected_frames[1:])
        if b > a + 1
    ]
    return {
        "input": str(input_path),
        "output": str(output_path),
        "candidate_frames": candidate_frames,
        "raw_multi_candidate_frames": raw_multi_frames,
        "selected_frames": selected,
        "selected_coverage": round(selected / candidate_frames, 4) if candidate_frames else 0.0,
        "longest_selected_gap_frames": max(gaps, default=0),
        "mean_selected_gap_frames": round(sum(gaps) / len(gaps), 2) if gaps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/benchmark/tracks.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="data/processed/benchmark/puck_filtered.jsonl",
        type=Path,
    )
    parser.add_argument("--width", default=2778, type=int)
    parser.add_argument("--height", default=1284, type=int)
    args = parser.parse_args()

    report = filter_puck(args.input, args.output, args.width, args.height)
    print(json.dumps(report, indent=2))
\n\nif __name__ == "__main__":\n    main()\n