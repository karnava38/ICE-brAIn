"""Run the first ICE-brAIn benchmark on a hockey video."""

from __future__ import annotations

import argparse
from pathlib import Path

from ice_brain.pipeline.benchmark_runner import run_benchmark
from ice_brain.tracking.hockeyai_tracker import HockeyAIByteTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", default=Path("models/HockeyAI_model_weight.pt"), type=Path)
    parser.add_argument("--output", default=Path("data/processed/benchmark"), type=Path)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.conf <= 1.0:
        raise SystemExit("--conf must be > 0 and <= 1")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    tracker = HockeyAIByteTracker(
        model_path=args.model,
        confidence_threshold=args.conf,
        tracker_config=args.tracker,
        device=args.device,
    )

    summary = run_benchmark(
        video_path=args.video,
        tracker=tracker,
        output_dir=args.output,
        sample_stride=args.stride,
        class_name=tracker.class_name,
    )

    print("\nICE-brAIn benchmark")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
