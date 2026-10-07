"""Run the first ICE-brAIn benchmark on a hockey video."""

from __future__ import annotations

import argparse
from pathlib import Path

from ice_brain.detection.hockeyai import HockeyAIDetector
from ice_brain.pipeline.benchmark_runner import run_benchmark


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
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    detector = HockeyAIDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    summary = run_benchmark(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output,
        confidence_threshold=args.conf,
        sample_stride=args.stride,
        tracker_config=args.tracker,
        device=args.device,
        class_name=detector.class_name,
    )

    print("\nICE-brAIn benchmark")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
