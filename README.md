# ICE-brAIn

Hockey video analytics platform.

## Goal

Build a modular computer-vision pipeline that converts hockey game video into structured tracking data and, later, events and analytics.

## Current MVP

~~~text
Video
  -> HockeyAI object detection
  -> ByteTrack
  -> Structured tracking output
  -> Benchmark
~~~

We intentionally start with a small, measurable pipeline before implementing higher-level hockey intelligence such as possession, passes, shots, zone entries, and tactical events.

## Current status

The repository now contains:

- modular package structure for detection, tracking, calibration, identity, puck, possession, events, and metrics;
- HockeyAI detector/tracker adapters;
- a headless benchmark runner that writes tracks.jsonl and summary.json;
- unit tests and GitHub Actions CI.

The HockeyAI weights are not stored in Git. They are downloaded from the project's Hugging Face repository.

## Quick start

Install the project in a Python 3.11+ environment:

~~~bash
pip install -e ".[dev]"
~~~

Download the HockeyAI weights:

~~~bash
python scripts/download_hockeyai.py
~~~

Run a benchmark on a real hockey video:

~~~bash
python scripts/benchmark_video.py --video path/to/game_clip.mp4
~~~

For a first smoke test on a weak CPU machine, use a short clip and a larger frame stride:

~~~bash
python scripts/benchmark_video.py \
  --video path/to/game_clip.mp4 \
  --stride 4 \
  --device cpu
~~~

The benchmark produces:

~~~text
data/processed/benchmark/
├── tracks.jsonl
└── summary.json
~~~

summary.json currently records source FPS, frame count, resolution, processing FPS, number of track observations, and per-class observation counts.

## Design principles

- Keep detection, tracking, calibration, identity, puck, possession, events, and metrics as separate modules.
- Treat HockeyAI as a detector/tracker dependency rather than copying its application architecture.
- Benchmark on real hockey footage before training new models.
- Record model and pipeline versions with outputs so results remain reproducible.
- Make CPU execution possible for development; use a GPU when processing realistic match volumes.

## Repository layout

~~~text
src/
  ice_brain/
    detection/
    tracking/
    calibration/
    identity/
    puck/
    possession/
    events/
    metrics/
    pipeline/
models/
configs/
data/
  raw/
  benchmark/
  processed/
tests/
scripts/
.github/workflows/
~~~

## Next milestone

Build the benchmark report around a fixed 30–60 second hockey clip and measure:

1. player detection quality;
2. puck recall and false positives;
3. ID switches / tracking quality;
4. processing FPS;
5. stability across different camera views.

Only after this baseline will we decide which components need custom training.
