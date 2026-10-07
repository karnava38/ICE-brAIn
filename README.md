# ICE-brAIn

Hockey video analytics platform.

## Goal

Build a modular computer-vision pipeline that converts hockey game video into structured tracking data and, later, events and analytics.

## Current MVP

```
Video
  -> Object detection (HockeyAI)
  -> Multi-object tracking (ByteTrack)
  -> Structured tracking output
  -> Benchmark
```

We intentionally start with a small, measurable pipeline before implementing higher-level hockey intelligence such as possession, passes, shots, zone entries, and tactical events.

## Design principles

- Keep detection, tracking, calibration, identity, puck, possession, events, and metrics as separate modules.
- Treat HockeyAI as a detector dependency rather than copying its application architecture.
- Benchmark on real hockey footage before training new models.
- Record model and pipeline versions with outputs so results remain reproducible.

## Repository layout

```
src/
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
```
