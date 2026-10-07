# Benchmark footage

The first supplied benchmark source is a 77-second screen recording of live hockey footage.

For automated CV benchmarking, use the prepared 55-second segment from approximately 00:15 to 01:10:

- landscape display: 2778x1284
- source rate: approximately 33.82 FPS
- source video codec: HEVC
- content: live game, broadcast-style elevated camera
- visible scoreboard and rink-side graphics remain in the footage

The first ~15 seconds are phone/system UI rather than hockey footage, and the final seconds contain video-player controls, so they are excluded from the initial benchmark.

The video itself is not stored in Git. Keep it in the local benchmark dataset and pass its path to the benchmark script.
