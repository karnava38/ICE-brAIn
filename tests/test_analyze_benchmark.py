import json

from ice_brain.metrics.benchmark import BenchmarkStats
from scripts.analyze_benchmark import analyze


def test_analyze_benchmark(tmp_path):
    path = tmp_path / "tracks.jsonl"
    rows = [
        {"frame_index": 0, "track_id": 1, "class_name": "puck", "confidence": 0.9},
        {"frame_index": 0, "track_id": 2, "class_name": "puck", "confidence": 0.8},
        {"frame_index": 1, "track_id": 1, "class_name": "puck", "confidence": 0.7},
        {"frame_index": 0, "track_id": 10, "class_name": "player", "confidence": 0.95},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = analyze(path)
    assert report["total_observations"] == 4
    assert report["classes"]["puck"]["unique_track_ids"] == 2
    assert report["classes"]["puck"]["frames_with_1_observation"] == 1
    assert report["classes"]["puck"]["frames_with_2plus_observations"] == 1
    assert report["classes"]["player"]["unique_track_ids"] == 1
