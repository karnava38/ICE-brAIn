from ice_brain.metrics.benchmark import BenchmarkStats


def test_benchmark_fps():
    stats = BenchmarkStats(frames_processed=120, elapsed_seconds=10)
    assert stats.fps == 12.0


def test_class_counts_accumulate():
    stats = BenchmarkStats()
    stats.add_class_counts({"player": 3, "puck": 1})
    stats.add_class_counts({"player": 2, "goalie": 1})

    assert stats.class_counts == {"player": 5, "puck": 1, "goalie": 1}
