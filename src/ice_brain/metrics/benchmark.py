"""Benchmark measurements for the first ICE-brAIn pipeline."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class BenchmarkStats:
    frames_processed: int = 0
    elapsed_seconds: float = 0.0
    detections: int = 0
    tracks: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds

    def add_class_counts(self, counts: dict[str, int]) -> None:
        for name, count in counts.items():
            self.class_counts[name] = self.class_counts.get(name, 0) + count
