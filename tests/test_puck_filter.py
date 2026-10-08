from ice_brain.puck.filter import PuckCandidate, SinglePuckFilter


def candidate(frame, track, confidence, x, y):
    return PuckCandidate(
        frame_index=frame,
        track_id=track,
        confidence=confidence,
        x=x,
        y=y,
        bbox=(x - 2, y - 2, x + 2, y + 2),
    )


def test_selects_one_candidate_per_frame():
    selector = SinglePuckFilter(1000, 500)

    first = selector.update([
        candidate(0, 1, 0.8, 100, 100),
        candidate(0, 2, 0.9, 900, 400),
    ])
    second = selector.update([
        candidate(1, 1, 0.7, 120, 105),
        candidate(1, 3, 0.95, 880, 390),
    ])

    assert first.track_id == 2
    assert second.track_id == 3


def test_continuity_can_beat_a_small_confidence_difference():
    selector = SinglePuckFilter(1000, 500)

    selector.update([candidate(0, 1, 0.90, 100, 100)])
    selected = selector.update([
        candidate(1, 1, 0.75, 110, 105),
        candidate(1, 2, 0.78, 800, 400),
    ])

    assert selected.track_id == 1
