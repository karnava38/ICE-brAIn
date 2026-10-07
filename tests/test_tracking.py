from ice_brain.tracking.base import Track


def test_track_center():
    track = Track(
        frame_index=5,
        track_id=17,
        class_id=4,
        confidence=0.8,
        x1=100,
        y1=50,
        x2=140,
        y2=90,
    )

    assert track.center == (120.0, 70.0)
