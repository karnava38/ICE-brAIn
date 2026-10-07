from ice_brain.detection.base import Detection


def test_detection_geometry():
    detection = Detection(
        frame_index=10,
        class_id=4,
        confidence=0.9,
        x1=10,
        y1=20,
        x2=30,
        y2=60,
    )

    assert detection.center == (20.0, 40.0)
    assert detection.width == 20
    assert detection.height == 40
