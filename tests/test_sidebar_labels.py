"""Readable sidebar labels for feature constants."""
from face_analyzer.ui.sidebar import feature_label


def test_feature_labels_are_readable():
    assert feature_label("FACE_LANDMARKS") == "Face landmarks"
    assert feature_label("RECONSTRUCTION_3D") == "3D reconstruction"
    assert feature_label("FACE_DETECTOR_YOLO") == "YOLO face detector"
    assert feature_label("LIVENESS") == "Liveness"
