"""Sidebar controls: model selection, gallery, identity search, and the control panel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from .. import inference
from ..demo import DEMO_FACE_DETECTOR, is_demo_mode

MODEL_DISPLAY_NAMES = {
    "caffe": "Caffe",
    "dex": "DEX",
    "mivolo": "MiVOLO",
    "fairface": "FairFace",
    "deepface": "DeepFace",
    "dan": "DAN",
    "mini_xception": "Mini Xception",
    "ferplus": "FERPlus",
    "hsemotion": "HSEmotion",
    "mobilenet": "MobileNet",
    "mobilenetv2": "MobileNetV2",
    "colorimetric": "Colorimetric",
    "mediapipe": "MediaPipe",
    "eccv16": "ECCV16",
    "vggface": "VGG-Face",
    "lbph": "LBPH",
    "yolo": "YOLO",
    "ssd": "SSD",
    "scrfd": "SCRFD",
    "retinaface": "RetinaFace",
}


def _display_model_name(model_key: str) -> str:
    """Return a readable product label while preserving the runtime model key."""
    return MODEL_DISPLAY_NAMES.get(model_key, model_key.replace("_", " ").title())


def _model_checkboxes(label: str, nets: dict, container=None, help: str | None = None) -> set:
    """Render one checkbox per loaded model for a feature.

    Returns the set of model keys selected by the user.
    """
    active = set()
    if not nets:
        return active
    container = container if container is not None else st.sidebar
    container.markdown(f"**{label.title()}**")
    if help:
        container.caption(help)
    for key in nets:
        if container.checkbox(_display_model_name(key), value=True, key=f"chk_{label}_{key}"):
            active.add(key)
    return active


def _landmark_enable_button(label: str, nets: dict, state_key: str, container=None, help: str | None = None) -> set:
    """Render a single ENABLE/DISABLE toggle button for a landmark family.

    Returns the set of loaded model keys if enabled, else empty set.
    """
    if not nets:
        return set()
    container = container if container is not None else st.sidebar
    enabled = st.session_state.setdefault(state_key, True)
    display_label = label.title()
    button_label = f"Disable {display_label}" if enabled else f"Enable {display_label}"
    container.markdown(f"**{display_label}**")
    if help:
        container.caption(help)
    if container.button(button_label, key=f"enable_{state_key}", width="stretch"):
        st.session_state[state_key] = not enabled
        st.rerun()
    container.caption("Enabled" if enabled else "Disabled")
    return set(nets) if enabled else set()


@dataclass
class SidebarState:
    """Every value the sidebar controls produce for one script run."""

    conf_threshold: float
    face_detector: str
    active_age: set[str] = field(default_factory=set)
    active_gender: set[str] = field(default_factory=set)
    active_race: set[str] = field(default_factory=set)
    active_emotion: set[str] = field(default_factory=set)
    active_glasses: set[str] = field(default_factory=set)
    active_mask: set[str] = field(default_factory=set)
    active_hair_color: set[str] = field(default_factory=set)
    active_eye_color: set[str] = field(default_factory=set)
    active_recognition: set[str] = field(default_factory=set)
    active_liveness: set[str] = field(default_factory=set)
    active_gaze: set[str] = field(default_factory=set)
    active_colorization: set[str] = field(default_factory=set)
    active_face_landmarks: set[str] = field(default_factory=set)
    active_hands: set[str] = field(default_factory=set)
    search_gallery: dict[str, Any] = field(default_factory=dict)
    enable_crowd_count: bool = False

    def to_config(self, **extra: Any) -> inference.AnalysisConfig:
        """Build the pipeline config from these selections plus per-call extras (gallery, trackers, ...)."""
        return inference.AnalysisConfig(
            conf_threshold=self.conf_threshold,
            active_age=self.active_age, active_gender=self.active_gender, active_emotion=self.active_emotion,
            active_race=self.active_race, active_recognition=self.active_recognition,
            active_glasses=self.active_glasses, active_mask=self.active_mask,
            active_hair_color=self.active_hair_color, active_eye_color=self.active_eye_color,
            active_face_landmarks=self.active_face_landmarks, active_hands=self.active_hands,
            active_gaze=self.active_gaze,
            face_detector=self.face_detector,
            active_liveness=self.active_liveness,
            **extra,
        )


def render_sidebar(models: Any) -> SidebarState:
    """Render the model-selection, gallery, identity-search and control-panel sidebar sections."""
    st.sidebar.markdown("### Model selection")
    if models.offline_features:
        st.sidebar.caption(f"Unavailable: {', '.join(models.offline_features)}. Model file or dependency missing.")

    with st.sidebar.expander("Detection", expanded=True):
        preferred = DEMO_FACE_DETECTOR if is_demo_mode() else "yolo"
        active_face_detector = inference.resolve_face_detector(models, preferred) or "ssd"
        _face_detector_options = inference.available_face_detectors(models)
        if len(_face_detector_options) > 1:
            active_face_detector = st.selectbox(
                "Face detector", _face_detector_options, index=_face_detector_options.index(active_face_detector),
                format_func=_display_model_name,
                help="One detector runs per frame. YOLO is preferred when loaded; SSD is the fallback when present.",
            )

    with st.sidebar.expander("Classification", expanded=True):
        active_age = _model_checkboxes("AGE", models.age_nets, st, help="Estimated age per detected face.")
        active_gender = _model_checkboxes("GENDER", models.gender_nets, st, help="Estimated gender label per detected face.")
        active_race = _model_checkboxes("RACE", models.race_nets, st, help="Estimated race or ethnicity label per detected face.")
        active_emotion = _model_checkboxes("EMOTION", models.emotion_nets, st, help="Estimated facial expression across seven categories.")
        active_glasses = _model_checkboxes("GLASSES", models.glasses_nets, st, help="Detects whether the face appears to wear glasses.")
        active_mask = _model_checkboxes("MASK", models.mask_nets, st, help="Detects whether the face appears to wear a mask.")
        active_hair_color = _model_checkboxes("HAIR COLOR", models.hair_color_nets, st, help="Estimates dominant hair color.")
        active_eye_color = _model_checkboxes("EYE COLOR", models.eye_color_nets, st, help="Estimates dominant eye color.")

    with st.sidebar.expander("Identity and biometrics", expanded=False):
        active_recognition = _model_checkboxes("RECOGNITION", models.recognition_nets, st, help="Matches faces against saved gallery and local reference photos.")
        active_liveness = _model_checkboxes("LIVENESS", models.liveness_nets, st, help="Uses blink history to flag possible still-photo spoofing.")
        if models.liveness_nets:
            st.caption("Liveness runs in live webcam mode. A single image has no blink history.")
        active_gaze = _model_checkboxes("GAZE", models.gaze_nets, st, help="Estimates gaze direction per detected face.")

    with st.sidebar.expander("Landmarks and experimental", expanded=False):
        active_colorization = _model_checkboxes(
            "AUTO-COLORIZE B&W", models.colorization_nets, st,
            help="Converts detected grayscale source images to color before face detection runs.",
        )
        active_face_landmarks = _landmark_enable_button(
            "Face landmarks", models.face_landmarks_nets, "face_landmarks_enabled", st,
            help="Overlays facial mesh/keypoints on each detected face.",
        )
        active_hands = _landmark_enable_button(
            "Hand landmarks", models.hand_nets, "hand_landmarks_enabled", st,
            help="Overlays hand keypoints on the whole frame, independent of face detection.",
        )

    st.session_state.setdefault("gallery", inference.load_gallery())

    search_gallery = {}
    if models.recognition_nets:
        st.sidebar.markdown("### Gallery")
        gallery = st.session_state["gallery"]
        if not gallery:
            st.sidebar.caption("No enrolled identities yet.")
        for name in list(gallery):
            col_name, col_del = st.sidebar.columns([3, 1])
            col_name.text(name)
            if col_del.button("X", key=f"del_gallery_{name}"):
                del st.session_state["gallery"][name]
                inference.save_gallery(st.session_state["gallery"])
                st.rerun()

        st.sidebar.markdown("### Identity search")
        st.sidebar.caption("Matches local directories only. No live internet search.")
        custom_search_dir = st.sidebar.text_input(
            "Search directory (optional)", value="", placeholder="/path/to/reference/photos",
            help="Extra directory of named reference photos to search, in addition to the bundled known_people/.",
        )

        @st.cache_resource
        def _load_known_people_gallery():
            """Embed the bundled known_people/ reference photos once per process."""
            net = models.recognition_nets.get("vggface")
            return inference.build_gallery_from_directory(models.face_net, net, inference.KNOWN_PEOPLE_DIR) if net else {}

        search_gallery = dict(_load_known_people_gallery())
        if custom_search_dir:
            net = models.recognition_nets.get("vggface")
            if net is not None:
                search_gallery.update(inference.build_gallery_from_directory(models.face_net, net, custom_search_dir))

    # Sidebar interface controls
    st.sidebar.markdown("### Control panel")
    conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.7)

    enable_crowd_count = st.sidebar.checkbox("Aggregate demographic summary", value=False, key="crowd_count_enabled")
    if enable_crowd_count:
        st.sidebar.caption(
            "Aggregates age/gender/race across every face detected in an image into a total count "
            "plus a breakdown per active model. Confirm local policy and consent before using this "
            "on images of people."
        )

    return SidebarState(
        conf_threshold=conf_threshold, face_detector=active_face_detector,
        active_age=active_age, active_gender=active_gender, active_race=active_race,
        active_emotion=active_emotion, active_glasses=active_glasses, active_mask=active_mask,
        active_hair_color=active_hair_color, active_eye_color=active_eye_color,
        active_recognition=active_recognition, active_liveness=active_liveness, active_gaze=active_gaze,
        active_colorization=active_colorization, active_face_landmarks=active_face_landmarks,
        active_hands=active_hands, search_gallery=search_gallery, enable_crowd_count=enable_crowd_count,
    )
