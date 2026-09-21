from pathlib import Path


APP_SOURCE = (Path(__file__).parents[1] / "src" / "app.py").read_text()


def test_page_has_product_title_and_face_icon():
    assert 'page_title="Multimodal Face Analyzer"' in APP_SOURCE
    assert 'page_icon=":material/face:"' in APP_SOURCE


def test_ui_does_not_force_one_color_on_every_span():
    assert ".stApp p, .stApp label, .stApp span" not in APP_SOURCE


def test_ui_uses_current_streamlit_width_api():
    assert "use_container_width" not in APP_SOURCE
    assert 'width="stretch"' in APP_SOURCE


def test_primary_workflow_starts_with_collapsed_settings():
    assert 'initial_sidebar_state="collapsed"' in APP_SOURCE


def test_navigation_and_settings_use_sentence_case():
    for label in (
        '"Theme"',
        '### Model selection',
        '"Detection"',
        '"Classification"',
        '"Identity and biometrics"',
        '### Control panel',
    ):
        assert label in APP_SOURCE


def test_results_have_visual_summary_and_visible_face_details():
    assert '.metric("Faces detected"' in APP_SOURCE
    assert '.metric("Models active"' in APP_SOURCE
    assert 'st.markdown("### Face details")' in APP_SOURCE
    assert "Hover or tap a face box for a quick preview. Review full details below." in APP_SOURCE


def test_result_actions_use_clear_labels():
    assert '"Open image"' in APP_SOURCE
    assert '"Download results as JSON"' in APP_SOURCE
    assert '"Download results as CSV"' in APP_SOURCE
    assert '"Scan all faces for recognition"' in APP_SOURCE


def test_results_explain_model_limits():
    assert "These outputs are model estimates, not biometric proof." in APP_SOURCE
    assert "Do not use them as the sole basis for high-impact decisions." in APP_SOURCE


def test_live_controls_use_readable_status_labels():
    assert 'st.segmented_control("Capture mode"' in APP_SOURCE
    assert '"Classifier frame skip"' in APP_SOURCE
    assert 'live_fps.metric("Live FPS", f"{fps:.1f}")' in APP_SOURCE
    assert "Waiting for a detected face…" in APP_SOURCE


def test_streamlit_chrome_and_upload_instructions_keep_theme_contrast():
    assert '[data-testid="stToolbar"]' in APP_SOURCE
    assert '[data-testid="stFileUploaderDropzoneInstructions"]' in APP_SOURCE
    assert '[data-testid="stExpander"] summary' in APP_SOURCE


def test_model_labels_preserve_canonical_product_names():
    assert "MODEL_DISPLAY_NAMES" in APP_SOURCE
    for label in ("SSR-Net", "MiVOLO", "FairFace", "DeepFace", "HSEmotion", "MediaPipe"):
        assert f'"{label}"' in APP_SOURCE
    assert "format_func=_display_model_name" in APP_SOURCE


def test_brutalist_theme_keeps_accent_actions_and_readouts_legible():
    assert 'body:has(.brutalist-theme) .app-hero-readout strong' in APP_SOURCE
    assert 'body:has(.brutalist-theme) div[data-testid="stFileUploader"] button *' in APP_SOURCE


def test_camera_permission_panel_uses_active_theme_tokens():
    assert '[data-testid="stCameraInputWebcamComponent"] > div:first-child' in APP_SOURCE
    assert '[data-testid="stCameraInputWebcamComponent"] p' in APP_SOURCE
    assert '[data-testid="stCameraInputButton"]' in APP_SOURCE
