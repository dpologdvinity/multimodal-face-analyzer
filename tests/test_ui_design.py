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
