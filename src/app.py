import base64
import threading
import time
from collections import deque
import csv
import io
import json
from html import escape

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from streamlit_webrtc import webrtc_streamer

import inference

# Module-scope shared state for live webcam stream, guarded by locks for thread-safe access
# from streamlit-webrtc callbacks running in separate threads.
LIVE_METRICS = deque(maxlen=120)
LIVE_METRICS_LOCK = threading.Lock()
LIVE_STATE = {"faces": [], "error": None, "updated": 0.0}
LIVE_STATE_LOCK = threading.Lock()
IMAGE_DISPLAY_WIDTH = 900

# Page setup and visual system
st.set_page_config(
    page_title="Multimodal Face Analyzer",
    page_icon=":material/face:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

THEME_MARKER_CLASSES = {
    "Light cyberpunk": "light-theme",
    "Amber Terminal": "amber-theme",
    "Synthwave": "synthwave-theme",
    "Phosphor Green": "phosphor-theme",
    "Brutalist": "brutalist-theme",
    "Corporate Slate": "corporate-theme",
    "Midnight Enterprise": "midnight-theme",
}
THEME_ACCENTS = {
    "Optical Bench": "#4fb8ac",
    "Light cyberpunk": "#087a52",
    "Amber Terminal": "#ffb02e",
    "Synthwave": "#ff2fb8",
    "Phosphor Green": "#6bffa0",
    "Brutalist": "#0a0a0a",
    "Corporate Slate": "#2f5aa8",
    "Midnight Enterprise": "#4f8ff0",
}

theme = st.sidebar.selectbox(
    "Theme",
    [
        "Optical Bench", "Light cyberpunk", "Amber Terminal", "Synthwave", "Phosphor Green",
        "Brutalist", "Corporate Slate", "Midnight Enterprise",
    ],
    key="theme",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:wght@400;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --base: #14181b;
        --surface: #1b2124;
        --surface-raised: #232a2e;
        --line: #38434798;
        --text: #eef2f0;
        --muted: #8ea3a2;
        --accent: #4fb8ac;
        --alert: #e8a33d;
    }

    body:has(.light-theme) {
        --base: #f3f8f6;
        --surface: #ffffff;
        --surface-raised: #e4f0eb;
        --line: #a9c5ba;
        --text: #16312a;
        --muted: #4f6d63;
        --accent: #087a52;
        --alert: #b42318;
    }

    body:has(.amber-theme) {
        --base: #120d05;
        --surface: #1d1409;
        --surface-raised: #2b1e0c;
        --line: #5a3f16;
        --text: #ffcf7a;
        --muted: #b8873f;
        --accent: #ffb02e;
        --alert: #ff5f4d;
    }

    body:has(.synthwave-theme) {
        --base: #170826;
        --surface: #23103a;
        --surface-raised: #2e1650;
        --line: #6b2e8f;
        --text: #f4e3ff;
        --muted: #c68fe6;
        --accent: #ff2fb8;
        --alert: #ffe45e;
    }
    body:has(.synthwave-theme) .stApp {
        background:
            linear-gradient(180deg, rgba(255, 47, 184, 0.08) 0%, transparent 40%),
            repeating-linear-gradient(0deg, rgba(198, 143, 230, 0.06) 0 1px, transparent 1px 32px),
            repeating-linear-gradient(90deg, rgba(198, 143, 230, 0.06) 0 1px, transparent 1px 32px),
            var(--base);
    }
    body:has(.synthwave-theme) .app-hero {
        flex-direction: column;
        align-items: center;
        text-align: center;
        border-left: none;
        border-bottom: 2px solid var(--accent);
        padding: 0 0 1.5rem;
        margin: 0 auto 2.5rem;
    }
    body:has(.synthwave-theme) .app-hero-readout { text-align: center; }
    body:has(.synthwave-theme) .app-hero h1 {
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        text-shadow: 0 0 12px var(--accent), 0 0 28px rgba(255, 47, 184, 0.6);
    }
    body:has(.synthwave-theme) div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        justify-content: center;
        border-bottom: 2px solid var(--line);
    }
    body:has(.synthwave-theme) button[role="tab"][aria-selected="true"] {
        text-shadow: 0 0 8px var(--accent);
    }
    body:has(.synthwave-theme) .stButton button,
    body:has(.synthwave-theme) .stDownloadButton button {
        border-radius: 999px !important;
        border: 1px solid var(--accent) !important;
        box-shadow: 0 0 10px rgba(255, 47, 184, 0.45);
    }

    body:has(.phosphor-theme) {
        --base: #010401;
        --surface: #061006;
        --surface-raised: #0a1a0a;
        --line: #1f6b1f;
        --text: #33ff66;
        --muted: #1f9c3f;
        --accent: #6bffa0;
        --alert: #ffcf33;
    }
    body:has(.phosphor-theme) .stApp,
    body:has(.phosphor-theme) .stApp * {
        font-family: 'IBM Plex Mono', monospace !important;
        letter-spacing: 0.01em;
    }
    body:has(.phosphor-theme) .stApp {
        background:
            repeating-linear-gradient(0deg, rgba(0, 0, 0, 0.35) 0 1px, transparent 1px 3px),
            var(--base);
    }
    body:has(.phosphor-theme) *,
    body:has(.phosphor-theme) .stButton button,
    body:has(.phosphor-theme) .stDownloadButton button,
    body:has(.phosphor-theme) input,
    body:has(.phosphor-theme) [data-baseweb] {
        border-radius: 0 !important;
        box-shadow: none !important;
        text-shadow: none;
    }
    body:has(.phosphor-theme) .app-hero {
        border-left: none;
        border: 1px solid var(--line);
        padding: 0.9rem 1.2rem;
        margin: 0 0 1.5rem;
    }
    body:has(.phosphor-theme) .app-hero h1 {
        font-size: 1.6rem;
        text-transform: uppercase;
    }
    body:has(.phosphor-theme) .app-hero h1::before { content: "> "; }
    body:has(.phosphor-theme) .app-hero p::before { content: "# "; }
    body:has(.phosphor-theme) .block-container {
        padding-top: 1.2rem;
    }

    body:has(.brutalist-theme) {
        --base: #f5f3ee;
        --surface: #ffffff;
        --surface-raised: #ffffff;
        --line: #0a0a0a;
        --text: #0a0a0a;
        --muted: #3a3a3a;
        --accent: #ffe500;
        --alert: #ff3b30;
    }
    body:has(.brutalist-theme) .stApp {
        background: var(--base);
        font-family: 'IBM Plex Mono', monospace !important;
    }
    body:has(.brutalist-theme) .app-hero {
        border: 4px solid var(--line);
        border-left: 4px solid var(--line);
        background: var(--accent);
        box-shadow: 8px 8px 0 var(--line);
        padding: 1.2rem 1.5rem;
        margin: 0 0 2.5rem;
    }
    body:has(.brutalist-theme) .app-hero h1 {
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0;
        color: var(--line);
    }
    body:has(.brutalist-theme) .app-hero p { color: var(--line); }
    body:has(.brutalist-theme) *,
    body:has(.brutalist-theme) .stButton button,
    body:has(.brutalist-theme) .stDownloadButton button,
    body:has(.brutalist-theme) [data-baseweb] {
        border-radius: 0 !important;
    }
    body:has(.brutalist-theme) .stButton button,
    body:has(.brutalist-theme) .stDownloadButton button {
        border: 3px solid var(--line) !important;
        box-shadow: 4px 4px 0 var(--line) !important;
        font-weight: 700;
        text-transform: uppercase;
    }
    body:has(.brutalist-theme) section[data-testid="stSidebar"] {
        background: var(--base) !important;
        border-right: 4px solid var(--line) !important;
    }
    body:has(.brutalist-theme) div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        border-bottom: 4px solid var(--line);
        gap: 0;
    }
    body:has(.brutalist-theme) button[role="tab"] {
        border: 3px solid var(--line) !important;
        border-bottom: none !important;
        text-transform: uppercase;
        font-weight: 700;
    }

    body:has(.corporate-theme) {
        --base: #f4f6f9;
        --surface: #ffffff;
        --surface-raised: #eef1f6;
        --line: #d7dce4;
        --text: #1c2733;
        --muted: #5b6b7c;
        --accent: #2f5aa8;
        --alert: #c0392b;
    }
    body:has(.corporate-theme) .stApp {
        background: var(--base);
    }
    body:has(.corporate-theme) .app-hero {
        border-left: none;
        border-bottom: 1px solid var(--line);
        padding: 0 0 1.4rem;
        margin: 0 0 2rem;
    }
    body:has(.corporate-theme) .app-hero h1 {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        letter-spacing: 0;
        color: var(--text);
    }
    body:has(.corporate-theme) .stApp h2,
    body:has(.corporate-theme) .stApp h3 {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
    }
    body:has(.corporate-theme) section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--line) !important;
    }
    body:has(.corporate-theme) section[data-testid="stSidebar"] h3 {
        color: var(--muted);
        font-family: 'DM Sans', sans-serif;
        text-transform: none;
        letter-spacing: 0.02em;
        font-weight: 600;
    }
    body:has(.corporate-theme) .stButton button,
    body:has(.corporate-theme) .stDownloadButton button {
        border-radius: 6px !important;
        border: 1px solid var(--line) !important;
        box-shadow: 0 1px 2px rgba(28, 39, 51, 0.08) !important;
    }
    body:has(.corporate-theme) [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(28, 39, 51, 0.06);
    }
    body:has(.corporate-theme) div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--line);
    }
    body:has(.corporate-theme) button[role="tab"][aria-selected="true"] {
        color: var(--accent);
        border-bottom: 2px solid var(--accent);
    }

    body:has(.midnight-theme) {
        --base: #0d1117;
        --surface: #161b22;
        --surface-raised: #1c2229;
        --line: #2d333b;
        --text: #e6edf3;
        --muted: #8b949e;
        --accent: #4f8ff0;
        --alert: #f85149;
    }
    body:has(.midnight-theme) .stApp {
        background: var(--base);
        font-family: 'DM Sans', sans-serif;
    }
    body:has(.midnight-theme) .app-hero {
        border-left: none;
        border-bottom: 1px solid var(--line);
        padding: 0 0 1.4rem;
        margin: 0 0 2rem;
    }
    body:has(.midnight-theme) .app-hero h1 {
        font-weight: 600;
        letter-spacing: 0;
    }
    body:has(.midnight-theme) section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--line) !important;
    }
    body:has(.midnight-theme) section[data-testid="stSidebar"] h3 {
        color: var(--muted);
        text-transform: none;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    body:has(.midnight-theme) .stButton button,
    body:has(.midnight-theme) .stDownloadButton button {
        border-radius: 6px !important;
        border: 1px solid var(--line) !important;
        background: var(--surface-raised) !important;
    }
    body:has(.midnight-theme) [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
    }
    body:has(.midnight-theme) div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--line);
    }
    body:has(.midnight-theme) button[role="tab"][aria-selected="true"] {
        color: var(--accent);
        border-bottom: 2px solid var(--accent);
    }

    .stApp {
        background: radial-gradient(circle at 85% 0%, color-mix(in srgb, var(--accent) 18%, transparent) 0, var(--base) 34rem);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2.5rem;
        padding-bottom: 5rem;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.01em;
    }
    .stApp,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stFileUploader"] label {
        color: var(--text);
    }
    .stApp [data-testid="stCaptionContainer"] p { color: var(--muted); }
    header[data-testid="stHeader"],
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        background: var(--surface) !important;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] button,
    header[data-testid="stHeader"] [data-testid="stToolbar"] button span {
        color: var(--text) !important;
    }
    .app-hero {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 2rem;
        border-left: none;
        border-bottom: 1px solid var(--line);
        padding: 0 0 1.1rem;
        margin: 0 0 2.2rem;
    }
    .app-hero-heading h1 {
        font-size: clamp(1.6rem, 2.4vw, 2.2rem);
        line-height: 1.15;
        margin: 0 0 0.35rem;
        font-weight: 600;
    }
    .app-hero-heading p { color: var(--muted); margin: 0; font-size: 0.95rem; max-width: 52ch; }
    .app-hero-readout {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--muted);
        text-align: right;
        white-space: nowrap;
        padding-bottom: 0.2rem;
    }
    .app-hero-readout strong { color: var(--accent); font-weight: 600; }
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h3 {
        color: var(--accent);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        margin-top: 1.7rem;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p { line-height: 1.4; }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.75rem;
        border-bottom: 1px solid var(--line);
    }
    div[data-testid="stTabs"] button[role="tab"] {
        color: var(--muted);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--accent);
    }
    div[data-testid="stFileUploader"] {
        border: 1px dashed var(--line);
        border-radius: 12px;
        background: var(--surface);
        padding: 1rem;
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: var(--surface-raised);
        border-color: var(--line);
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: var(--muted) !important;
    }
    div[data-testid="stFileUploader"] button {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: var(--base) !important;
    }
    div[data-testid="stFileUploader"] button * { color: var(--base) !important; }
    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--surface);
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary svg {
        background: var(--surface) !important;
        color: var(--text) !important;
        fill: var(--text) !important;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface-raised);
        color: var(--text);
        font-weight: 600;
        transition: background 120ms ease, border-color 120ms ease;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: var(--surface);
        border-color: var(--accent);
        color: var(--accent);
    }
    div[data-testid="stButton"] > button:focus-visible,
    div[data-testid="stDownloadButton"] > button:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid var(--line);
    }
    .target-card {
        position: relative;
        border: 1px solid var(--line);
        border-radius: 2px;
        background: var(--surface);
        padding: 1.1rem 1.25rem;
        margin: 0.9rem 0 1.35rem;
    }
    .target-card::before,
    .target-card::after {
        content: "";
        position: absolute;
        width: 14px;
        height: 14px;
        pointer-events: none;
    }
    .target-card::before {
        top: -1px;
        left: -1px;
        border-top: 2px solid var(--accent);
        border-left: 2px solid var(--accent);
    }
    .target-card::after {
        bottom: -1px;
        right: -1px;
        border-bottom: 2px solid var(--accent);
        border-right: 2px solid var(--accent);
    }
    .target-card-id {
        color: var(--accent);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        margin-bottom: 0.7rem;
    }
    .target-card-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.4rem 0;
        border-bottom: 1px solid var(--line);
        font-size: 0.88rem;
    }
    .target-card-row:last-child { border-bottom: none; }
    .target-card-row .k { color: var(--muted); }
    .target-card-row .v { color: var(--text); text-align: right; overflow-wrap: anywhere; }
    .face-hover-image {
        position: relative;
        width: min(100%, 900px);
        margin-bottom: 1rem;
    }
    .face-hover-image > img {
        display: block;
        width: 100%;
        height: 100%;
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    .face-hover-target {
        position: absolute;
        z-index: 1;
        cursor: help;
        border-radius: 4px;
    }
    .face-hover-target:hover,
    .face-hover-target:focus-visible {
        z-index: 3;
        outline: 2px solid var(--accent);
        background: color-mix(in srgb, var(--accent) 12%, transparent);
    }
    .face-hover-info {
        display: none;
        position: absolute;
        top: calc(100% + 0.5rem);
        left: 0;
        width: min(18rem, 75vw);
        z-index: 4;
    }
    .face-hover-target.place-right .face-hover-info { left: auto; right: 0; }
    .face-hover-target.place-up .face-hover-info { top: auto; bottom: calc(100% + 0.5rem); }
    .face-hover-target:hover .face-hover-info,
    .face-hover-target:focus .face-hover-info { display: block; }
    .face-hover-info .target-card {
        margin: 0;
        padding: 0.8rem;
        max-height: min(70vh, 22rem);
        overflow: auto;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
    }
    .face-hover-label {
        position: absolute;
        top: 0;
        left: 0;
        transform: translateY(-100%);
        background: rgba(0, 0, 0, 0.72);
        color: var(--accent);
        font-size: 0.65rem;
        padding: 2px 6px;
        border-radius: 4px 4px 0 0;
        white-space: nowrap;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        pointer-events: none;
        z-index: 2;
    }
    @media (max-width: 640px) {
        .block-container { padding: 1.25rem 1rem 3rem; }
        .app-hero { flex-direction: column; align-items: flex-start; gap: 0.6rem; margin-bottom: 1.5rem; }
        .app-hero-readout { text-align: left; }
        .target-card-row { display: block; }
        .target-card-row .v { display: block; text-align: left; margin-top: 0.15rem; }
    }
    @media (max-width: 760px) {
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div:first-child {
            width: min(86vw, 300px) !important;
            min-width: 0 !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.2rem; }
        div[data-testid="stTabs"] button[role="tab"] { flex: 1; padding: 0.7rem 0.5rem; }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            min-height: 9rem;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button { transition: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cache load_models so models persist across Streamlit reruns (avoids reloading expensive
# neural network weights for each interaction with sliders, buttons, tabs, etc.).
load_models = st.cache_resource(inference.load_models)

# Same "avoid redoing pure work on an unrelated rerun" reasoning as load_models above: any
# widget interaction (a model checkbox, an export button) reruns this whole script, which
# would otherwise re-decode (and re-downscale) the same uploaded/captured image bytes every
# time. max_entries bounds memory since each cached entry holds a full decoded frame.
decode_image_bytes = st.cache_data(max_entries=16)(inference.decode_image_bytes)


@st.cache_resource
def _get_face_tracker() -> inference.FaceTracker:
    """#2: one FaceTracker instance for the live webcam stream, cached (not session_state) so
    it's the same object across Streamlit reruns and reachable from streamlit-webrtc's own
    callback thread -- same reasoning as load_models() above, see FaceTracker's docstring."""
    return inference.FaceTracker()


@st.cache_resource
def _get_liveness_tracker() -> inference.LivenessTracker:
    """Keep blink history stable across Streamlit reruns for the LIVE webcam stream."""
    return inference.LivenessTracker()


@st.cache_resource
def _get_voice_fusion() -> inference.VoiceFaceFusion:
    """#10: same caching reasoning as _get_face_tracker() above -- the audio callback and the
    video callback are different threads and need to share the SAME buffer instance."""
    return inference.VoiceFaceFusion()

try:
    models = load_models()
except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

st.markdown(
    '<div class="app-hero">'
    '<div class="app-hero-heading"><h1>Multimodal Face Analyzer</h1>'
    '<p>Upload a photo or open your webcam, turn on the detectors you want, '
    'then read each face\'s results below.</p></div>'
    f'<div class="app-hero-readout">DETECTORS READY<br/>'
    f'<strong>{models.loaded_feature_count} / {models.total_feature_count}</strong></div>'
    '</div>',
    unsafe_allow_html=True,
)

MODEL_DISPLAY_NAMES = {
    "caffe": "Caffe",
    "ssrnet": "SSR-Net",
    "dex": "DEX",
    "mivolo": "MiVOLO",
    "fairface": "FairFace",
    "deepface": "DeepFace",
    "dan": "DAN",
    "efficientnet": "EfficientNet",
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


st.sidebar.markdown("### Model selection")
if models.offline_features:
    st.sidebar.caption(f"Unavailable: {', '.join(models.offline_features)}. Model file or dependency missing.")

with st.sidebar.expander("Detection", expanded=True):
    active_face_detector = "yolo" if models.yolo_face_nets else "ssd"
    _face_detector_options = (
        (["yolo"] if models.yolo_face_nets else [])
        + ["ssd"]
        + (["scrfd"] if models.scrfd_face_nets else [])
        + (["retinaface"] if models.retinaface_nets else [])
    )
    if len(_face_detector_options) > 1:
        active_face_detector = st.selectbox(
            "Face detector", _face_detector_options, index=_face_detector_options.index(active_face_detector),
            format_func=_display_model_name,
            help="One detector runs per frame. YOLO is preferred when loaded; SSD is the always-available fallback.",
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


def _reset_adjustments(prefixes: tuple[str, ...]) -> None:
    """Reset image adjustment sliders to their default values."""
    for state_key in list(st.session_state):
        if not any(state_key.startswith(f"{prefix}_") for prefix in prefixes):
            continue
        for adj_key, (_, _, default) in inference.IMAGE_ADJUSTMENT_RANGES.items():
            if state_key.endswith(f"_{adj_key}"):
                st.session_state[state_key] = default
                break


def _adjustment_sliders(caption: str, key_prefix: str, column_count: int = 2) -> dict:
    """Render brightness/contrast/saturation sliders and return their current values."""
    st.caption(caption)
    if st.button("Reset these sliders", key=f"{key_prefix}_reset"):
        _reset_adjustments((key_prefix,))
    values = {}
    columns = st.columns(column_count)
    for index, (adj_key, (adj_min, adj_max, adj_default)) in enumerate(inference.IMAGE_ADJUSTMENT_RANGES.items()):
        with columns[index % column_count]:
            values[adj_key] = st.slider(
                adj_key.replace("_", " ").title(), adj_min, adj_max, adj_default, key=f"{key_prefix}_{adj_key}"
            )
    return values


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

if theme in THEME_MARKER_CLASSES:
    st.markdown(f'<div class="{THEME_MARKER_CLASSES[theme]}"></div>', unsafe_allow_html=True)


def _target_card_html(face: dict) -> str:
    """Render one face's results as a HUD-style dossier card (native markup, not pixel text --
    keeps results legible no matter how many faces are packed into one image)."""
    rows = ""
    for result in sorted(face.get("model_results", []), key=lambda row: (row["Feature"], row["Model"])):
        feature = result["Feature"]
        model = result["Model"]
        label = feature if model == "derived" else f"{feature} ({model})"
        rows += f'<div class="target-card-row"><span class="k">{escape(label)}</span><span class="v">{escape(result["Output"])}</span></div>'
    if not rows:
        rows = '<div class="target-card-row"><span class="k">STATUS</span><span class="v">no model output</span></div>'
    return f'<div class="target-card"><div class="target-card-id">FACE {face["idx"]:02d}</div>{rows}</div>'


def _one_line_summary(face: dict) -> str:
    """Compact always-visible label for a face box -- feature: output pairs, no model names."""
    seen = {}
    for result in sorted(face.get("model_results", []), key=lambda row: (row["Feature"], row["Model"])):
        seen.setdefault(result["Feature"], result["Output"])
    if not seen:
        return f"F{face['idx']:02d}"
    parts = "  ".join(f"{feature}: {output}" for feature, output in seen.items())
    return f"F{face['idx']:02d}  {parts}"


def _hoverable_face_image(frame_bgr: np.ndarray, faces: list[dict]) -> str:
    """Render the annotated frame with focusable hover regions over detected boxes."""
    height, width = frame_bgr.shape[:2]
    success, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Could not encode annotated image")
    source = base64.b64encode(encoded).decode("ascii")
    targets = []
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
        y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
        placement = (" place-right" if x1 + x2 > width else "") + (" place-up" if y1 + y2 > height else "")
        style = f"left:{x1 / width * 100:.4f}%;top:{y1 / height * 100:.4f}%;width:{(x2 - x1) / width * 100:.4f}%;height:{(y2 - y1) / height * 100:.4f}%"
        targets.append(
            f'<div class="face-hover-target{placement}" style="{style}" tabindex="0" '
            f'aria-label="Face {face["idx"]}: hover or focus for details">'
            f'<div class="face-hover-label">{escape(_one_line_summary(face))}</div>'
            f'<div class="face-hover-info">{_target_card_html(face)}</div></div>'
        )
    return (
        f'<div class="face-hover-image" style="aspect-ratio:{width}/{height}">'
        f'<img src="data:image/jpeg;base64,{source}" alt="Annotated image with detected faces">'
        f'{"".join(targets)}</div>'
    )


@st.dialog("IMAGE PREVIEW", width="large")
def _show_fullscreen_image(image_rgb: np.ndarray, caption: str) -> None:
    st.image(image_rgb, caption=caption, width="stretch")


def _render_bounded_image(image_rgb: np.ndarray, caption: str, key: str, width: int = IMAGE_DISPLAY_WIDTH) -> None:
    """Keep routine output within the viewport while retaining a full-resolution modal view."""
    image_tools, _ = st.columns([1, 12])
    with image_tools:
        if st.button("Open image", key=f"fullscreen_{key}", help="Open the full-resolution image", width="content"):
            _show_fullscreen_image(image_rgb, caption)
    st.image(image_rgb, caption=caption, width=width)


def _render_photo_editor(frame_bgr: np.ndarray, identifier: str, adjustment_key: str, caption: str) -> np.ndarray:
    """Show an optional mouse crop and image-adjustment panel beside the source image."""
    cropped_key = f"cropped_photo_{identifier}"
    if cropped_key in st.session_state:
        frame_bgr = st.session_state[cropped_key]
    editing_key = f"photo_editor_{identifier}"
    image_col, editor_col = st.columns([3, 2])
    with editor_col:
        if st.button("Edit image", key=f"edit_image_{identifier}"):
            st.session_state[editing_key] = not st.session_state.get(editing_key, False)
        editing = st.session_state.get(editing_key, False)
        if editing:
            st.caption("Drag the crop rectangle, then adjust the preview before analysis.")
            adjustments = _adjustment_sliders(
                "Adjust the image before detection.", adjustment_key, column_count=1,
            )
        else:
            adjustments = {
                name: values[2] for name, values in inference.IMAGE_ADJUSTMENT_RANGES.items()
            }
    with image_col:
        if editing:
            source = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            cropped = st_cropper(
                source, realtime_update=True, box_color=THEME_ACCENTS.get(theme, "#76dfb1"), aspect_ratio=None,
                return_type="image", key=f"cropper_{identifier}",
            )
            edited_bgr = cv2.cvtColor(np.asarray(cropped), cv2.COLOR_RGB2BGR)
            if st.button("Crop photo", key=f"crop_photo_{identifier}"):
                st.session_state[f"cropped_photo_{identifier}"] = edited_bgr
                st.session_state[editing_key] = False
                st.rerun()
        else:
            edited_bgr = frame_bgr
            _render_bounded_image(cv2.cvtColor(edited_bgr, cv2.COLOR_BGR2RGB), caption, f"source_{identifier}")
    # Only run the expensive image adjustment if at least one slider is non-zero (non-default).
    if any(adjustments.values()):
        edited_bgr = inference.apply_image_adjustments(edited_bgr, adjustments)
    return edited_bgr


def process_and_display(frame: np.ndarray, identifier: str, conf_threshold: float) -> None:
    """Run detection/inference on frame and render result in Streamlit."""
    frame = _render_photo_editor(frame, identifier, "global_adj", "Source photo")
    frame, was_colorized = inference.maybe_colorize(models, frame, active_colorization)

    active_labels = [
        name for name, active in (
            ("age", active_age), ("gender", active_gender), ("emotion", active_emotion),
            ("race", active_race), ("recognition", active_recognition), ("glasses", active_glasses),
            ("mask", active_mask), ("hair color", active_hair_color), ("eye color", active_eye_color),
            ("gaze", active_gaze), ("liveness", active_liveness),
        ) if active
    ]
    spinner_text = f"Analyzing with {', '.join(active_labels)}..." if active_labels else "Detecting faces..."
    with st.spinner(spinner_text):
        annotated_frame, cropped_faces, has_faces, hands_detected = inference.analyze_frame(
            models, frame, conf_threshold, active_age, active_gender, active_emotion, active_race,
            active_recognition, st.session_state.get("gallery", {}),
            active_glasses, active_mask, active_hair_color, active_eye_color,
            active_face_landmarks, active_hands, active_gaze,
            {name: values[2] for name, values in inference.IMAGE_ADJUSTMENT_RANGES.items()}, face_adjustments,
            face_detector=active_face_detector,
            active_liveness=active_liveness,
        )

    if was_colorized:
        st.caption("Source converted from grayscale before analysis.")
    if hands_detected:
        st.caption("Hand landmarks detected and overlaid on the image.")

    if not has_faces:
        with st.container(border=True):
            st.warning(
                f"No face detected in {identifier}. Try a brighter, closer image or lower the confidence threshold."
            )
            _render_bounded_image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), identifier, f"annotated_{identifier}"
            )
        return

    st.markdown(f"### Results for `{identifier}`")

    with st.container(border=True):
        summary_cols = st.columns(3)
        summary_cols[0].metric("Faces detected", len(cropped_faces))
        summary_cols[1].metric("Models active", len(active_labels))
        summary_cols[2].metric("Detector", active_face_detector.upper())

    with st.container(border=True):
        st.caption("Model limitations")
        st.write(
            "These outputs are model estimates, not biometric proof. "
            "Do not use them as the sole basis for high-impact decisions."
        )

    export_rows = []
    for face in cropped_faces:
        export_rows.append({
            key: value for key, value in face.items()
            if key not in {"image", "embedding", "raw_columns"} and isinstance(value, (str, int, float, bool, list, type(None)))
        })
    export_json = json.dumps(export_rows, indent=2, default=str)
    csv_buffer = io.StringIO()
    if export_rows:
        fieldnames = sorted({key for row in export_rows for key in row})
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()} for row in export_rows)
    if enable_crowd_count:
        with st.expander(f"Aggregate summary: {len(cropped_faces)} faces detected", expanded=False):
            aggregate = inference.aggregate_demographics(cropped_faces)
            if not aggregate:
                st.caption("No age, gender, or race model is active. Enable one to see a breakdown.")
            for feature in inference.AGGREGATE_FEATURES:
                for model_key, counts in aggregate.get(feature, {}).items():
                    st.caption(f"{feature.upper()} ({model_key})")
                    st.bar_chart(counts)

    with st.container(border=True):
        st.caption("Hover or tap a face box for a quick preview. Review full details below.")
        image_tools, _ = st.columns([1, 12])
        with image_tools:
            if st.button(
                "Open image", key=f"fullscreen_annotated_{identifier}",
                help="Open the full-resolution image", width="content",
            ):
                _show_fullscreen_image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), identifier)
        st.markdown(_hoverable_face_image(annotated_frame, cropped_faces), unsafe_allow_html=True)

    export_col_json, export_col_csv = st.columns(2)
    export_col_json.download_button(
        "Download results as JSON", export_json, f"{identifier}_results.json", "application/json",
        key=f"json_dl_{identifier}",
    )
    export_col_csv.download_button(
        "Download results as CSV", csv_buffer.getvalue(), f"{identifier}_results.csv", "text/csv",
        key=f"csv_dl_{identifier}",
    )

    st.markdown("### Face details")

    if st.button("Scan all faces for recognition", key=f"scan_btn_{identifier}"):
        faces_bgr = [cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR) for face in cropped_faces]
        matches = inference.match_faces_eigenfaces_batch(faces_bgr)
        scan_frame = frame.copy()
        inference.draw_recognition_scan(scan_frame, [(face["box"], match is not None) for face, match in zip(cropped_faces, matches)])
        _render_bounded_image(
            cv2.cvtColor(scan_frame, cv2.COLOR_BGR2RGB), "Recognition scan", f"scan_{identifier}"
        )

        recognized_count = sum(match is not None for match in matches)
        st.caption(f"Recognition scan complete: {recognized_count}/{len(matches)} faces matched in saved faces.")
        for face, match in zip(cropped_faces, matches):
            if match:
                st.text(f"#{face['idx']}: Recognized -- saved face ID {match[0]} (distance {match[1]:.0f})")
            else:
                st.text(f"#{face['idx']}: Unrecognized")

    cols = st.columns(min(len(cropped_faces), 2))
    for i, face in enumerate(cropped_faces):
        with cols[i % len(cols)]:
            face_preview_col, face_editor_col = st.columns([3, 2])
            face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
            face_editor_open_key = f"face_editor_open_{identifier}_{face['idx']}"
            st.session_state.setdefault(face_editor_open_key, False)
            op_key = f"image_op_result_{identifier}_{face['idx']}"
            with face_editor_col:
                if st.button(
                    "Close editor" if st.session_state[face_editor_open_key] else "Edit face",
                    key=f"edit_face_toggle_{identifier}_{face['idx']}",
                ):
                    st.session_state[face_editor_open_key] = not st.session_state[face_editor_open_key]
                    st.rerun()
                if st.session_state[face_editor_open_key]:
                    individual_adjustments = _adjustment_sliders(
                        "Edit this crop only. Analysis labels use the detected crop.",
                        f"individual_adj_{identifier}_{face['idx']}",
                        column_count=1,
                    )
                else:
                    individual_adjustments = {
                        name: values[2] for name, values in inference.IMAGE_ADJUSTMENT_RANGES.items()
                    }
                edited_face_bgr = (
                    inference.apply_image_adjustments(face_bgr, individual_adjustments)
                    if any(individual_adjustments.values()) else face_bgr
                )
                if st.session_state[face_editor_open_key]:
                    st.download_button(
                        "Download edited face", cv2.imencode(".png", edited_face_bgr)[1].tobytes(),
                        file_name=f"face_{face['idx']}_edited.png", mime="image/png",
                        key=f"face_edit_dl_{identifier}_{face['idx']}",
                    )

                    op = st.selectbox("Image operation", inference.IMAGE_OP_OPTIONS, key=f"image_op_{identifier}_{face['idx']}")
                    op_params = {}
                    if op == "intensity":
                        op_params["method"] = st.selectbox("Intensity method", inference.INTENSITY_METHODS,
                                                            key=f"intensity_method_{identifier}_{face['idx']}")
                    elif op == "sharpen":
                        op_params["method"] = st.selectbox("Sharpen method", inference.SHARPEN_METHODS,
                                                            key=f"sharpen_method_{identifier}_{face['idx']}")
                    elif op == "denoise":
                        op_params["method"] = st.selectbox("Denoise method", inference.DENOISE_METHODS,
                                                            key=f"denoise_method_{identifier}_{face['idx']}")
                    if st.button("Apply image operation", key=f"image_op_btn_{identifier}_{face['idx']}"):
                        st.session_state[op_key] = inference.apply_image_op(edited_face_bgr, op, **op_params)
            with face_preview_col:
                _render_bounded_image(
                    cv2.cvtColor(edited_face_bgr, cv2.COLOR_BGR2RGB), f"Face {face['idx']:02d}",
                    f"face_{identifier}_{face['idx']}", width=360,
                )
                if st.session_state[face_editor_open_key] and op_key in st.session_state:
                    result = st.session_state[op_key]
                    _render_bounded_image(
                        cv2.cvtColor(result, cv2.COLOR_BGR2RGB), "Processed face",
                        f"processed_face_{identifier}_{face['idx']}", width=360,
                    )
                    st.download_button("Download face PNG", cv2.imencode(".png", result)[1].tobytes(),
                                       file_name=f"face_{face['idx']}_processed.png", mime="image/png",
                                       key=f"image_op_dl_{identifier}_{face['idx']}")
            st.markdown(_target_card_html(face), unsafe_allow_html=True)

            col_search, col_save = st.columns(2)
            if col_search.button("Search", key=f"search_btn_{identifier}_{face['idx']}"):
                face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                found = False
                if face["embedding"] is not None and search_gallery:
                    match = inference.match_face_identity(np.array(face["embedding"], dtype=np.float32), search_gallery)
                    if match:
                        st.success(f"Photo match: {match[0]} ({match[1] * 100:.0f}%).")
                        found = True
                eigen_match = inference.match_face_eigenfaces(face_bgr)
                if eigen_match:
                    st.success(f"Eigenface match: saved face {eigen_match[0]} (distance {eigen_match[1]:.0f}).")
                    found = True
                if not found:
                    st.warning("No match found in saved faces or local reference photos.")

            if col_save.button("Save face", key=f"save_btn_{identifier}_{face['idx']}"):
                face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                saved_id = inference.save_face(face_bgr, face["raw_columns"])
                st.info(f"Face saved with ID {saved_id}.")

            lbph_available = models.recognition_nets.get("lbph") is not None
            has_more_actions = (
                face["embedding"] is not None or lbph_available
                or models.reconstruction_3d_nets or models.age_progression_nets
            )
            if has_more_actions:
                with st.expander("More actions", expanded=False):
                    if face["embedding"] is not None or lbph_available:
                        enroll_name = st.text_input("Enroll as", key=f"enroll_name_{identifier}_{face['idx']}", placeholder="Enter a saved face name")
                        if st.button("Enroll", key=f"enroll_btn_{identifier}_{face['idx']}") and enroll_name:
                            try:
                                safe_name = inference.validate_lbph_name(enroll_name) if lbph_available else enroll_name.strip()
                                if not safe_name:
                                    raise ValueError("Enrollment name cannot be empty.")
                                if face["embedding"] is not None:
                                    st.session_state["gallery"][safe_name] = np.array(face["embedding"], dtype=np.float32)
                                    inference.save_gallery(st.session_state["gallery"])
                                if lbph_available:
                                    inference.enroll_lbph_face(safe_name, cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR))
                                st.rerun()
                            except ValueError as exc:
                                st.error(f"Could not enroll face: {exc}")

                    if models.reconstruction_3d_nets:
                        if st.button("Create 3D reconstruction", key=f"recon3d_btn_{identifier}_{face['idx']}"):
                            face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                            result = inference.run_3d_reconstruction(models, face_bgr)
                            if result is None:
                                st.warning("No reconstruction available. Face landmarks were not found in this crop.")
                            else:
                                vertices, faces, colors = result
                                obj_text = inference.mesh_to_obj_str(vertices, faces, colors)
                                st.download_button(
                                    "Download 3D mesh (.obj)", data=obj_text, file_name=f"face_{identifier}_{face['idx']}.obj",
                                    mime="text/plain", key=f"recon3d_dl_{identifier}_{face['idx']}",
                                )

                    if models.age_progression_nets:
                        st.caption("Age progression is for non-commercial research use only.")
                        col_src_age, col_tgt_age = st.columns(2)
                        source_age = col_src_age.number_input(
                            "Source age", min_value=0, max_value=100, value=30,
                            key=f"reage_src_{identifier}_{face['idx']}",
                        )
                        target_age = col_tgt_age.number_input(
                            "Target age", min_value=0, max_value=100, value=60,
                            key=f"reage_tgt_{identifier}_{face['idx']}",
                        )
                        if st.button("Run age progression", key=f"reage_btn_{identifier}_{face['idx']}"):
                            face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                            aged_bgr = inference.run_age_progression(models, face_bgr, source_age, target_age)
                            st.session_state[f"reage_result_{identifier}_{face['idx']}"] = aged_bgr
                        result_key = f"reage_result_{identifier}_{face['idx']}"
                        if result_key in st.session_state:
                            aged_bgr = st.session_state[result_key]
                            col_before, col_after = st.columns(2)
                            col_before.image(face["image"], caption="Before")
                            col_after.image(cv2.cvtColor(aged_bgr, cv2.COLOR_BGR2RGB), caption="After")
                            st.download_button(
                                "Download aged face", cv2.imencode(".png", aged_bgr)[1].tobytes(),
                                file_name=f"face_{identifier}_{face['idx']}_aged.png", mime="image/png",
                                key=f"reage_dl_{identifier}_{face['idx']}",
                            )


global_adjustments = {
    name: values[2] for name, values in inference.IMAGE_ADJUSTMENT_RANGES.items()
}
face_adjustments = {
    name: values[2] for name, values in inference.IMAGE_ADJUSTMENT_RANGES.items()
}

tab_upload, tab_webcam = st.tabs(["Image upload", "Webcam"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Choose images to analyze",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                frame = decode_image_bytes(uploaded_file.read())
            except ValueError as exc:
                st.error(f"Could not read {uploaded_file.name}: {exc}")
                continue
            process_and_display(frame, uploaded_file.name, conf_threshold)

with tab_webcam:
    capture_mode = st.segmented_control("Capture mode", ["Snapshot", "Live"], default="Snapshot")

    if capture_mode == "Snapshot":
        webcam_image = st.camera_input("Take a snapshot")

        if webcam_image:
            try:
                frame = decode_image_bytes(webcam_image.read())
            except ValueError as exc:
                st.error(f"Could not read camera image: {exc}")
                frame = None
            if frame is not None:
                process_and_display(frame, "WEBCAM_CAPTURE", conf_threshold)
    else:
        st.caption("Live analysis updates as people enter or leave view. Each face keeps a stable ID as it moves.")
        frame_skip = st.slider(
            "Classifier frame skip", 1, 10, 1,
            help="Run age/gender/emotion/race/recognition/etc. classifiers every Nth frame "
            "instead of every frame. Face detection and the hand/face-landmark overlays "
            "still run every frame, so the video stays smooth. These classifiers' outputs "
            "aren't otherwise drawn onto the LIVE video (see target cards in Image upload / "
            "SNAPSHOT for that), so skipping them here only reduces CPU load, with no visible "
            "staleness to interpolate around.",
        )
        # Counter for frame-skip logic: run slow classifiers every Nth frame to reduce
        # CPU load while keeping face detection (fast) and landmarks (smooth) at full rate.
        frame_counter = {"n": 0}
        _NO_MODELS: set = set()
        face_tracker = _get_face_tracker()
        liveness_tracker = _get_liveness_tracker()
        reset_col, voice_col = st.columns([1, 2])
        if reset_col.button("Reset tracking IDs", key="reset_tracking_ids"):
            face_tracker.reset()
            liveness_tracker.reset()

        # #10: off by default -- requesting the microphone is a permission prompt the user
        # didn't ask for just by opening the webcam tab, so it needs its own explicit opt-in
        # rather than riding along with LIVE mode's existing camera request.
        enable_voice_fusion = voice_col.checkbox(
            "Enable microphone-assisted fusion (experimental)", value=False, key="enable_voice_fusion",
            help="Heuristic only: cross-checks mic loudness against the largest face's emotion "
                 "label. Not a trained speech-emotion model -- see src/inference.py's "
                 "VoiceFaceFusion docstring for why.",
        )
        voice_fusion = _get_voice_fusion() if enable_voice_fusion else None
        if voice_fusion is not None and voice_col.button("Reset voice buffer", key="reset_voice_buffer"):
            voice_fusion.reset()
        gallery_snapshot = dict(st.session_state.get("gallery", {}))

        def _video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            """Process each video frame: run detection/inference, update LIVE state, handle voice fusion."""
            try:
                frame_started = time.perf_counter()
                metrics = {}
                img = frame.to_ndarray(format="bgr24")
                img, _ = inference.maybe_colorize(models, img, active_colorization)
                frame_counter["n"] += 1
                run_classifiers = frame_counter["n"] % frame_skip == 0
                # Pass empty model sets if classifiers are skipped this frame; face detection
                # still runs (always fast), so video remains smooth while expensive classifiers run sparse.
                annotated_frame, cropped_faces, _, _ = inference.analyze_frame(
                    models, img, conf_threshold,
                    active_age if run_classifiers else _NO_MODELS,
                    active_gender if run_classifiers else _NO_MODELS,
                    active_emotion if run_classifiers else _NO_MODELS,
                    active_race if run_classifiers else _NO_MODELS,
                    active_recognition if run_classifiers else _NO_MODELS, gallery_snapshot,
                    active_glasses if run_classifiers else _NO_MODELS,
                    active_mask if run_classifiers else _NO_MODELS,
                    active_hair_color if run_classifiers else _NO_MODELS,
                    active_eye_color if run_classifiers else _NO_MODELS,
                    active_face_landmarks, active_hands,
                    active_gaze if run_classifiers else _NO_MODELS,
                    global_adjustments, face_adjustments,
                    face_detector=active_face_detector, metrics=metrics, tracker=face_tracker,
                    liveness_tracker=liveness_tracker,
                    active_liveness=active_liveness,
                )
                metrics["frame_ms"] = (time.perf_counter() - frame_started) * 1000
                metrics["timestamp"] = time.monotonic()
                live_faces = [
                    {key: face[key] for key in ("idx", "model_results")}
                    for face in cropped_faces
                ]
                with LIVE_METRICS_LOCK:
                    LIVE_METRICS.append(metrics)
                with LIVE_STATE_LOCK:
                    LIVE_STATE.update(faces=live_faces, error=None, updated=time.monotonic())
                if voice_fusion is not None and cropped_faces:
                    # v1 scope: fuse against the single largest detected face.
                    largest = max(cropped_faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
                    emotion_label = largest["emotion"][0] if largest["emotion"] else None
                    voice_arousal = voice_fusion.current_arousal()
                    voice_fusion.set_latest_status({
                        "voice_arousal": voice_arousal,
                        "emotion": emotion_label,
                        "consistency": inference.fuse_voice_and_emotion(voice_arousal, emotion_label) if emotion_label else None,
                    })
                return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            except Exception as exc:
                # Inference failure must not crash the WebRTC video stream. Log the error to
                # LIVE_STATE so the UI thread can display it, but always return a frame
                # (raw passthrough) to keep the stream alive and the camera usable.
                with LIVE_STATE_LOCK:
                    LIVE_STATE["error"] = f"{type(exc).__name__}: {exc}"
                return frame

        def _audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
            """Ingest audio samples into voice fusion tracker if enabled."""
            if voice_fusion is not None:
                samples = inference.audio_frame_to_mono_float(frame.to_ndarray())
                voice_fusion.ingest_audio(samples, frame.sample_rate)
            return frame

        webrtc_ctx = webrtc_streamer(
            key="live-face-feed",
            video_frame_callback=_video_frame_callback,
            audio_frame_callback=_audio_frame_callback if enable_voice_fusion else None,
            media_stream_constraints={"video": True, "audio": enable_voice_fusion},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

        if voice_fusion is not None:
            status = voice_fusion.get_latest_status()
            if status is None:
                st.caption("Microphone fusion is waiting for audio and a detected face…")
            else:
                consistency_text = status["consistency"] or "n/a (emotion label not categorized)"
                st.caption(
                    f"Microphone fusion: {status['voice_arousal']}; largest face emotion: "
                    f"{status['emotion']}; consistency: {consistency_text}."
                )

        live_info = st.empty()
        live_fps = st.empty()
        while webrtc_ctx.state.playing:
            with LIVE_STATE_LOCK:
                live_snapshot = dict(LIVE_STATE)
            if live_snapshot["error"]:
                live_info.error(f"Live frame error: {live_snapshot['error']}")
            elif live_snapshot["faces"]:
                with live_info.container():
                    st.markdown("#### Live face details")
                    for face in live_snapshot["faces"]:
                        st.markdown(_target_card_html(face), unsafe_allow_html=True)
            else:
                live_info.caption("Waiting for a detected face…")
            with LIVE_METRICS_LOCK:
                live_metrics = list(LIVE_METRICS)
            intervals = np.diff([item["timestamp"] for item in live_metrics[-30:]])
            fps = 1.0 / float(np.mean(intervals)) if len(intervals) and np.mean(intervals) > 0 else 0.0
            live_fps.metric("Live FPS", f"{fps:.1f}")
            time.sleep(0.25)
        with LIVE_METRICS_LOCK:
            live_metrics = list(LIVE_METRICS)
        if live_metrics:
            latency_rows = []
            for item in live_metrics:
                for model_name, values in item.get("model_latency_ms", {}).items():
                    latency_rows.extend({"Model": model_name, "Latency (ms)": value} for value in values)
            if latency_rows:
                latency_frame = pd.DataFrame(latency_rows)
                summary = latency_frame.groupby("Model", as_index=False)["Latency (ms)"].mean()
                summary["Latency (ms)"] = summary["Latency (ms)"].round(1)
                st.dataframe(summary, hide_index=True, width="stretch")
            emotion_rows = []
            start_time = live_metrics[0]["timestamp"]
            for item in live_metrics:
                for sample in item.get("emotion_samples", []):
                    emotion_rows.append({
                        "Seconds": item["timestamp"] - start_time,
                        "Model": sample["model"],
                        "Emotion": sample["emotion"],
                    })
            if emotion_rows:
                st.markdown("#### Emotion over time")
                emotion_frame = pd.DataFrame(emotion_rows)
                for model_name, model_frame in emotion_frame.groupby("Model"):
                    labels = sorted(model_frame["Emotion"].unique())
                    chart_rows = []
                    for _, row in model_frame.iterrows():
                        chart_rows.append({
                            "Seconds": row["Seconds"],
                            **{label: float(label == row["Emotion"]) for label in labels},
                        })
                    chart = pd.DataFrame(chart_rows).groupby("Seconds").max().sort_index()
                    st.caption(f"{model_name}: dominant emotion (1 = active, 0 = inactive)")
                    st.line_chart(chart, height=220)
