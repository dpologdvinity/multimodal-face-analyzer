# Code Architecture Modularization & Accuracy Improvement Spec

## 1. Intent and Scope

The Multimodal Face Analyzer has grown into a rich computer-vision application with multi-model face detection, demographic inference, facial attributes, transformations, and persistence. However, `src/inference.py` has grown into a 3,167-line monolith with entangled responsibilities (network loading, preprocessing, detection, classification, fusion, eigenfaces PCA, SQLite DB operations, and rendering). Additionally, confirmed accuracy benchmarks revealed gaps in colorimetric hair color classification (0/3 correct) and DEX age predictions (0 answered).

This architectural specification details:
1. Decomposing `src/inference.py` into focused domain subpackages under `src/`.
2. Establishing `src/inference.py` as a complete backward-compatible facade ensuring zero broken imports across all 115 test files and CLI tools.
3. Preserving all `src/app.py` UI string contracts tested by the regression test suite.
4. Improving confirmed attribute accuracy (calibrating hair color sampling and DEX predictions).

---

## 2. Architecture & Component Boundaries

The domain logic is decomposed into six focused packages:

```
src/
├── core/
│   ├── types.py            # BoundingBox, Detection, FaceResult, Models dataclass
│   ├── constants.py        # Labels (AGE_LIST, GENDER_LIST, RACE_*, EMOTION_*) and thresholds
│   └── image_utils.py      # Crop padding, alignment, adjustments, skin HSV masks
├── detectors/
│   ├── base.py             # Base detector interface
│   ├── ssd.py              # OpenCV DNN TensorFlow SSD / ResNet-10 detector
│   ├── yolo.py             # YOLOv8-Face ONNX detector (with DFL / NMS decoding)
│   ├── scrfd.py            # SCRFD ONNX detector (distance-to-bbox regression)
│   ├── retinaface.py       # RetinaFace ONNX detector (variance-scaled anchor regression)
│   └── factory.py          # Unified detect_faces dispatcher
├── attributes/
│   ├── age.py              # Caffe, FairFace, DEX, MiVOLO age estimation
│   ├── gender.py           # Caffe, FairFace, DeepFace, MiVOLO gender classification
│   ├── race.py             # FairFace and DeepFace race classifiers
│   ├── emotion.py          # DAN, FER+, Mini-Xception, HSEmotion backends
│   ├── accessories.py      # Drowsiness, Glasses, Mask, Hair & Eye color sampling
│   └── transformers.py     # Whole-frame Colorization, Pose estimation, Hand landmarks, 3D Recon, Re-aging
├── fusion/
│   ├── ranking.py          # Age ranking (MiVOLO > FairFace > DEX > Caffe)
│   └── ensembles.py        # Weighted blend fusion (gender, race with top-2 split, emotion)
├── gallery/
│   ├── database.py         # SQLite schema, face record persistence (faces.db)
│   ├── eigenfaces.py       # PCA projection, model training, and recognition
│   └── search.py           # Local reference folder directory matching
├── pipeline/
│   ├── config.py           # AnalysisConfig dataclass helper
│   ├── loader.py           # load_models() scanner and offline feature tracker
│   └── analyzer.py         # analyze_frame() execution loop and demographic aggregation
├── inference.py            # Backward-compatible facade re-exporting all public symbols
└── app.py                  # Streamlit application preserving UI contract strings
```

---

## 3. Data Flow & Interface Contracts

### 3.1 `AnalysisConfig` and `analyze_frame` Signature
`src/pipeline/analyzer.py` implements the standard analysis pipeline:
```python
def analyze_frame(
    models: Models,
    frame: np.ndarray,
    conf_threshold: float = 0.5,
    active_age: set = ...,
    active_gender: set = ...,
    active_emotion: set = ...,
    active_race: set = ...,
    active_recognition: set = ...,
    gallery: dict = ...,
    active_glasses: set = ...,
    active_mask: set = ...,
    active_hair_color: set = ...,
    active_eye_color: set = ...,
    active_face_landmarks: set = ...,
    active_hands: set = ...,
    active_gaze: set = ...,
    global_adjustments: dict = ...,
    face_adjustments: dict = ...,
    face_detector: str = "yolo",
    metrics: dict | None = None,
    tracker: FaceTracker | None = None,
    liveness_tracker: LivenessTracker | None = None,
    active_liveness: set | None = None,
    config: AnalysisConfig | None = None,
) -> tuple[np.ndarray, list[dict], bool, bool]:
    ...
```

### 3.2 Facade Invariant (`src/inference.py`)
`src/inference.py` preserves:
- All function exports (`analyze_frame`, `load_models`, `predict_hair_color_colorimetric`, `predict_eye_color_colorimetric`, `save_face_to_db`, etc.).
- All data classes (`Models`, `BoundingBox`, `Detection`, `FaceResult`).
- All module-level constants (`AGE_LIST`, `GENDER_LIST`, `EMOTION_LABELS_DAN`, `RACE_CANONICAL_LABELS`, `BASE_DIR`, etc.).
- Direct re-exports guarantee that `import inference` and `from inference import ...` behave identically to the pre-refactor state.

### 3.3 UI Contract Invariant (`src/app.py`)
`src/app.py` retains:
- All required contract strings and headings checked by static tests:
  - `page_title="Multimodal Face Analyzer"`, `page_icon=":material/face:"`
  - `initial_sidebar_state="collapsed"`
  - `width="stretch"` and no `use_container_width`
  - Required labels: `"Theme"`, `### Model selection`, `"Detection"`, `"Classification"`, `"Identity and biometrics"`, `### Control panel`
  - Result metrics: `.metric("Faces detected"`, `.metric("Models active"`, `st.markdown("### Face details")`
  - Notice text: `"These outputs are model estimates, not biometric proof."`

---

## 4. Accuracy Improvements

### 4.1 Hair Color Sampling Calibration
The colorimetric hair classifier (`src/attributes/accessories.py`) currently scores 0/3 on `tools/benchmark.py --confirmed` because its thresholds are overly strict:
- **White Hair**: Relax median value threshold to allow ambient/shadowed white hair ($V \ge 120$ instead of $V > 180$) when saturation is low ($S < 30$).
- **Blonde Hair**: Expand saturation range to encompass pale blonde ($S \ge 25$ instead of $S > 60$) within yellow hue range ($8 < H < 32$).
- **Black Hair**: Raise value threshold to match typical exposure ($V < 75$ instead of $V < 50$).
- **Target**: Achieve 3/3 on the confirmed dataset (`joe-biden.jpg` $\rightarrow$ White, `5yr-asian-girl-happy.jpg` $\rightarrow$ Black, `3yr-white-girl-sad.jpg` $\rightarrow$ Blonde).

### 4.2 DEX Age Estimator Calibration
Ensure the DEX Caffe model outputs continuous expectation scores without returning unhandled exceptions or 0 answered in the benchmark harness.

---

## 5. Error Handling & Graceful Degradation

- Missing model files in `models/` continue to report via `models.offline_features` without crashing.
- Missing optional packages (`torch`, `onnxruntime`, `tensorflow`) gracefully disable dependent backends.
- Corrupt or empty image inputs raise standard validation errors handled cleanly by `app.py`.

---

## 6. Verification and Acceptance Criteria

1. **Full Pytest Suite**: All 115 existing tests in `tests/` pass with zero regressions (`/home/kaitlyn/.venvs/vision_env/bin/pytest`).
2. **Benchmark Verification**: `tools/benchmark.py --confirmed --detector yolo` executes cleanly and demonstrates improved scores for hair color (from 0/3 to 3/3).
3. **Module Independence**: Submodules in `src/detectors/`, `src/attributes/`, and `src/fusion/` can be imported and unit-tested without loading unrelated model weights.
4. **Git Isolation**: All work is tracked and verified inside the dedicated `.worktrees/feat-modular-architecture` worktree on branch `feat/modular-architecture`.
