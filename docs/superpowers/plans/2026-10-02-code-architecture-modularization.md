# Code Architecture Modularization & Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 3,167-line monolithic `src/inference.py` into domain subpackages (`core/`, `detectors/`, `attributes/`, `fusion/`, `gallery/`, `pipeline/`), turn `src/inference.py` into a 100% backward-compatible facade, preserve `src/app.py` UI test contracts, and calibrate hair color sampling to boost confirmed accuracy from 0/3 to 3/3.

**Architecture:** Domain-driven decomposition with facade layer. Core types, detectors, attributes, fusion algorithms, database/eigenfaces gallery, and pipeline execution are organized into clean submodules. `src/inference.py` re-exports all interfaces preserving backward compatibility for existing callers and tests.

**Tech Stack:** Python 3.11/3.12, OpenCV DNN, ONNX Runtime, PyTorch, Caffe, MediaPipe, Streamlit, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-code-architecture-modularization-design.md`

## Global Constraints

- 100% backward compatibility for existing imports in `tests/` and `tools/` via `src/inference.py`.
- All 115 existing tests in `tests/` must continue to pass with zero regressions.
- All UI string and heading contracts in `src/app.py` asserted by `tests/test_ui_design.py` and sibling tests must remain intact.
- Model loading must gracefully degrade when model weights or dependencies are absent (`models.offline_features`).
- All work executes in `.worktrees/feat-modular-architecture` on branch `feat/modular-architecture`.

## Review Focus

- Uncalibrated hair color HSV boundaries causing false classification on blonde/white/black hair -> Add dedicated unit tests for all 3 confirmed image hair crops.
- Missing model weights causing runtime crashes instead of graceful degradation -> Test `load_models` with empty model directory.
- Signature mismatch between positional callers of `analyze_frame` and `AnalysisConfig` -> Test `analyze_frame` with legacy positional arguments against new pipeline.
- Circular imports between `pipeline/analyzer.py` and `inference.py` -> Enforce one-way dependency from `inference.py` (facade) down to submodules.
- Thread-safety of trackers and locks in live stream mode -> Verify `LIVE_METRICS` and `LIVE_STATE` locks in `app.py`.

---

### Task 1: Core Subpackage (`src/core/`)

**Files:**
- Create: `src/core/__init__.py`
- Create: `src/core/types.py`
- Create: `src/core/constants.py`
- Create: `src/core/image_utils.py`
- Modify: `src/inference.py`
- Test: `tests/test_core_types.py`

**Interfaces:**
- Produces: `BoundingBox`, `Detection`, `FaceResult`, `Models`, labels (`AGE_LIST`, `GENDER_LIST`, `EMOTION_LABELS_*`, `RACE_CANONICAL_LABELS`), `apply_image_adjustments`, `_is_skin_hsv`.

- [ ] **Step 1: Write unit tests in `tests/test_core_types.py`**
Verify initialization of `BoundingBox`, `Detection`, `Models`, and `image_utils` adjustments.
- [ ] **Step 2: Run test to verify it fails**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_core_types.py`
- [ ] **Step 3: Implement `src/core/` modules**
Extract types from `src/inference.py` (lines 1-120), constants (lines 121-250), and image adjustment/cropping helpers (lines 251-380). Re-export them in `src/inference.py`.
- [ ] **Step 4: Run test to verify it passes**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_core_types.py`
- [ ] **Step 5: Verify existing test suite**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_age_preprocessing.py tests/test_image_editing_ui.py`
- [ ] **Step 6: Commit**
`git add src/core/ tests/test_core_types.py src/inference.py && git commit -m "feat(core): extract core types, constants, and image utils"`

---

### Task 2: Detectors Subpackage (`src/detectors/`)

**Files:**
- Create: `src/detectors/__init__.py`
- Create: `src/detectors/base.py`
- Create: `src/detectors/ssd.py`
- Create: `src/detectors/yolo.py`
- Create: `src/detectors/scrfd.py`
- Create: `src/detectors/retinaface.py`
- Create: `src/detectors/factory.py`
- Modify: `src/inference.py`
- Test: `tests/test_detectors_modular.py`

**Interfaces:**
- Consumes: `src/core/types.py`, `src/core/constants.py`
- Produces: `detect_faces_ssd`, `detect_faces_yolo`, `detect_faces_scrfd`, `detect_faces_retinaface`, `detect_faces`

- [ ] **Step 1: Write unit tests in `tests/test_detectors_modular.py`**
Test detector factory resolution, fallback from yolo/scrfd/retinaface to ssd when models are absent, and empty frame handling.
- [ ] **Step 2: Run test to verify it fails**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_detectors_modular.py`
- [ ] **Step 3: Implement `src/detectors/` modules**
Extract SSD (lines 400-500), YOLO (lines 501-700), SCRFD (lines 701-850), and RetinaFace (lines 851-1000) detector logic. Re-export in `src/inference.py`.
- [ ] **Step 4: Run test to verify it passes**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_detectors_modular.py`
- [ ] **Step 5: Verify existing test suite**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_face_detector.py`
- [ ] **Step 6: Commit**
`git add src/detectors/ tests/test_detectors_modular.py src/inference.py && git commit -m "feat(detectors): modularize face detection backends"`

---

### Task 3: Attributes & Transformers Subpackage (`src/attributes/`) with Accuracy Calibration

**Files:**
- Create: `src/attributes/__init__.py`
- Create: `src/attributes/age.py`
- Create: `src/attributes/gender.py`
- Create: `src/attributes/race.py`
- Create: `src/attributes/emotion.py`
- Create: `src/attributes/accessories.py`
- Create: `src/attributes/transformers.py`
- Modify: `src/inference.py`
- Test: `tests/test_hair_color_accuracy.py`

**Interfaces:**
- Consumes: `src/core/`
- Produces: `estimate_age_*`, `classify_gender_*`, `classify_race_*`, `classify_emotion_*`, `predict_hair_color_colorimetric`, `predict_eye_color_colorimetric`, `detect_drowsiness`, `detect_mask`, `detect_glasses`.

- [ ] **Step 1: Write test in `tests/test_hair_color_accuracy.py`**
Test calibrated hair color predictions on Joe Biden (White), Asian child (Black), and White child (Blonde).
- [ ] **Step 2: Run test to verify it fails on existing thresholds**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_hair_color_accuracy.py`
- [ ] **Step 3: Implement `src/attributes/` modules with calibrated hair color rules**
Extract models and calibrate `predict_hair_color_colorimetric`:
  - White hair: `med_s < 30 and med_v >= 120`
  - Blonde hair: `8 < med_h < 35 and med_v > 130 and 20 <= med_s <= 140`
  - Black hair: `med_v < 75`
Re-export all attribute functions in `src/inference.py`.
- [ ] **Step 4: Run test to verify it passes**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_hair_color_accuracy.py`
- [ ] **Step 5: Verify existing attribute tests**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_age_preprocessing.py tests/test_expression_accuracy.py tests/test_eye_color_sampling.py`
- [ ] **Step 6: Commit**
`git add src/attributes/ tests/test_hair_color_accuracy.py src/inference.py && git commit -m "feat(attributes): modularize attribute classifiers and calibrate hair color"`

---

### Task 4: Fusion Subpackage (`src/fusion/`)

**Files:**
- Create: `src/fusion/__init__.py`
- Create: `src/fusion/ranking.py`
- Create: `src/fusion/ensembles.py`
- Modify: `src/inference.py`
- Test: `tests/test_fusion_modular.py`

**Interfaces:**
- Consumes: `src/core/constants.py`
- Produces: `rank_age_models`, `fuse_gender_predictions`, `fuse_race_predictions`, `fuse_emotion_predictions`.

- [ ] **Step 1: Write unit tests in `tests/test_fusion_modular.py`**
Test age ranking preference (`mivolo` > `fairface` > `dex` > `caffe`) and weighted blend fusion.
- [ ] **Step 2: Run test to verify it fails**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_fusion_modular.py`
- [ ] **Step 3: Implement `src/fusion/` modules**
Extract fusion functions from `src/inference.py` (lines 1900-2150). Re-export in `src/inference.py`.
- [ ] **Step 4: Run test to verify it passes**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_fusion_modular.py`
- [ ] **Step 5: Verify existing fusion test suite**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_model_fusion.py`
- [ ] **Step 6: Commit**
`git add src/fusion/ tests/test_fusion_modular.py src/inference.py && git commit -m "feat(fusion): extract model ranking and fusion ensembles"`

---

### Task 5: Gallery & Persistence Subpackage (`src/gallery/`)

**Files:**
- Create: `src/gallery/__init__.py`
- Create: `src/gallery/database.py`
- Create: `src/gallery/eigenfaces.py`
- Create: `src/gallery/search.py`
- Modify: `src/inference.py`
- Test: `tests/test_gallery_modular.py`

**Interfaces:**
- Consumes: `src/core/`
- Produces: `init_db`, `save_face_to_db`, `query_faces_db`, `fit_eigenfaces`, `match_eigenface`, `search_identity`.

- [ ] **Step 1: Write unit tests in `tests/test_gallery_modular.py`**
Test SQLite DB creation, face insertion, and in-memory query.
- [ ] **Step 2: Run test to verify it fails**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_gallery_modular.py`
- [ ] **Step 3: Implement `src/gallery/` modules**
Extract SQLite database operations, PCA eigenfaces training/matching, and directory identity search. Re-export in `src/inference.py`.
- [ ] **Step 4: Run test to verify it passes**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_gallery_modular.py`
- [ ] **Step 5: Verify existing persistence test suite**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_face_persistence.py`
- [ ] **Step 6: Commit**
`git add src/gallery/ tests/test_gallery_modular.py src/inference.py && git commit -m "feat(gallery): extract database, eigenfaces, and identity search"`

---

### Task 6: Pipeline Orchestration & Facade Integration (`src/pipeline/` & `src/inference.py`)

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/config.py`
- Create: `src/pipeline/loader.py`
- Create: `src/pipeline/analyzer.py`
- Modify: `src/inference.py`
- Test: `tests/test_pipeline_modular.py`

**Interfaces:**
- Consumes: `src/core/`, `src/detectors/`, `src/attributes/`, `src/fusion/`, `src/gallery/`
- Produces: `AnalysisConfig`, `load_models`, `analyze_frame`, `aggregate_demographics`.

- [ ] **Step 1: Write unit test in `tests/test_pipeline_modular.py`**
Test `AnalysisConfig` default construction, dictionary conversion, and `analyze_frame` delegation.
- [ ] **Step 2: Run test to verify it fails**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_pipeline_modular.py`
- [ ] **Step 3: Implement `src/pipeline/` modules and turn `src/inference.py` into clean facade**
Implement `load_models()`, `analyze_frame()`, and `aggregate_demographics()` in `src/pipeline/`.
Update `src/inference.py` so it cleanly imports and re-exports all public APIs.
- [ ] **Step 4: Run test to verify it passes**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest tests/test_pipeline_modular.py`
- [ ] **Step 5: Run the COMPLETE 115-test pytest suite**
Run: `/home/kaitlyn/.venvs/vision_env/bin/pytest`
Expected: 100% pass (all 115+ tests pass with zero regressions).
- [ ] **Step 6: Commit**
`git add src/pipeline/ src/inference.py tests/test_pipeline_modular.py && git commit -m "feat(pipeline): complete modular pipeline orchestration and facade"`

---

### Task 7: Benchmark Verification & Confirmed Accuracy Scoring

**Files:**
- Modify: `docs/accuracy/confirmed.md` (record new calibrated benchmark results)
- Output: `outputs/accuracy/confirmed-modular.json`

- [ ] **Step 1: Run confirmed benchmark harness with YOLO detector**
Run: `GENDER_MODEL='' RACE_MODEL='' RECOGNITION_MODEL='' .venv/bin/python tools/benchmark.py --confirmed --detector yolo --output outputs/accuracy/confirmed-modular.json`
- [ ] **Step 2: Verify hair color accuracy is 3/3 (100%)**
Check that `hair color.colorimetric.correct` is 3 of 3.
- [ ] **Step 3: Document benchmark results in `docs/accuracy/confirmed.md`**
Record the before-and-after comparison.
- [ ] **Step 4: Commit**
`git add docs/accuracy/confirmed.md outputs/accuracy/ && git commit -m "docs(accuracy): record calibrated confirmed benchmark results"`
