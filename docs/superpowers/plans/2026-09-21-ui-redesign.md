# Face analyzer UI redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the existing Streamlit face analyzer into a polished, readable, responsive analysis workspace without changing inference behavior.

**Architecture:** Keep the existing single-page Streamlit flow and inference helpers. Apply changes in isolated worktrees, merging each tested branch into `master` in dependency order. Reuse existing theme classes, result helpers, and session-state keys.

**Tech Stack:** Python, Streamlit, existing HTML/CSS theme layer, pytest, agent-browser.

**Spec:** `docs/superpowers/specs/2026-09-21-ui-redesign-design.md`

## Global Constraints

- Preserve detector, model-selection, gallery, webcam, and thread-safety invariants.
- Preserve all current theme names and their visual differences.
- Do not add a frontend dependency.
- Replace deprecated `use_container_width` with `width="stretch"`.
- Keep accessible labels, keyboard focus, reduced-motion handling, and readable contrast.
- Each coherent fix gets its own `feature/ui-*` worktree and conventional commit.

## Review Focus

- Narrow mobile viewport: sidebar must not obscure the primary workflow; test in `390x844`.
- Light and dark themes: uploader, toolbar, buttons, captions, and focus states must retain contrast.
- Multiple uploaded files: each result stays visually grouped and export behavior remains intact.
- Zero detected faces: message must explain next action without breaking image preview.
- Live mode: camera and microphone permissions remain explicit; live state remains thread-safe.

### Task 1: Shared visual foundation

Files: `src/app.py`, `tests/test_ui_design.py`.

Add a stable page icon/title, fix global color selectors that cause white text on white Streamlit controls, remove deprecated width arguments, and normalize theme-safe control contrast.

Commit: `feat: establish polished ui foundation`

### Task 2: Responsive navigation and settings

Files: `src/app.py`, `tests/test_ui_design.py`.

Reduce sidebar visual noise, use sentence-case labels, add mobile-safe layout rules, and keep advanced settings grouped without changing selection state keys.

Commit: `fix: make analysis controls responsive`

### Task 3: Analysis workflow and results hierarchy

Files: `src/app.py`, `tests/test_ui_design.py`.

Make upload/camera entry points prominent, group each file result, add a concise result summary, move export actions after the visual result, and expose face details without hover dependency.

Commit: `feat: improve analysis result hierarchy`

### Task 4: Trust, copy, and live status

Files: `src/app.py`, `tests/test_ui_design.py`.

Replace terminal-style status copy, add visible model limitation/privacy guidance, improve camera/live labels, and make live metrics readable without changing callback behavior.

Commit: `fix: clarify analysis status and limitations`

### Task 5: Browser verification and merge

Run source tests, full pytest, desktop and mobile screenshots, interaction smoke test, console/error check, and accessibility audit. Merge each branch into `master` only after its focused tests pass.
