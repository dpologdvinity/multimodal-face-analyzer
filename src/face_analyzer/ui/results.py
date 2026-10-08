"""Per-image results view: annotated image, exports, face cards, and per-face actions."""
from __future__ import annotations

import base64
import csv
import io
import json
from html import escape
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from .. import inference
from .adjustments import adjustment_sliders
from .sidebar import SidebarState
from .theme import THEME_ACCENTS

IMAGE_DISPLAY_WIDTH = 900


def _model_result_order(row: dict) -> tuple:
    """Sort key grouping rows by feature, with a fused answer ahead of its component models."""
    return (row["Feature"], row["Model"] not in inference.HEADLINE_MODEL_KEYS, row["Model"])


def target_card_html(face: dict) -> str:
    """Render one face's results as a HUD-style dossier card (native markup, not pixel text --
    keeps results legible no matter how many faces are packed into one image)."""
    rows = ""
    for result in sorted(face.get("model_results", []), key=_model_result_order):
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
    for result in sorted(face.get("model_results", []), key=_model_result_order):
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
            f'<div class="face-hover-info">{target_card_html(face)}</div></div>'
        )
    return (
        f'<div class="face-hover-image" style="aspect-ratio:{width}/{height}">'
        f'<img src="data:image/jpeg;base64,{source}" alt="Annotated image with detected faces">'
        f'{"".join(targets)}</div>'
    )


@st.dialog("IMAGE PREVIEW", width="large")
def _show_fullscreen_image(image_rgb: np.ndarray, caption: str) -> None:
    """Show the full-resolution image in a modal dialog."""
    st.image(image_rgb, caption=caption, width="stretch")


def _render_bounded_image(image_rgb: np.ndarray, caption: str, key: str, width: int = IMAGE_DISPLAY_WIDTH) -> None:
    """Keep routine output within the viewport while retaining a full-resolution modal view."""
    image_tools, _ = st.columns([1, 12])
    with image_tools:
        if st.button("Open image", key=f"fullscreen_{key}", help="Open the full-resolution image", width="content"):
            _show_fullscreen_image(image_rgb, caption)
    st.image(image_rgb, caption=caption, width=width)


def _render_photo_editor(
    frame_bgr: np.ndarray, identifier: str, adjustment_key: str, caption: str, *, theme: str,
) -> np.ndarray:
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
            adjustments = adjustment_sliders(
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


def process_and_display(
    models: Any, sidebar: SidebarState, frame: np.ndarray, identifier: str, *,
    theme: str, face_adjustments: dict,
) -> None:
    """Run detection/inference on frame and render result in Streamlit."""
    frame = _render_photo_editor(frame, identifier, "global_adj", "Source photo", theme=theme)
    frame, was_colorized = inference.maybe_colorize(models, frame, sidebar.active_colorization)

    active_labels = [
        name for name, active in (
            ("age", sidebar.active_age), ("gender", sidebar.active_gender),
            ("emotion", sidebar.active_emotion), ("race", sidebar.active_race),
            ("recognition", sidebar.active_recognition), ("glasses", sidebar.active_glasses),
            ("mask", sidebar.active_mask), ("hair color", sidebar.active_hair_color),
            ("eye color", sidebar.active_eye_color), ("gaze", sidebar.active_gaze),
            ("liveness", sidebar.active_liveness),
        ) if active
    ]
    spinner_text = f"Analyzing with {', '.join(active_labels)}..." if active_labels else "Detecting faces..."
    with st.spinner(spinner_text):
        annotated_frame, cropped_faces, has_faces, hands_detected = inference.analyze_frame(
            models, frame,
            sidebar.to_config(
                gallery=st.session_state.get("gallery", {}),
                global_adjustments={name: values[2] for name, values in inference.IMAGE_ADJUSTMENT_RANGES.items()},
                face_adjustments=face_adjustments,
            ),
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
        summary_cols[2].metric("Detector", sidebar.face_detector.upper())

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
    if sidebar.enable_crowd_count:
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
        inference.draw_recognition_scan(scan_frame, [(face["box"], match is not None) for face, match in zip(cropped_faces, matches, strict=False)])
        _render_bounded_image(
            cv2.cvtColor(scan_frame, cv2.COLOR_BGR2RGB), "Recognition scan", f"scan_{identifier}"
        )

        recognized_count = sum(match is not None for match in matches)
        st.caption(f"Recognition scan complete: {recognized_count}/{len(matches)} faces matched in saved faces.")
        for face, match in zip(cropped_faces, matches, strict=False):
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
                    individual_adjustments = adjustment_sliders(
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
            st.markdown(target_card_html(face), unsafe_allow_html=True)

            col_search, col_save = st.columns(2)
            if col_search.button("Search", key=f"search_btn_{identifier}_{face['idx']}"):
                face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                found = False
                if face["embedding"] is not None and sidebar.search_gallery:
                    match = inference.match_face_identity(
                        np.array(face["embedding"], dtype=np.float32), sidebar.search_gallery,
                    )
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
