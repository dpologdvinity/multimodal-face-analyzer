"""
High-level inference wrapper for MiVOLO integration into multimodal-face-analyzer.

This module provides a simplified interface to MiVOLO's age and gender estimation
without requiring knowledge of the internal YOLOv8 detector or dual-input pipeline.

License: Apache 2.0 (WildChlamydia/MiVOLO)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .loader import load_mivolo_model_from_hf, load_detector_from_hf
from .data.misc import prepare_classification_images
from .structures import PersonAndFaceResult

_logger = logging.getLogger("MiVOLO")


class MiVOLOInference:
    """
    High-level inference interface for MiVOLO age/gender estimation.

    Usage:
        mivolo = MiVOLOInference(
            model_path="models/mivolo_v2.safetensors",
            config_path="...",
            detector_path="models/yolov8x_person_face.pt",
            device="cuda"
        )

        # Single face crop (no body context)
        age, gender, gender_conf = mivolo.predict_face(face_crop)

        # With optional body context for improved accuracy
        age, gender, gender_conf = mivolo.predict_face_with_body(face_crop, body_crop)
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        detector_path: Optional[str] = None,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize MiVOLO inference engine.

        Args:
            model_path: Path to model.safetensors (HuggingFace format)
            config_path: Path to config.json (HuggingFace format)
            detector_path: Optional path to yolov8x_person_face.pt detector
                          (only needed for detect_and_predict, not for direct face crop inference)
            device: Computation device (cuda, cpu, mps, etc.)
            half: Use float16 half precision (faster, slightly less accurate)
            verbose: Enable verbose logging
        """
        self.device = device
        self.verbose = verbose

        # Load MiVOLO model in face-only mode for direct face crop inference
        # (use_persons=False means we don't expect paired body crops, just faces)
        self.model = load_mivolo_model_from_hf(
            model_weights_path=model_path,
            config_path=config_path,
            device=device,
            use_persons=False,  # Face-only mode
            disable_faces=False,
            half=half,
            verbose=verbose,
        )

        # Optionally load detector for full-image inference
        self.detector = None
        if detector_path:
            try:
                self.detector = load_detector_from_hf(
                    detector_path, device=device, half=half, verbose=verbose
                )
            except Exception as e:
                _logger.warning(f"Failed to load detector: {e}")

        _logger.info(
            f"MiVOLO initialized: {self.model.input_size}x{self.model.input_size}, "
            f"device={device}, half={half}"
        )

    def predict_face(
        self, face_crop: np.ndarray
    ) -> Tuple[float, str, float]:
        """
        Predict age and gender for a single face crop.

        Args:
            face_crop: BGR image of a face (any size, will be resized)

        Returns:
            (age, gender, gender_confidence)
                age: float, estimated age in years
                gender: str, 'male' or 'female'
                gender_confidence: float, confidence in [0, 1]
        """
        # Preprocess face
        input_tensor = prepare_classification_images(
            [face_crop],
            target_size=self.model.input_size,
            mean=self.model.data_config["mean"],
            std=self.model.data_config["std"],
            device=self.device,
        )

        if input_tensor is None:
            raise ValueError("Failed to preprocess face crop")

        # In face-only mode, pad with zeros for missing body input to match training format
        if self.model.meta.with_persons_model and not self.model.meta.use_person_crops:
            body_zeros = torch.zeros_like(input_tensor)
            input_tensor = torch.cat((input_tensor, body_zeros), dim=1)

        # Run inference
        with torch.no_grad():
            if self.model.half:
                input_tensor = input_tensor.half()
            output = self.model.model(input_tensor)

        # Decode output
        age, gender, gender_conf = self._decode_output(output[0])
        return age, gender, gender_conf

    def predict_face_with_body(
        self,
        face_crop: np.ndarray,
        body_crop: Optional[np.ndarray] = None,
    ) -> Tuple[float, str, float]:
        """
        Predict age and gender for a face with optional body context.

        When body_crop is provided, MiVOLO uses dual-input cross-attention to
        combine face and body information for more accurate predictions.

        Args:
            face_crop: BGR image of face
            body_crop: BGR image of body/person (optional, can be None)

        Returns:
            (age, gender, gender_confidence)
        """
        # Preprocess both crops
        if body_crop is not None:
            # Use the dual-input model
            face_input = prepare_classification_images(
                [face_crop],
                target_size=self.model.input_size,
                mean=self.model.data_config["mean"],
                std=self.model.data_config["std"],
                device=self.device,
            )
            body_input = prepare_classification_images(
                [body_crop],
                target_size=self.model.input_size,
                mean=self.model.data_config["mean"],
                std=self.model.data_config["std"],
                device=self.device,
            )

            if face_input is None or body_input is None:
                raise ValueError("Failed to preprocess crops")

            # Concatenate on channel dimension (MiVOLO's dual-input format)
            model_input = torch.cat((face_input, body_input), dim=1)
        else:
            # Face-only input
            model_input = prepare_classification_images(
                [face_crop],
                target_size=self.model.input_size,
                mean=self.model.data_config["mean"],
                std=self.model.data_config["std"],
                device=self.device,
            )

        if model_input is None:
            raise ValueError("Failed to preprocess")

        # In face-only mode, pad with zeros for missing body input to match training format
        if model_input.shape[1] == 3 and self.model.meta.with_persons_model:
            body_zeros = torch.zeros_like(model_input)
            model_input = torch.cat((model_input, body_zeros), dim=1)

        # Run inference
        with torch.no_grad():
            if self.model.half:
                model_input = model_input.half()
            output = self.model.model(model_input)

        # Decode output
        age, gender, gender_conf = self._decode_output(output[0])
        return age, gender, gender_conf

    def _decode_output(self, output: torch.Tensor) -> Tuple[float, str, float]:
        """
        Decode MiVOLO model output into age and gender predictions.

        Args:
            output: Model output tensor of shape (3,) or (1,) depending on only_age

        Returns:
            (age, gender, gender_confidence)
        """
        if self.model.meta.only_age:
            # Only age output
            age_raw = output.item()
        else:
            # output = [gender_male_logit, gender_female_logit, age_raw]
            # First two dims are gender logits, last is age
            age_raw = output[-1].item()

        # Denormalize age
        age = age_raw * (
            self.model.meta.max_age - self.model.meta.min_age
        ) + self.model.meta.avg_age
        age = round(age, 2)

        if self.model.meta.only_age:
            # No gender output
            return age, None, None

        # Decode gender
        gender_logits = output[:2]
        gender_probs = torch.softmax(gender_logits, dim=0)
        gender_conf = gender_probs.max().item()
        gender_idx = gender_probs.argmax().item()

        gender = "male" if gender_idx == 0 else "female"

        return age, gender, gender_conf

    def detect_and_predict(
        self, image: np.ndarray
    ) -> list[Tuple[float, float, float, float, float, str, float]]:
        """
        Detect all faces in an image and predict age/gender.

        Requires detector to be loaded.

        Returns:
            List of (x1, y1, x2, y2, age, gender, gender_conf) for each face
        """
        if self.detector is None:
            raise RuntimeError("Detector not loaded. Provide detector_path in __init__")

        # Detect faces
        detected = self.detector.predict(image)

        results = []
        for face_idx in detected.get_bboxes_inds("face"):
            bbox = detected.get_bbox_by_ind(face_idx)
            x1, y1, x2, y2 = bbox.cpu().numpy()

            # Crop face
            face_crop = image[int(y1):int(y2), int(x1):int(x2)]

            # Predict
            age, gender, gender_conf = self.predict_face(face_crop)
            results.append((x1, y1, x2, y2, age, gender, gender_conf))

        return results


# Convenience functions for simple usage
def predict_age_gender_from_face(
    face_crop: np.ndarray,
    model_path: str = "models/mivolo_v2.safetensors",
    config_path: str = "models/mivolo_v2_config.json",
    device: str = "cuda",
) -> Tuple[float, str, float]:
    """
    Simple one-shot prediction for a face crop.

    Args:
        face_crop: BGR face image
        model_path: Path to MiVOLO checkpoint
        config_path: Path to MiVOLO config
        device: Computation device

    Returns:
        (age, gender, gender_confidence)
    """
    mivolo = MiVOLOInference(model_path, config_path, device=device)
    return mivolo.predict_face(face_crop)
