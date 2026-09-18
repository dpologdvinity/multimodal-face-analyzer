"""
Adapter to load MiVOLO from HuggingFace safetensors checkpoint format.

The official MiVOLO code expects a .pth.tar checkpoint with embedded metadata
(min_age, max_age, avg_age, etc.). The HuggingFace version uses safetensors
with metadata in config.json. This adapter bridges the two formats.

License: Apache 2.0 (WildChlamydia/MiVOLO)
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from safetensors.torch import load_file

_logger = logging.getLogger("MiVOLO")


def create_hf_compatible_checkpoint(
    model_weights_path: str,
    config_path: str,
    temp_output_path: Optional[str] = None,
) -> str:
    """
    Convert HuggingFace safetensors + config.json format to a format compatible
    with MiVOLO's official loading code.

    Args:
        model_weights_path: Path to model.safetensors
        config_path: Path to config.json
        temp_output_path: Optional path to save converted checkpoint. If None, uses temp file.

    Returns:
        Path to the converted checkpoint file that can be loaded by MiVOLO().

    This creates a temporary .pth file with:
        - "state_dict": model weights from safetensors (with prefix stripped)
        - "min_age", "max_age", "avg_age": from config.json
        - "no_gender": inverted from config["only_age"]
        - "with_persons_model": from config.json
    """
    # Load metadata from config
    with open(config_path) as f:
        config = json.load(f)

    # Load model weights from safetensors
    weights = load_file(model_weights_path)

    # Strip "mivolo.model." prefix from keys to match MiVOLO's expected format
    state_dict = {}
    for key, value in weights.items():
        # Remove "mivolo.model." prefix if present
        if key.startswith("mivolo.model."):
            new_key = key.replace("mivolo.model.", "")
            state_dict[new_key] = value
        else:
            state_dict[key] = value

    # Prepare the checkpoint dict in MiVOLO's expected format
    checkpoint = {
        "state_dict": state_dict,
        "min_age": config.get("min_age", 0),
        "max_age": config.get("max_age", 122),
        "avg_age": config.get("avg_age", 61.0),
        "no_gender": config.get("only_age", False),  # Inverted: no_gender means only_age
        "with_persons_model": config.get("with_persons_model", True),
    }

    # Save to temporary or specified location
    if temp_output_path is None:
        fd, temp_output_path = tempfile.mkstemp(suffix=".pth", prefix="mivolo_")
        os.close(fd)

    torch.save(checkpoint, temp_output_path)
    _logger.info(f"Created compatible checkpoint at {temp_output_path}")

    return temp_output_path


def get_mivolo_config_from_hf(
    config_path: str,
) -> Dict[str, Any]:
    """Load MiVOLO configuration from HuggingFace config.json."""
    with open(config_path) as f:
        return json.load(f)


def load_mivolo_model_from_hf(
    model_weights_path: str,
    config_path: str,
    device: str = "cuda",
    use_persons: bool = True,
    disable_faces: bool = False,
    half: bool = True,
    verbose: bool = False,
):
    """
    Load MiVOLO model from HuggingFace format checkpoints.

    Args:
        model_weights_path: Path to model.safetensors
        config_path: Path to config.json
        device: Device to load model on (cuda, cpu, etc.)
        use_persons: Whether to use person/body crops in inference
        disable_faces: Whether to disable face crops
        half: Whether to use half precision (float16)
        verbose: Enable verbose logging

    Returns:
        Loaded MiVOLO model instance
    """
    from .model.mi_volo import MiVOLO

    # Create compatible checkpoint
    compat_checkpoint_path = create_hf_compatible_checkpoint(
        model_weights_path, config_path
    )

    try:
        # Load using official MiVOLO code
        model = MiVOLO(
            compat_checkpoint_path,
            device=device,
            use_persons=use_persons,
            disable_faces=disable_faces,
            half=half,
            verbose=verbose,
        )
        return model
    finally:
        # Clean up temp checkpoint
        if os.path.exists(compat_checkpoint_path):
            try:
                os.remove(compat_checkpoint_path)
            except:
                pass  # Ignore cleanup errors


def load_detector_from_hf(
    detector_weights_path: str,
    device: str = "cuda",
    half: bool = True,
    verbose: bool = False,
):
    """
    Load YOLOv8 face+person detector.

    Args:
        detector_weights_path: Path to yolov8x_person_face.pt
        device: Device to load on (cuda, cpu)
        half: Whether to use half precision
        verbose: Enable verbose logging

    Returns:
        Loaded Detector instance
    """
    from .model.yolo_detector import Detector

    # Note: The .pt file is a pickle-based PyTorch checkpoint.
    # For PyTorch 2.6+, this requires allow_pickle behavior.
    # The Detector class handles this internally via ultralytics.

    detector = Detector(
        detector_weights_path,
        device=device,
        half=half,
        verbose=verbose,
    )
    return detector
