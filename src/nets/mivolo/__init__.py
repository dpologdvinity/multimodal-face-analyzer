"""
MiVOLO: Multi-input Transformer for Age and Gender Estimation
Official code from github.com/WildChlamydia/MiVOLO (Apache 2.0 License)

This module contains the official MiVOLO inference code with minimal modifications
for integration into multimodal-face-analyzer.

Key components:
  - model.mi_volo: MiVOLO model class for age/gender prediction
  - model.yolo_detector: YOLOv8 face+person detector wrapper
  - predictor: High-level Predictor class combining detector and model
  - structures: PersonAndFaceResult data structures
  - data.misc: Image preprocessing utilities
"""

from .model.mi_volo import MiVOLO
from .model.yolo_detector import Detector
from .predictor import Predictor
from .structures import PersonAndFaceResult, PersonAndFaceCrops

__all__ = [
    "MiVOLO",
    "Detector",
    "Predictor",
    "PersonAndFaceResult",
    "PersonAndFaceCrops",
]

__version__ = "2.0"
