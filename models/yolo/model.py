"""YOLO model — default backend (CUDA/ROCm/OpenVINO)."""
from .model_base import YoloModel, YoloModel as Model

__all__ = ["YoloModel", "Model"]
