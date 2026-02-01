"""Build Jinja2 template context from manifest + user selections."""

from __future__ import annotations

from typing import Any

from .manifest import ModelManifest
from .resolver import ResolvedDeps


# Map model family to a Python class name for the generated model.py
MODEL_CLASS_MAP: dict[str, str] = {
    "ultralytics": "YoloModel",
    "depth": "DepthModel",
    "dino": "GroundingDinoModel",
    "sam": "SegmentAnythingModel",
    "florence": "FlorenceModel",
    "mmpose": "RtmPoseModel",
    "tracking": "ByteTrackTracker",
}

# Map backend to runner class name
RUNNER_CLASS_MAP: dict[str, str] = {
    "cuda": "CudaRunner",
    "rocm": "RocmRunner",
    "openvino": "OpenVinoRunner",
    "tensorrt": "TensorRtRunner",
    "onnx": "OnnxRunner",
}

# Map source.type to model_family for template branching
FAMILY_MAP: dict[str, str] = {
    "ultralytics": "ultralytics",
    "huggingface": "huggingface",
    "github": "custom",
    "mmpose": "onnx",
}


def _has_text_input(manifest: ModelManifest) -> bool:
    """Check if the model accepts text input."""
    inp = manifest.input
    # grounding_dino: input.secondary.type == "text"
    if "secondary" in inp and isinstance(inp["secondary"], dict):
        return inp["secondary"].get("type") == "text"
    # florence: input.text_input
    if "text_input" in inp:
        return True
    return False


def _has_prompt_input(manifest: ModelManifest) -> bool:
    """Check if the model accepts prompt input (points, boxes, masks)."""
    return "prompts" in manifest.input


def _get_model_class(manifest: ModelManifest) -> str:
    """Determine the Python class name for the model."""
    return MODEL_CLASS_MAP.get(manifest.model.family, "Model")


def _get_model_family(manifest: ModelManifest) -> str:
    """Determine the model family for template branching."""
    return FAMILY_MAP.get(manifest.source.type, "custom")


def build_context(
    model_name: str,
    backend: str,
    variant: str,
    package_name: str,
    manifest: ModelManifest,
    resolved_deps: ResolvedDeps,
) -> dict[str, Any]:
    """Build the full Jinja2 template context.

    This dict is passed to every template during rendering.
    """
    return {
        # Package identity
        "package_name": package_name,
        "node_name": f"{package_name}_node",

        # Model info
        "model_name": model_name,
        "model_family": _get_model_family(manifest),
        "model_class": _get_model_class(manifest),
        "model_display_name": manifest.model.name,
        "variant": variant,
        "variant_id": variant.replace(f"{model_name}_", ""),  # yolo_v8s -> v8s

        # Backend
        "backend": backend,
        "runner_class": RUNNER_CLASS_MAP[backend],

        # Output type drives node.py publisher logic
        "output_type": manifest.output.type,
        "output_format": manifest.output.format,
        "output_fields": [f.model_dump() for f in manifest.output.fields],

        # Input features
        "has_text_input": _has_text_input(manifest),
        "has_prompt_input": _has_prompt_input(manifest),

        # ROS config
        "publishers": [p.model_dump() for p in manifest.ros.publishers],
        "subscribers": [s.model_dump() for s in manifest.ros.subscribers],
        "services": [s.model_dump() for s in manifest.ros.services],
        "parameters": [p.model_dump() for p in manifest.ros.parameters],

        # Source info
        "source_type": manifest.source.type,
        "source_repo": manifest.source.repo,

        # Dependencies
        "pip_deps": resolved_deps.pip_deps,
        "torch_index_url": resolved_deps.torch_index_url,
        "ros_deps": resolved_deps.ros_deps,

        # Defaults
        "input_topic": "/camera/image_raw",
        "device_id": 0,
        "python_version": "3.10",
    }
