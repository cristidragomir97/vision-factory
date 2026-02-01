"""Dependency resolution for generated packages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackendConfig:
    """Configuration for an inference backend."""

    torch_index_url: str | None
    extra_deps: list[str] = field(default_factory=list)


@dataclass
class ResolvedDeps:
    """Resolved dependency set for a generated package."""

    pip_deps: list[str]
    torch_index_url: str | None
    ros_deps: list[str]
    warnings: list[str] = field(default_factory=list)


BACKENDS: dict[str, BackendConfig] = {
    "cuda": BackendConfig(
        torch_index_url="https://download.pytorch.org/whl/cu124",
    ),
    "rocm": BackendConfig(
        torch_index_url="https://download.pytorch.org/whl/rocm6.2",
    ),
    "openvino": BackendConfig(
        torch_index_url=None,
        extra_deps=["openvino>=2024.0"],
    ),
    "tensorrt": BackendConfig(
        torch_index_url="https://download.pytorch.org/whl/cu124",
        extra_deps=["tensorrt>=10.0"],
    ),
    "onnx": BackendConfig(
        torch_index_url=None,
        extra_deps=["onnxruntime>=1.17.0"],
    ),
}

BASE_DEPS = [
    "torch>=2.5.1",
    "torchvision>=0.20.1",
    "numpy>=1.21.6,<2.0",
    "opencv-python>=4.8.0",
    "pyyaml>=6.0",
]

ROS_DEPS = [
    "rclpy",
    "sensor_msgs",
    "std_msgs",
    "cv_bridge",
]

MODEL_DEPS: dict[str, list[str]] = {
    "yolo": ["ultralytics>=8.2.70"],
    "grounding_dino": ["transformers>=4.40,<5.0"],
    "depth_anything": ["transformers>=4.40,<5.0", "timm>=0.6.7"],
    "zoedepth": ["transformers>=4.40,<5.0", "timm>=0.6.7"],
    "segment_anything": ["segment-anything-2"],
    "florence": ["transformers>=4.40,<5.0", "einops"],
    "rtmpose": ["rtmlib", "onnxruntime"],
    "bytetrack": ["scipy", "lap==0.4.0", "cython==0.29.34"],
}

# Extra ROS deps per output type
OUTPUT_ROS_DEPS: dict[str, list[str]] = {
    "detections": ["vision_msgs"],
    "depth_map": [],
    "segmentation": [],
    "keypoints": [],
    "tracked_objects": ["vision_msgs"],
    "multi_task": ["vision_msgs"],
}


def resolve_dependencies(
    model_name: str,
    backend: str,
    output_type: str = "",
) -> ResolvedDeps:
    """Resolve the full dependency set for a model + backend combination.

    Args:
        model_name: Model name (e.g. "yolo")
        backend: Backend name (e.g. "cuda")
        output_type: Output type from manifest (e.g. "detections")

    Returns:
        ResolvedDeps with pip_deps, torch_index_url, ros_deps, warnings
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose from: {list(BACKENDS.keys())}")

    backend_cfg = BACKENDS[backend]
    warnings: list[str] = []

    # Build pip deps
    pip_deps = list(BASE_DEPS)
    pip_deps.extend(backend_cfg.extra_deps)
    pip_deps.extend(MODEL_DEPS.get(model_name, []))

    # Build ROS deps
    ros_deps = list(ROS_DEPS)
    ros_deps.extend(OUTPUT_ROS_DEPS.get(output_type, []))

    return ResolvedDeps(
        pip_deps=pip_deps,
        torch_index_url=backend_cfg.torch_index_url,
        ros_deps=ros_deps,
        warnings=warnings,
    )


def validate_selection(
    model_name: str,
    backend: str,
    variant: str,
    available_variants: list[str],
    supported_backends: list[str] | None = None,
) -> list[str]:
    """Validate a user's selections.

    Returns a list of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if backend not in BACKENDS:
        errors.append(f"Unknown backend '{backend}'. Choose from: {list(BACKENDS.keys())}")
    elif supported_backends and backend not in supported_backends:
        errors.append(
            f"Backend '{backend}' is not supported by model '{model_name}'. "
            f"Supported backends: {supported_backends}"
        )

    if variant not in available_variants:
        errors.append(
            f"Unknown variant '{variant}' for model '{model_name}'. "
            f"Choose from: {available_variants}"
        )

    return errors
