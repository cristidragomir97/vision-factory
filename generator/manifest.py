"""Pydantic models for parsing model manifest YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class ModelInfo(BaseModel):
    """Model identity and variants."""

    name: str
    family: str
    variants: list[str]
    default_variant: str


class SourceInfo(BaseModel):
    """Where to get the model weights."""

    model_config = ConfigDict(extra="allow")

    type: str  # "ultralytics", "huggingface", "github", "mmpose"
    repo: str
    alternatives: list[str] = []


class OutputField(BaseModel):
    """Single output field definition."""

    model_config = ConfigDict(extra="allow")

    name: str
    dtype: str
    shape: list[Any]
    description: str = ""


class OutputConfig(BaseModel):
    """Model output specification."""

    model_config = ConfigDict(extra="allow")

    type: str  # "detections", "depth_map", "segmentation", "keypoints", etc.
    format: str = ""
    fields: list[OutputField] = []


class RosPublisher(BaseModel):
    """ROS topic publisher definition."""

    model_config = ConfigDict(extra="allow")

    topic: str
    msg_type: str
    frame_id: str = "camera_frame"
    encoding: str = ""
    description: str = ""


class RosSubscriber(BaseModel):
    """ROS topic subscriber definition."""

    model_config = ConfigDict(extra="allow")

    topic: str
    msg_type: str
    description: str = ""


class RosService(BaseModel):
    """ROS service definition."""

    model_config = ConfigDict(extra="allow")

    name: str
    srv_type: str
    description: str = ""


class RosParameter(BaseModel):
    """ROS parameter definition."""

    model_config = ConfigDict(extra="allow")

    name: str
    type: str
    default: Any
    description: str = ""


class RosConfig(BaseModel):
    """ROS interface configuration."""

    publishers: list[RosPublisher] = []
    subscribers: list[RosSubscriber] = []
    services: list[RosService] = []
    parameters: list[RosParameter] = []


class ModelManifest(BaseModel):
    """Top-level model manifest.

    The `input` field is kept as a raw dict because manifests use
    different structures:
    - Simple: {type, format, preprocessing}
    - Multi-input: {primary: {...}, secondary: {...}}
    - Keyed: {image: {...}, prompts: {...}, task: {...}}
    - Non-image: {type: "detections", ...}
    """

    model_config = ConfigDict(extra="allow")

    model: ModelInfo
    source: SourceInfo
    input: dict[str, Any]
    output: OutputConfig
    postprocessing: dict[str, Any] = {}
    ros: RosConfig
    resources: dict[str, Any] = {}


def load_manifest(path: Path) -> ModelManifest:
    """Load and validate a single manifest YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelManifest.model_validate(data)


def load_all_manifests(manifests_dir: Path) -> dict[str, ModelManifest]:
    """Load all manifests from a directory.

    Returns a dict keyed by model name.
    """
    manifests: dict[str, ModelManifest] = {}
    for path in sorted(manifests_dir.glob("*.yaml")):
        manifest = load_manifest(path)
        manifests[manifest.model.name] = manifest
    return manifests
