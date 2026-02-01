"""Shared fixtures for generator tests."""

from pathlib import Path

import pytest

from generator.manifest import load_manifest, load_all_manifests


MODELS_DIR = Path(__file__).parent.parent / "models"


@pytest.fixture
def models_dir() -> Path:
    return MODELS_DIR


@pytest.fixture
def all_manifests(models_dir):
    return load_all_manifests(models_dir)


@pytest.fixture
def yolo_manifest(models_dir):
    return load_manifest(models_dir / "yolo" / "manifest.yaml")


@pytest.fixture
def depth_manifest(models_dir):
    return load_manifest(models_dir / "depth_anything" / "manifest.yaml")


@pytest.fixture
def dino_manifest(models_dir):
    return load_manifest(models_dir / "grounding_dino" / "manifest.yaml")
