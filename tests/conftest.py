"""Shared fixtures for generator tests."""

from pathlib import Path

import pytest

from generator.manifest import load_manifest, load_all_manifests


MANIFESTS_DIR = Path(__file__).parent.parent / "manifests"


@pytest.fixture
def manifests_dir() -> Path:
    return MANIFESTS_DIR


@pytest.fixture
def all_manifests(manifests_dir):
    return load_all_manifests(manifests_dir)


@pytest.fixture
def yolo_manifest(manifests_dir):
    return load_manifest(manifests_dir / "yolo.yaml")


@pytest.fixture
def depth_manifest(manifests_dir):
    return load_manifest(manifests_dir / "depth_anything.yaml")


@pytest.fixture
def dino_manifest(manifests_dir):
    return load_manifest(manifests_dir / "grounding_dino.yaml")
