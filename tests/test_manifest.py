"""Tests for manifest loading and validation."""

import pytest
from pathlib import Path

from generator.manifest import load_manifest, load_all_manifests, ModelManifest


class TestLoadManifest:
    """Test loading individual manifest files."""

    def test_load_yolo(self, yolo_manifest):
        assert yolo_manifest.model.name == "yolo"
        assert yolo_manifest.model.family == "ultralytics"
        assert "yolo_v8s" in yolo_manifest.model.variants
        assert yolo_manifest.model.default_variant == "yolo_v8s"
        assert yolo_manifest.output.type == "detections"

    def test_load_depth_anything(self, depth_manifest):
        assert depth_manifest.model.name == "depth_anything"
        assert depth_manifest.model.family == "depth"
        assert depth_manifest.output.type == "depth_map"
        assert depth_manifest.output.format == "relative"

    def test_load_grounding_dino(self, dino_manifest):
        assert dino_manifest.model.name == "grounding_dino"
        assert dino_manifest.source.type == "huggingface"
        assert dino_manifest.output.type == "detections"
        # Grounding DINO has multi-input (primary + secondary)
        assert "primary" in dino_manifest.input or "type" in dino_manifest.input

    def test_yolo_source(self, yolo_manifest):
        assert yolo_manifest.source.type == "ultralytics"
        assert yolo_manifest.source.repo == "ultralytics/ultralytics"

    def test_yolo_output_fields(self, yolo_manifest):
        field_names = [f.name for f in yolo_manifest.output.fields]
        assert "boxes" in field_names
        assert "scores" in field_names
        assert "class_ids" in field_names

    def test_yolo_ros_config(self, yolo_manifest):
        assert len(yolo_manifest.ros.publishers) > 0
        assert len(yolo_manifest.ros.parameters) > 0
        topic_names = [p.topic for p in yolo_manifest.ros.publishers]
        assert "detections" in topic_names

    def test_yolo_resources(self, yolo_manifest):
        assert "vram" in yolo_manifest.resources
        assert yolo_manifest.resources["vram"]["yolo_v8s"] == 400


class TestLoadAllManifests:
    """Test loading all manifests from directory."""

    def test_loads_all_manifests(self, all_manifests):
        assert len(all_manifests) == 8

    def test_expected_models_present(self, all_manifests):
        expected = {
            "yolo", "depth_anything", "grounding_dino",
            "segment_anything", "florence", "rtmpose",
            "zoedepth", "bytetrack",
        }
        assert set(all_manifests.keys()) == expected

    def test_all_have_required_fields(self, all_manifests):
        for name, manifest in all_manifests.items():
            assert manifest.model.name, f"{name}: missing model.name"
            assert manifest.model.variants, f"{name}: no variants"
            assert manifest.model.default_variant, f"{name}: no default_variant"
            assert manifest.source.type, f"{name}: missing source.type"
            assert manifest.output.type, f"{name}: missing output.type"
            assert manifest.ros.publishers, f"{name}: no publishers"

    def test_all_default_variants_are_valid(self, all_manifests):
        for name, manifest in all_manifests.items():
            assert manifest.model.default_variant in manifest.model.variants, (
                f"{name}: default_variant '{manifest.model.default_variant}' "
                f"not in variants {manifest.model.variants}"
            )

    def test_varied_input_structures(self, all_manifests):
        """Manifests have different input structures - all should parse."""
        # Simple flat input
        assert "type" in all_manifests["yolo"].input
        # Multi-input with primary/secondary
        assert "primary" in all_manifests["grounding_dino"].input
        # Keyed input (image + prompts)
        assert "image" in all_manifests["segment_anything"].input
        # Non-image input
        assert all_manifests["bytetrack"].input["type"] == "detections"


class TestInvalidManifest:
    """Test error handling for invalid manifests."""

    def test_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.yaml")

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(Exception):
            load_manifest(empty)

    def test_missing_model_field(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("source:\n  type: test\n  repo: test/test\n")
        with pytest.raises(Exception):
            load_manifest(bad)


class TestBackendsField:
    """Test backends field on manifests."""

    def test_yolo_has_tensorrt(self, yolo_manifest):
        assert "tensorrt" in yolo_manifest.model.backends
        assert "cuda" in yolo_manifest.model.backends

    def test_depth_anything_no_tensorrt(self, depth_manifest):
        assert "tensorrt" not in depth_manifest.model.backends
        assert "cuda" in depth_manifest.model.backends

    def test_all_manifests_have_backends(self, all_manifests):
        for name, manifest in all_manifests.items():
            assert len(manifest.model.backends) > 0, f"{name}: no backends"

    def test_default_backends_when_omitted(self, tmp_path):
        """Backends should default to [cuda, rocm, openvino] when not specified."""
        minimal = tmp_path / "manifest.yaml"
        minimal.write_text(
            "model:\n"
            "  name: test\n"
            "  family: test\n"
            "  variants: [v1]\n"
            "  default_variant: v1\n"
            "source:\n"
            "  type: test\n"
            "  repo: test/test\n"
            "input:\n"
            "  type: image\n"
            "output:\n"
            "  type: detections\n"
            "ros:\n"
            "  publishers:\n"
            "    - topic: test\n"
            "      msg_type: Test\n"
        )
        manifest = load_manifest(minimal)
        assert manifest.model.backends == ["cuda", "rocm", "openvino"]
