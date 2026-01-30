"""Tests for dependency resolution and context building."""

import pytest

from generator.resolver import resolve_dependencies, validate_selection, BACKENDS
from generator.context import build_context


class TestResolveDependencies:
    def test_cuda_backend(self):
        deps = resolve_dependencies("yolo", "cuda", "detections")
        assert deps.torch_index_url == "https://download.pytorch.org/whl/cu124"
        assert any("torch>=" in d for d in deps.pip_deps)

    def test_rocm_backend(self):
        deps = resolve_dependencies("yolo", "rocm", "detections")
        assert deps.torch_index_url == "https://download.pytorch.org/whl/rocm6.2"

    def test_openvino_backend(self):
        deps = resolve_dependencies("yolo", "openvino", "detections")
        assert deps.torch_index_url is None
        assert "openvino>=2024.0" in deps.pip_deps

    def test_yolo_deps(self):
        deps = resolve_dependencies("yolo", "cuda", "detections")
        assert any("ultralytics" in d for d in deps.pip_deps)

    def test_depth_anything_deps(self):
        deps = resolve_dependencies("depth_anything", "cuda", "depth_map")
        assert any("transformers" in d for d in deps.pip_deps)
        assert any("timm" in d for d in deps.pip_deps)

    def test_grounding_dino_deps(self):
        deps = resolve_dependencies("grounding_dino", "cuda", "detections")
        assert any("transformers" in d for d in deps.pip_deps)

    def test_base_deps_always_present(self):
        deps = resolve_dependencies("yolo", "cuda")
        assert any("numpy" in d for d in deps.pip_deps)
        assert any("opencv" in d for d in deps.pip_deps)

    def test_ros_deps(self):
        deps = resolve_dependencies("yolo", "cuda", "detections")
        assert "rclpy" in deps.ros_deps
        assert "cv_bridge" in deps.ros_deps
        assert "vision_msgs" in deps.ros_deps

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_dependencies("yolo", "tpu")

    def test_sam_openvino_warning(self):
        deps = resolve_dependencies("segment_anything", "openvino", "segmentation")
        assert len(deps.warnings) > 0
        assert "SAM2" in deps.warnings[0]

    def test_unknown_model_no_extra_deps(self):
        deps = resolve_dependencies("unknown_model", "cuda")
        # Should still work with just base deps
        assert any("torch>=" in d for d in deps.pip_deps)


class TestValidateSelection:
    def test_valid_selection(self):
        errors = validate_selection("yolo", "cuda", "yolo_v8s", ["yolo_v8n", "yolo_v8s", "yolo_v8m"])
        assert errors == []

    def test_invalid_backend(self):
        errors = validate_selection("yolo", "tpu", "yolo_v8s", ["yolo_v8s"])
        assert len(errors) == 1
        assert "backend" in errors[0].lower()

    def test_invalid_variant(self):
        errors = validate_selection("yolo", "cuda", "yolo_v99", ["yolo_v8n", "yolo_v8s"])
        assert len(errors) == 1
        assert "variant" in errors[0].lower()

    def test_multiple_errors(self):
        errors = validate_selection("yolo", "tpu", "yolo_v99", ["yolo_v8s"])
        assert len(errors) == 2


class TestBuildContext:
    def test_basic_context(self, yolo_manifest):
        from generator.resolver import resolve_dependencies
        deps = resolve_dependencies("yolo", "cuda", "detections")
        ctx = build_context("yolo", "cuda", "yolo_v8s", "my_detector", yolo_manifest, deps)

        assert ctx["package_name"] == "my_detector"
        assert ctx["node_name"] == "my_detector_node"
        assert ctx["model_name"] == "yolo"
        assert ctx["model_family"] == "ultralytics"
        assert ctx["model_class"] == "YoloModel"
        assert ctx["runner_class"] == "CudaRunner"
        assert ctx["variant"] == "yolo_v8s"
        assert ctx["backend"] == "cuda"
        assert ctx["output_type"] == "detections"
        assert ctx["has_text_input"] is False
        assert ctx["has_prompt_input"] is False

    def test_dino_has_text_input(self, dino_manifest):
        from generator.resolver import resolve_dependencies
        deps = resolve_dependencies("grounding_dino", "cuda", "detections")
        ctx = build_context("grounding_dino", "cuda", "grounding_dino_base", "my_dino", dino_manifest, deps)
        assert ctx["has_text_input"] is True

    def test_depth_context(self, depth_manifest):
        from generator.resolver import resolve_dependencies
        deps = resolve_dependencies("depth_anything", "cuda", "depth_map")
        ctx = build_context("depth_anything", "cuda", "depth_anything_v2_vitb", "my_depth", depth_manifest, deps)
        assert ctx["output_type"] == "depth_map"
        assert ctx["model_family"] == "huggingface"

    def test_publishers_in_context(self, yolo_manifest):
        from generator.resolver import resolve_dependencies
        deps = resolve_dependencies("yolo", "cuda", "detections")
        ctx = build_context("yolo", "cuda", "yolo_v8s", "my_detector", yolo_manifest, deps)
        assert len(ctx["publishers"]) > 0
        assert ctx["publishers"][0]["topic"] == "detections"

    def test_parameters_in_context(self, yolo_manifest):
        from generator.resolver import resolve_dependencies
        deps = resolve_dependencies("yolo", "cuda", "detections")
        ctx = build_context("yolo", "cuda", "yolo_v8s", "my_detector", yolo_manifest, deps)
        param_names = [p["name"] for p in ctx["parameters"]]
        assert "confidence_threshold" in param_names

    def test_variant_id_strips_model_prefix(self, yolo_manifest):
        from generator.resolver import resolve_dependencies
        deps = resolve_dependencies("yolo", "cuda", "detections")
        ctx = build_context("yolo", "cuda", "yolo_v8s", "my_detector", yolo_manifest, deps)
        assert ctx["variant_id"] == "v8s"
