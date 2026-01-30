"""Tests for the generator engine."""

import ast
import zipfile
from pathlib import Path

import pytest

from generator.engine import Generator, GenerationError


MANIFESTS_DIR = Path(__file__).parent.parent / "manifests"
CORE_TEMPLATES_DIR = Path(__file__).parent.parent / "generator" / "templates"
MODEL_TEMPLATES_DIR = Path(__file__).parent.parent / "model_templates"


@pytest.fixture
def gen():
    return Generator(MANIFESTS_DIR, CORE_TEMPLATES_DIR, MODEL_TEMPLATES_DIR)


class TestGeneratorInit:
    def test_loads_manifests(self, gen):
        assert len(gen.manifests) == 8
        assert "yolo" in gen.manifests

    def test_jinja_env_has_filters(self, gen):
        assert "python_repr" in gen.env.filters
        assert "yaml_repr" in gen.env.filters


class TestGenerateYolo:
    def test_generates_zip(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "cuda", "yolo_v8s", "my_detector", tmp_path)
        assert zip_path.exists()
        assert zip_path.suffix == ".zip"

    def test_zip_contains_expected_files(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "cuda", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()

        expected = [
            "my_detector/package.xml",
            "my_detector/setup.py",
            "my_detector/requirements.txt",
            "my_detector/Dockerfile",
            "my_detector/README.md",
            "my_detector/config/params.yaml",
            "my_detector/launch/vision.launch.py",
            "my_detector/my_detector/__init__.py",
            "my_detector/my_detector/node.py",
            "my_detector/my_detector/model.py",
            "my_detector/my_detector/runner.py",
        ]
        for f in expected:
            assert f in names, f"Missing: {f}"

    def test_generated_python_is_valid_syntax(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "cuda", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".py"):
                    content = zf.read(name).decode("utf-8")
                    try:
                        ast.parse(content)
                    except SyntaxError as e:
                        pytest.fail(f"Syntax error in {name}: {e}")

    def test_package_xml_is_valid_xml(self, gen, tmp_path):
        import xml.etree.ElementTree as ET
        zip_path = gen.generate("yolo", "cuda", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_detector/package.xml").decode("utf-8")
        # Should not raise
        ET.fromstring(content)

    def test_setup_py_contains_package_name(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "cuda", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_detector/setup.py").decode("utf-8")
        assert "my_detector" in content

    def test_requirements_has_ultralytics(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "cuda", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_detector/requirements.txt").decode("utf-8")
        assert "ultralytics" in content
        assert "cu124" in content  # CUDA index URL

    def test_node_has_detection_publisher(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "cuda", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_detector/my_detector/node.py").decode("utf-8")
        assert "Detection2DArray" in content

    def test_runner_has_cuda(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "cuda", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_detector/my_detector/runner.py").decode("utf-8")
        assert "CudaRunner" in content
        assert "torch.device" in content

    def test_model_has_yolo_code(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "cuda", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_detector/my_detector/model.py").decode("utf-8")
        assert "YOLO" in content
        assert "ultralytics" in content

    def test_default_variant(self, gen, tmp_path):
        """If variant is None, should use default from manifest."""
        zip_path = gen.generate("yolo", "cuda", None, "my_detector", tmp_path)
        assert zip_path.exists()

    def test_rocm_backend(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "rocm", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_detector/my_detector/runner.py").decode("utf-8")
        assert "RocmRunner" in content

    def test_openvino_backend(self, gen, tmp_path):
        zip_path = gen.generate("yolo", "openvino", "yolo_v8s", "my_detector", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_detector/my_detector/runner.py").decode("utf-8")
        assert "OpenVinoRunner" in content


class TestGenerateDepthAnything:
    def test_generates_zip(self, gen, tmp_path):
        zip_path = gen.generate(
            "depth_anything", "cuda", "depth_anything_v2_vitb", "my_depth", tmp_path)
        assert zip_path.exists()

    def test_generated_python_is_valid_syntax(self, gen, tmp_path):
        zip_path = gen.generate(
            "depth_anything", "cuda", "depth_anything_v2_vitb", "my_depth", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".py"):
                    content = zf.read(name).decode("utf-8")
                    try:
                        ast.parse(content)
                    except SyntaxError as e:
                        pytest.fail(f"Syntax error in {name}: {e}")

    def test_node_has_depth_publisher(self, gen, tmp_path):
        zip_path = gen.generate(
            "depth_anything", "cuda", "depth_anything_v2_vitb", "my_depth", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_depth/my_depth/node.py").decode("utf-8")
        assert "32FC1" in content

    def test_model_has_hf_code(self, gen, tmp_path):
        zip_path = gen.generate(
            "depth_anything", "cuda", "depth_anything_v2_vitb", "my_depth", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_depth/my_depth/model.py").decode("utf-8")
        assert "AutoModelForDepthEstimation" in content

    def test_runner_uses_hf_pattern(self, gen, tmp_path):
        zip_path = gen.generate(
            "depth_anything", "cuda", "depth_anything_v2_vitb", "my_depth", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_depth/my_depth/runner.py").decode("utf-8")
        assert "preprocess" in content
        assert "forward" in content
        assert "postprocess" in content

    def test_requirements_has_transformers(self, gen, tmp_path):
        zip_path = gen.generate(
            "depth_anything", "cuda", "depth_anything_v2_vitb", "my_depth", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_depth/requirements.txt").decode("utf-8")
        assert "transformers" in content
        assert "timm" in content


class TestGenerateGroundingDino:
    def test_generates_zip(self, gen, tmp_path):
        zip_path = gen.generate(
            "grounding_dino", "cuda", "grounding_dino_base", "my_dino", tmp_path)
        assert zip_path.exists()

    def test_generated_python_is_valid_syntax(self, gen, tmp_path):
        zip_path = gen.generate(
            "grounding_dino", "cuda", "grounding_dino_base", "my_dino", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".py"):
                    content = zf.read(name).decode("utf-8")
                    try:
                        ast.parse(content)
                    except SyntaxError as e:
                        pytest.fail(f"Syntax error in {name}: {e}")

    def test_node_has_text_subscriber(self, gen, tmp_path):
        zip_path = gen.generate(
            "grounding_dino", "cuda", "grounding_dino_base", "my_dino", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_dino/my_dino/node.py").decode("utf-8")
        assert "prompt" in content
        assert "String" in content

    def test_model_has_grounding_dino_code(self, gen, tmp_path):
        zip_path = gen.generate(
            "grounding_dino", "cuda", "grounding_dino_base", "my_dino", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_dino/my_dino/model.py").decode("utf-8")
        assert "AutoModelForZeroShotObjectDetection" in content

    def test_runner_passes_text(self, gen, tmp_path):
        zip_path = gen.generate(
            "grounding_dino", "cuda", "grounding_dino_base", "my_dino", tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("my_dino/my_dino/runner.py").decode("utf-8")
        assert "text" in content


class TestValidation:
    def test_unknown_model(self, gen, tmp_path):
        with pytest.raises(GenerationError, match="Unknown model"):
            gen.generate("nonexistent", "cuda", "v1", "pkg", tmp_path)

    def test_unknown_variant(self, gen, tmp_path):
        with pytest.raises(GenerationError, match="variant"):
            gen.generate("yolo", "cuda", "yolo_v99", "pkg", tmp_path)

    def test_unknown_backend(self, gen, tmp_path):
        with pytest.raises(GenerationError, match="backend"):
            gen.generate("yolo", "tpu", "yolo_v8s", "pkg", tmp_path)
