"""Tests for the CLI interface."""

from pathlib import Path

from click.testing import CliRunner

from generator.cli import main


@classmethod
def runner():
    return CliRunner()


class TestListModels:
    def test_lists_models(self):
        result = CliRunner().invoke(main, ["list-models"])
        assert result.exit_code == 0
        assert "yolo" in result.output
        assert "depth_anything" in result.output
        assert "grounding_dino" in result.output


class TestInfo:
    def test_yolo_info(self):
        result = CliRunner().invoke(main, ["info", "yolo"])
        assert result.exit_code == 0
        assert "yolo" in result.output
        assert "ultralytics" in result.output
        assert "yolo_v8s" in result.output

    def test_unknown_model(self):
        result = CliRunner().invoke(main, ["info", "nonexistent"])
        assert result.exit_code != 0


class TestGenerate:
    def test_generate_yolo(self, tmp_path):
        result = CliRunner().invoke(main, [
            "generate",
            "--model", "yolo",
            "--backend", "cuda",
            "--variant", "yolo_v8s",
            "--package-name", "test_pkg",
            "--output", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert (tmp_path / "test_pkg.zip").exists()

    def test_generate_default_variant(self, tmp_path):
        result = CliRunner().invoke(main, [
            "generate",
            "--model", "yolo",
            "--backend", "cuda",
            "--package-name", "test_pkg",
            "--output", str(tmp_path),
        ])
        assert result.exit_code == 0

    def test_generate_bad_model(self, tmp_path):
        result = CliRunner().invoke(main, [
            "generate",
            "--model", "nonexistent",
            "--backend", "cuda",
            "--package-name", "test_pkg",
            "--output", str(tmp_path),
        ])
        assert result.exit_code != 0

    def test_generate_bad_variant(self, tmp_path):
        result = CliRunner().invoke(main, [
            "generate",
            "--model", "yolo",
            "--backend", "cuda",
            "--variant", "yolo_v99",
            "--package-name", "test_pkg",
            "--output", str(tmp_path),
        ])
        assert result.exit_code != 0
