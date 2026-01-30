"""Generator engine: manifest → resolve → render → zip."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import jinja2

from .manifest import ModelManifest, load_all_manifests
from .resolver import resolve_dependencies, validate_selection
from .context import build_context


class GenerationError(Exception):
    """Raised when generation fails validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _python_repr(value: Any) -> str:
    """Jinja2 filter: render a value as a Python literal."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return repr(value)
    return str(value)


def _yaml_repr(value: Any) -> str:
    """Jinja2 filter: render a value as a YAML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[" + ", ".join(_yaml_repr(v) for v in value) + "]"
    return str(value)


class Generator:
    """Core generator: loads manifests, renders templates, produces zip."""

    def __init__(
        self,
        manifests_dir: Path,
        core_templates_dir: Path,
        model_templates_dir: Path | None = None,
    ):
        self.manifests = load_all_manifests(manifests_dir)
        self.model_templates_dir = model_templates_dir

        # Set up Jinja2 with multiple template directories
        loaders = [jinja2.FileSystemLoader(str(core_templates_dir))]
        if model_templates_dir and model_templates_dir.exists():
            loaders.append(jinja2.FileSystemLoader(str(model_templates_dir)))

        self.env = jinja2.Environment(
            loader=jinja2.ChoiceLoader(loaders),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.env.filters["python_repr"] = _python_repr
        self.env.filters["yaml_repr"] = _yaml_repr

    def generate(
        self,
        model_name: str,
        backend: str,
        variant: str | None,
        package_name: str,
        output_dir: Path,
    ) -> Path:
        """Generate a ROS2 package zip.

        Args:
            model_name: Model name (e.g. "yolo")
            backend: Backend name (e.g. "cuda")
            variant: Model variant (uses default if None)
            package_name: Name for the generated ROS package
            output_dir: Directory to write the zip file

        Returns:
            Path to the generated zip file
        """
        # Look up manifest
        if model_name not in self.manifests:
            raise GenerationError([
                f"Unknown model '{model_name}'. "
                f"Available: {list(self.manifests.keys())}"
            ])

        manifest = self.manifests[model_name]

        # Default variant
        if variant is None:
            variant = manifest.model.default_variant

        # Validate
        errors = validate_selection(
            model_name, backend, variant, manifest.model.variants
        )
        if errors:
            raise GenerationError(errors)

        # Resolve deps
        resolved = resolve_dependencies(model_name, backend, manifest.output.type)

        # Build context
        ctx = build_context(
            model_name, backend, variant, package_name, manifest, resolved
        )

        # Render all templates
        files = self._render_all(ctx)

        # Pack into zip
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._pack_zip(files, package_name, output_dir)

    def _render_all(self, ctx: dict) -> dict[str, str]:
        """Render all templates, returning {relative_path: content}."""
        pkg = ctx["package_name"]
        model = ctx["model_name"]
        backend = ctx["backend"]
        files: dict[str, str] = {}

        # Base templates
        base_templates = {
            "package.xml": "base/package.xml.j2",
            "setup.py": "base/setup.py.j2",
            "requirements.txt": "base/requirements.txt.j2",
            "Dockerfile": "base/Dockerfile.j2",
            "README.md": "base/README.md.j2",
            "config/params.yaml": "base/params.yaml.j2",
            "launch/vision.launch.py": "base/vision.launch.py.j2",
            f"{pkg}/__init__.py": "base/__init__.py.j2",
            f"{pkg}/node.py": "base/node.py.j2",
        }

        for out_path, template_path in base_templates.items():
            files[out_path] = self._render(template_path, ctx)

        # Ament resource marker (required by colcon build)
        files[f"resource/{pkg}"] = ""

        # Runner template
        files[f"{pkg}/runner.py"] = self._render(f"runners/{backend}.py.j2", ctx)

        # Model template (from model_templates_dir or fallback)
        model_template = f"models/{model}.py.j2"
        try:
            files[f"{pkg}/model.py"] = self._render(model_template, ctx)
        except jinja2.TemplateNotFound:
            # No model template available — generate a placeholder
            files[f"{pkg}/model.py"] = self._render_model_placeholder(ctx)

        return files

    def _render(self, template_path: str, ctx: dict) -> str:
        """Render a single template."""
        template = self.env.get_template(template_path)
        return template.render(**ctx)

    def _render_model_placeholder(self, ctx: dict) -> str:
        """Generate a placeholder model.py when no template exists."""
        model_name = ctx["model_name"]
        variant = ctx["variant"]
        lines = [
            f'"""Model: {model_name} ({variant})"""',
            "",
            f"# No model template found for '{model_name}'.",
            f"# Add a template at models/{model_name}.py.j2",
            "",
            "raise NotImplementedError(",
            f'    "Model template for {model_name} not found. "',
            f'    "Please add it to the model templates directory."',
            ")",
            "",
        ]
        return "\n".join(lines)

    def _pack_zip(
        self,
        files: dict[str, str],
        package_name: str,
        output_dir: Path,
    ) -> Path:
        """Pack rendered files into a zip."""
        zip_path = output_dir / f"{package_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel_path, content in sorted(files.items()):
                zf.writestr(f"{package_name}/{rel_path}", content)
        return zip_path
