"""CLI interface for the vision package generator."""

from __future__ import annotations

from pathlib import Path

import click

from .engine import Generator, GenerationError

# Default paths relative to this package
_ROOT = Path(__file__).parent.parent
MODELS_DIR = _ROOT / "models"
CORE_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _get_generator(
    models_dir: Path | None = None,
) -> Generator:
    return Generator(
        models_dir=models_dir or MODELS_DIR,
        core_templates_dir=CORE_TEMPLATES_DIR,
    )


@click.group()
def main():
    """ROS2 Vision Package Generator.

    Generate self-contained ROS2 packages for vision models.
    """
    pass


@main.command()
@click.option("--model", required=True, help="Model architecture (e.g. yolo, depth_anything)")
@click.option("--backend", required=True, type=click.Choice(["cuda", "rocm", "openvino", "tensorrt", "onnx"]))
@click.option("--variant", default=None, help="Model variant (uses default if omitted)")
@click.option("--package-name", required=True, help="Name for the generated ROS2 package")
@click.option("--output", default="./output", help="Output directory for the zip file")
@click.option("--models", default=None, type=click.Path(exists=True, path_type=Path),
              help="Path to models directory")
def generate(model, backend, variant, package_name, output, models):
    """Generate a ROS2 vision package."""
    gen = _get_generator(models)

    try:
        zip_path = gen.generate(
            model_name=model,
            backend=backend,
            variant=variant,
            package_name=package_name,
            output_dir=Path(output),
        )
        click.echo(f"Generated: {zip_path}")
        click.echo(f"  Model:   {model} ({variant or 'default'})")
        click.echo(f"  Backend: {backend}")
        click.echo(f"  Package: {package_name}")
    except GenerationError as e:
        for err in e.errors:
            click.echo(f"Error: {err}", err=True)
        raise SystemExit(1)


@main.command("list-models")
@click.option("--models", default=None, type=click.Path(exists=True, path_type=Path))
def list_models(models):
    """List available model architectures."""
    gen = _get_generator(models)

    click.echo("Available models:\n")
    for name, manifest in sorted(gen.manifests.items()):
        output_type = manifest.output.type
        n_variants = len(manifest.model.variants)
        click.echo(f"  {name:25s} {output_type:20s} ({n_variants} variants)")


@main.command()
@click.argument("model_name")
@click.option("--models", default=None, type=click.Path(exists=True, path_type=Path))
def info(model_name, models):
    """Show details about a model architecture."""
    gen = _get_generator(models)

    if model_name not in gen.manifests:
        click.echo(f"Unknown model: {model_name}", err=True)
        click.echo(f"Available: {', '.join(sorted(gen.manifests.keys()))}", err=True)
        raise SystemExit(1)

    m = gen.manifests[model_name]
    click.echo(f"Model:    {m.model.name}")
    click.echo(f"Family:   {m.model.family}")
    click.echo(f"Source:   {m.source.type} ({m.source.repo})")
    click.echo(f"Output:   {m.output.type} ({m.output.format})")
    click.echo(f"Default:  {m.model.default_variant}")
    click.echo(f"Variants:")
    for v in m.model.variants:
        marker = " (default)" if v == m.model.default_variant else ""
        vram = m.resources.get("vram", {}).get(v, "?")
        click.echo(f"  - {v}{marker}  [{vram} MB VRAM]")
