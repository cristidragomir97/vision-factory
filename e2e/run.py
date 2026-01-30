#!/usr/bin/env python3
"""E2E test runner for vision-factory.

Generates a ROS2 vision package, builds it in Docker, runs full inference
with a user-supplied rosbag, and produces a markdown report.

Usage:
    python e2e/run.py \
        --model yolo \
        --backend cuda \
        --rosbag /path/to/camera_bag \
        --ros-distro jazzy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Allow importing generator when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.engine import Generator, GenerationError

_ROOT = Path(__file__).parent.parent
MANIFESTS_DIR = _ROOT / "manifests"
CORE_TEMPLATES_DIR = _ROOT / "generator" / "templates"
MODEL_TEMPLATES_DIR = _ROOT / "model_templates"
E2E_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StageResult:
    def __init__(self, name: str):
        self.name = name
        self.status = "SKIP"
        self.duration: float | None = None
        self.error: str | None = None
        self.log: str = ""

    def pass_(self, duration: float, log: str = ""):
        self.status = "PASS"
        self.duration = duration
        self.log = log

    def fail(self, duration: float, error: str, log: str = ""):
        self.status = "FAIL"
        self.duration = duration
        self.error = error
        self.log = log


def _run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _check_prerequisites(require_gpu: bool):
    """Fail fast if docker or nvidia-smi are not available."""
    if not shutil.which("docker"):
        print("ERROR: docker not found in PATH", file=sys.stderr)
        sys.exit(1)

    if require_gpu:
        result = _run_cmd(["nvidia-smi"])
        if result.returncode != 0:
            print("ERROR: nvidia-smi failed — NVIDIA GPU required", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_generate(
    model: str, backend: str, variant: str | None, package_name: str, work_dir: Path,
) -> tuple[StageResult, Path | None]:
    """Generate the ROS2 package zip and extract it."""
    stage = StageResult("Package generation")
    t0 = time.monotonic()

    try:
        gen = Generator(MANIFESTS_DIR, CORE_TEMPLATES_DIR, MODEL_TEMPLATES_DIR)
        zip_path = gen.generate(
            model_name=model,
            backend=backend,
            variant=variant,
            package_name=package_name,
            output_dir=work_dir,
        )
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(work_dir)

        stage.pass_(time.monotonic() - t0)
        return stage, work_dir / package_name
    except GenerationError as e:
        stage.fail(time.monotonic() - t0, "; ".join(e.errors))
        return stage, None


def stage_docker_build(
    work_dir: Path,
    package_name: str,
    node_name: str,
    output_topic: str,
    rosbag_path: Path | None,
    ros_distro: str,
    duration: float,
    input_source: str,
    enable_rqt: bool,
) -> tuple[StageResult, str | None]:
    """Render the e2e Dockerfile and build the image."""
    stage = StageResult("Docker build")
    t0 = time.monotonic()

    # Copy test harness into build context
    shutil.copy2(E2E_DIR / "test_harness.py", work_dir / "test_harness.py")

    # Copy rosbag into build context (only for rosbag mode)
    if input_source == "rosbag" and rosbag_path:
        bag_dest = work_dir / "test_bag"
        if rosbag_path.is_dir():
            shutil.copytree(rosbag_path, bag_dest)
        else:
            bag_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rosbag_path, bag_dest)

    # Render Dockerfile from template
    import jinja2
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(E2E_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("Dockerfile.e2e.j2")
    dockerfile_content = template.render(
        package_name=package_name,
        node_name=node_name,
        output_topic=output_topic,
        ros_distro=ros_distro,
        duration=str(duration),
        input_source=input_source,
        enable_rqt=enable_rqt,
    )
    (work_dir / "Dockerfile").write_text(dockerfile_content)

    image_tag = f"e2e-{package_name}:{ros_distro}"
    result = _run_cmd(
        ["docker", "build", "-t", image_tag, "."],
        cwd=work_dir,
        timeout=600,
    )

    if result.returncode != 0:
        stage.fail(
            time.monotonic() - t0,
            f"Docker build failed (exit {result.returncode})",
            log=result.stdout + "\n" + result.stderr,
        )
        return stage, None

    stage.pass_(time.monotonic() - t0, log=result.stdout)
    return stage, image_tag


def stage_docker_run(
    image_tag: str,
    require_gpu: bool,
    usb_cam: bool = False,
    enable_rqt: bool = False,
) -> tuple[StageResult, dict | None]:
    """Run the e2e container and parse the test harness output."""
    stage = StageResult("Inference")
    t0 = time.monotonic()

    cmd = ["docker", "run", "--rm"]
    if require_gpu:
        cmd += ["--gpus", "all"]
    if usb_cam:
        # Pass through video devices for webcam access
        cmd += ["--device", "/dev/video0:/dev/video0"]
    if enable_rqt:
        # X11 forwarding for GUI
        cmd += [
            "-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
        ]
    cmd.append(image_tag)

    result = _run_cmd(cmd, timeout=300)
    full_output = result.stdout + "\n" + result.stderr

    # Parse the JSON report from the harness
    report_data = None
    if "---E2E_REPORT_START---" in result.stdout:
        try:
            json_str = result.stdout.split("---E2E_REPORT_START---")[1].split("---E2E_REPORT_END---")[0]
            report_data = json.loads(json_str.strip())
        except (IndexError, json.JSONDecodeError) as e:
            stage.fail(
                time.monotonic() - t0,
                f"Failed to parse harness output: {e}",
                log=full_output,
            )
            return stage, None

    if result.returncode != 0 and report_data is None:
        stage.fail(
            time.monotonic() - t0,
            f"Container exited with code {result.returncode}",
            log=full_output,
        )
        return stage, None

    if report_data and report_data.get("errors"):
        stage.fail(
            time.monotonic() - t0,
            "; ".join(report_data["errors"]),
            log=full_output,
        )
        return stage, report_data

    stage.pass_(time.monotonic() - t0, log=full_output)
    return stage, report_data


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(
    stages: list[StageResult],
    harness_data: dict | None,
    model: str,
    backend: str,
    variant: str,
    ros_distro: str,
    input_source: str = "rosbag",
    enable_rqt: bool = False,
) -> str:
    """Generate a markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    overall = "PASS" if all(s.status == "PASS" for s in stages) else "FAIL"

    lines = [
        f"# E2E Test Report",
        "",
        f"**Model:** {model} | **Backend:** {backend} | **Variant:** {variant}",
        f"**ROS Distro:** {ros_distro} | **Input:** {input_source}"
        + (f" | **rqt:** yes" if enable_rqt else ""),
        f"**Date:** {now} | **Result:** {overall}",
        "",
        "## Stages",
        "",
        "| Stage | Status | Duration |",
        "|-------|--------|----------|",
    ]

    for s in stages:
        dur = f"{s.duration:.1f}s" if s.duration is not None else "-"
        lines.append(f"| {s.name} | {s.status} | {dur} |")

    lines.append("")

    if harness_data:
        lines += [
            "## Inference Summary",
            "",
            f"- **Node started:** {harness_data.get('node_started', False)}",
            f"- **Node startup time:** {harness_data.get('node_startup_time_s', '-')}s",
            f"- **Bag played:** {harness_data.get('bag_played', False)}",
            f"- **Messages received:** {harness_data.get('messages_received', 0)}",
            f"- **Output topic:** {harness_data.get('output_topic', '-')}",
            "",
        ]

    # Errors section
    errors = []
    for s in stages:
        if s.error:
            errors.append(f"- **{s.name}:** {s.error}")
    if harness_data and harness_data.get("errors"):
        for e in harness_data["errors"]:
            errors.append(f"- **Harness:** {e}")

    if errors:
        lines += ["## Errors", ""] + errors + [""]
    else:
        lines += ["## Errors", "", "None", ""]

    # Build logs in collapsible sections
    for s in stages:
        if s.log:
            lines += [
                f"<details><summary>{s.name} log</summary>",
                "",
                "```",
                s.log[-5000:],  # Truncate to last 5000 chars
                "```",
                "",
                "</details>",
                "",
            ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="E2E test runner for vision-factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model architecture (e.g. yolo)")
    parser.add_argument("--backend", required=True, help="Backend (cuda, rocm, openvino)")
    parser.add_argument("--variant", default=None, help="Model variant (uses default if omitted)")
    parser.add_argument("--package-name", default=None, help="Package name (default: e2e_<model>)")
    parser.add_argument("--ros-distro", default="jazzy", help="ROS2 distro (default: jazzy)")
    parser.add_argument("--duration", type=float, default=30.0, help="Inference duration in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("./e2e-output"),
                        help="Directory for report output")

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--rosbag", type=Path, help="Path to rosbag directory")
    input_group.add_argument("--usb-cam", action="store_true",
                             help="Use usb_cam package as input (live webcam)")

    # Visualization
    parser.add_argument("--rqt", action="store_true",
                        help="Open rqt_image_view on the output topic (requires X11)")

    args = parser.parse_args()

    package_name = args.package_name or f"e2e_{args.model}"
    require_gpu = args.backend in ("cuda", "rocm")
    input_source = "usb_cam" if args.usb_cam else "rosbag"

    # Validate rosbag path
    if args.rosbag and not args.rosbag.exists():
        print(f"ERROR: Rosbag path does not exist: {args.rosbag}", file=sys.stderr)
        sys.exit(1)

    # Prerequisites
    _check_prerequisites(require_gpu)

    # Determine output topic from manifest
    gen = Generator(MANIFESTS_DIR, CORE_TEMPLATES_DIR, MODEL_TEMPLATES_DIR)
    if args.model not in gen.manifests:
        print(f"ERROR: Unknown model '{args.model}'", file=sys.stderr)
        sys.exit(1)
    manifest = gen.manifests[args.model]
    output_topic = f"/{manifest.ros.publishers[0].topic}" if manifest.ros.publishers else "/output"

    variant = args.variant or manifest.model.default_variant
    node_name = f"{package_name}_node"

    stages: list[StageResult] = []
    harness_data = None

    with tempfile.TemporaryDirectory(prefix="e2e_") as tmp:
        work_dir = Path(tmp)

        # Stage 1: Generate
        print(f"[1/3] Generating package: {package_name} ({args.model}/{variant}/{args.backend})")
        s, pkg_dir = stage_generate(args.model, args.backend, variant, package_name, work_dir)
        stages.append(s)
        print(f"      {s.status} ({s.duration:.1f}s)")
        if s.status != "PASS":
            _finish(stages, harness_data, args, variant, input_source)
            return

        # Stage 2: Docker build
        print(f"[2/3] Building Docker image (ros:{args.ros_distro}, input:{input_source})...")
        s, image_tag = stage_docker_build(
            work_dir, package_name, node_name, output_topic,
            args.rosbag, args.ros_distro, args.duration,
            input_source=input_source, enable_rqt=args.rqt,
        )
        stages.append(s)
        print(f"      {s.status} ({s.duration:.1f}s)")
        if s.status != "PASS":
            _finish(stages, harness_data, args, variant, input_source)
            return

        # Stage 3: Run inference
        mode_desc = "live webcam" if args.usb_cam else "rosbag"
        if args.rqt:
            mode_desc += " + rqt"
        print(f"[3/3] Running inference ({mode_desc}, {args.duration}s)...")
        s, harness_data = stage_docker_run(
            image_tag, require_gpu,
            usb_cam=args.usb_cam, enable_rqt=args.rqt,
        )
        stages.append(s)
        print(f"      {s.status} ({s.duration:.1f}s)")
        if harness_data:
            print(f"      Messages received: {harness_data.get('messages_received', 0)}")

    _finish(stages, harness_data, args, variant, input_source)


def _finish(stages, harness_data, args, variant, input_source):
    """Write the report and exit."""
    report = generate_report(
        stages, harness_data,
        model=args.model,
        backend=args.backend,
        variant=variant,
        ros_distro=args.ros_distro,
        input_source=input_source,
        enable_rqt=args.rqt,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "e2e-report.md"
    report_path.write_text(report)
    print(f"\nReport: {report_path}")

    overall = all(s.status == "PASS" for s in stages)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
