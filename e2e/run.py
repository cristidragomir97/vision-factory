#!/usr/bin/env python3
"""E2E test runner for vision-factory.

Generates a ROS2 vision package, runs everything (vision node, camera,
viz, rosboard) inside a single Docker container, and produces a markdown report.

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
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Allow importing generator and e2e config
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from generator.engine import Generator, GenerationError

from config import (
    BASE_IMAGE_TAG,
    CUDA_VERSION,
    ROS_DISTRO,
)

_ROOT = Path(__file__).parent.parent
MODELS_DIR = _ROOT / "models"
CORE_TEMPLATES_DIR = _ROOT / "generator" / "templates"
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
    """Run a command, always streaming output and printing the full command."""
    timeout = kwargs.pop("timeout", None)

    # Always print the full command being executed
    cmd_str = " ".join(cmd)
    print(f"\n>>> {cmd_str}", flush=True)

    # Always stream output in real-time
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, **kwargs,
    )
    lines: list[str] = []
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"\n!!! TIMEOUT after {timeout}ms — killing process", flush=True)
        proc.kill()
        proc.wait()
    output = "".join(lines)

    print(f"<<< exit {proc.returncode}", flush=True)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=output, stderr="")


def _check_prerequisites(require_gpu: bool) -> None:
    """Fail fast if docker or nvidia-smi are not available."""
    print("\n=== Checking prerequisites ===")

    docker_path = shutil.which("docker")
    if not docker_path:
        print("ERROR: docker not found in PATH", file=sys.stderr)
        sys.exit(1)
    print(f"docker: {docker_path}")

    # Print docker version
    _run_cmd(["docker", "version", "--format", "{{.Server.Version}}"])

    if require_gpu:
        print("\nGPU required — checking nvidia-smi...")
        result = _run_cmd(["nvidia-smi"])
        if result.returncode != 0:
            print("ERROR: nvidia-smi failed — NVIDIA GPU required", file=sys.stderr)
            sys.exit(1)
    else:
        print("GPU not required for this backend, skipping nvidia-smi check.")


# ---------------------------------------------------------------------------
# Base image
# ---------------------------------------------------------------------------

def _ensure_base_image() -> None:
    """Build the base image if it doesn't already exist."""
    print(f"\n=== Checking base image: {BASE_IMAGE_TAG} ===")
    print(f"    CUDA_VERSION={CUDA_VERSION}  ROS_DISTRO={ROS_DISTRO}")

    result = subprocess.run(
        ["docker", "image", "inspect", BASE_IMAGE_TAG],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"Base image {BASE_IMAGE_TAG} already exists — skipping build.")
        # Print image ID and size for confirmation
        _run_cmd([
            "docker", "image", "ls", "--format",
            "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}",
            BASE_IMAGE_TAG,
        ])
        return

    print(f"Base image {BASE_IMAGE_TAG} not found — building from {E2E_DIR / 'Dockerfile.base'}...")
    cmd = [
        "docker", "build",
        "-f", str(E2E_DIR / "Dockerfile.base"),
        "--build-arg", f"CUDA_VERSION={CUDA_VERSION}",
        "--build-arg", f"ROS_DISTRO={ROS_DISTRO}",
        "-t", BASE_IMAGE_TAG,
        str(E2E_DIR),
    ]
    result = _run_cmd(cmd, timeout=1200)
    if result.returncode != 0:
        print("ERROR: Base image build failed", file=sys.stderr)
        sys.exit(1)
    print(f"Base image {BASE_IMAGE_TAG} built successfully.")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_generate(
    model: str, backend: str, variant: str | None, package_name: str, work_dir: Path,
) -> tuple[StageResult, Path | None]:
    """Generate the ROS2 package zip and extract it."""
    print(f"\n=== Stage: Package generation ===")
    print(f"    model={model}  backend={backend}  variant={variant}")
    print(f"    package_name={package_name}")
    print(f"    work_dir={work_dir}")
    print(f"    MODELS_DIR={MODELS_DIR}")
    print(f"    CORE_TEMPLATES_DIR={CORE_TEMPLATES_DIR}")

    stage = StageResult("Package generation")
    t0 = time.monotonic()

    try:
        gen = Generator(MODELS_DIR, CORE_TEMPLATES_DIR)
        print(f"    Available models: {list(gen.manifests.keys())}")
        zip_path = gen.generate(
            model_name=model,
            backend=backend,
            variant=variant,
            package_name=package_name,
            output_dir=work_dir,
        )
        print(f"    Generated zip: {zip_path}")

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            print(f"    Zip contents ({len(names)} files):")
            for n in names:
                print(f"      {n}")
            zf.extractall(work_dir)

        pkg_dir = work_dir / package_name
        print(f"    Extracted to: {pkg_dir}")

        stage.pass_(time.monotonic() - t0)
        return stage, pkg_dir
    except GenerationError as e:
        print(f"    GENERATION ERROR: {e}")
        stage.fail(time.monotonic() - t0, "; ".join(e.errors))
        return stage, None


def stage_docker_run(
    work_dir: Path,
    package_name: str,
    node_name: str,
    output_topic: str,
    input_source: str,
    rosbag_path: Path | None,
    ros_distro: str,
    duration: float,
    require_gpu: bool,
    enable_rosboard: bool,
    video_device: str,
) -> tuple[StageResult, dict | None]:
    """Run everything in a single container: vision node + camera + viz + rosboard."""
    print(f"\n=== Stage: Build & Inference ===")
    print(f"    package_name={package_name}")
    print(f"    node_name={node_name}")
    print(f"    output_topic={output_topic}")
    print(f"    input_source={input_source}")
    print(f"    rosbag_path={rosbag_path}")
    print(f"    ros_distro={ros_distro}")
    print(f"    duration={duration}s")
    print(f"    require_gpu={require_gpu}")
    print(f"    enable_rosboard={enable_rosboard}")
    print(f"    video_device={video_device}")
    print(f"    image={BASE_IMAGE_TAG}")

    stage = StageResult("Build & Inference")
    t0 = time.monotonic()

    # work_dir is the package directory (e.g. build_context/e2e_yolo/)
    # Place support files in the parent (build_context/) to keep them separate
    context_dir = work_dir.parent
    shutil.copy2(E2E_DIR / "test_harness.py", context_dir / "test_harness.py")
    shutil.copy2(E2E_DIR / "entrypoint.sh", context_dir / "entrypoint.sh")
    print(f"    Copied test_harness.py -> {context_dir / 'test_harness.py'}")
    print(f"    Copied entrypoint.sh   -> {context_dir / 'entrypoint.sh'}")

    # Copy viz_node.py if it exists
    viz_node_src = E2E_DIR / "viz_node.py"
    if viz_node_src.exists():
        shutil.copy2(viz_node_src, context_dir / "viz_node.py")
        print(f"    Copied viz_node.py     -> {context_dir / 'viz_node.py'}")

    pkg_path = work_dir.resolve()
    harness_path = (context_dir / "test_harness.py").resolve()
    entrypoint_path = (context_dir / "entrypoint.sh").resolve()

    cmd = ["docker", "run", "--rm"]
    if require_gpu:
        cmd += ["--gpus", "all"]

    # Mount the package, harness, and entrypoint into the container
    cmd += [
        "-v", f"{pkg_path}:/ros_ws/src/{package_name}:ro",
        "-v", f"{harness_path}:/ros_ws/test_harness.py:ro",
        "-v", f"{entrypoint_path}:/entrypoint.sh:ro",
    ]

    # Mount viz_node.py into the container
    viz_node_path = (context_dir / "viz_node.py").resolve()
    if viz_node_path.exists():
        cmd += ["-v", f"{viz_node_path}:/ros_ws/viz_node.py:ro"]

    # Mount rosbag if applicable
    if input_source == "rosbag" and rosbag_path:
        cmd += ["-v", f"{rosbag_path.resolve()}:/test_data/bag:ro"]

    # Mount USB camera device if applicable
    if input_source == "usb_cam":
        cmd += ["--device", f"{video_device}:{video_device}"]

    # Expose rosboard web UI
    if enable_rosboard:
        cmd += ["-p", "8888:8888"]

    # Pass config as env vars for the entrypoint
    cmd += [
        "-e", f"ROS_DISTRO={ros_distro}",
        "-e", f"PACKAGE_NAME={package_name}",
        "-e", f"NODE_NAME={node_name}",
    ]

    cmd.append(BASE_IMAGE_TAG)

    # Harness arguments passed after the image
    harness_args = [
        "/entrypoint.sh",
        "--package", package_name,
        "--node", node_name,
        "--input-source", input_source,
        "--output-topic", output_topic,
        "--duration", str(duration),
    ]
    if input_source == "rosbag":
        harness_args += ["--bag", "/test_data/bag"]
    if input_source == "usb_cam":
        harness_args += ["--video-device", video_device]
    if enable_rosboard:
        harness_args.append("--rosboard")

    cmd += harness_args

    # Print volume mounts summary
    print("\n    Volume mounts:")
    print(f"      {pkg_path} -> /ros_ws/src/{package_name}:ro")
    print(f"      {harness_path} -> /ros_ws/test_harness.py:ro")
    print(f"      {entrypoint_path} -> /entrypoint.sh:ro")
    if viz_node_path.exists():
        print(f"      {viz_node_path} -> /ros_ws/viz_node.py:ro")
    if input_source == "rosbag" and rosbag_path:
        print(f"      {rosbag_path.resolve()} -> /test_data/bag:ro")
    if input_source == "usb_cam":
        print(f"      device: {video_device}")
    if enable_rosboard:
        print(f"      rosboard: http://localhost:8888")

    result = _run_cmd(cmd, timeout=600)
    full_output = result.stdout + "\n" + result.stderr

    # Parse the JSON report from the harness
    report_data = None
    if "---E2E_REPORT_START---" in result.stdout:
        try:
            json_str = result.stdout.split("---E2E_REPORT_START---")[1].split("---E2E_REPORT_END---")[0]
            report_data = json.loads(json_str.strip())
            print(f"\n    Parsed harness report: {json.dumps(report_data, indent=2)}")
        except (IndexError, json.JSONDecodeError) as e:
            print(f"\n    Failed to parse harness JSON: {e}")
            stage.fail(
                time.monotonic() - t0,
                f"Failed to parse harness output: {e}",
                log=full_output,
            )
            return stage, None
    else:
        print("\n    WARNING: No ---E2E_REPORT_START--- marker found in output")

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
    enable_rosboard: bool = False,
) -> str:
    """Generate a markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    overall = "PASS" if all(s.status == "PASS" for s in stages) else "FAIL"

    lines = [
        f"# E2E Test Report",
        "",
        f"**Model:** {model} | **Backend:** {backend} | **Variant:** {variant}",
        f"**ROS Distro:** {ros_distro} | **Input:** {input_source}"
        + (f" | **rosboard:** yes" if enable_rosboard else ""),
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
            f"- **Input active:** {harness_data.get('input_active', False)}",
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
    parser.add_argument("--backend", required=True, help="Backend (cuda, rocm, openvino, onnx, tensorrt)")
    parser.add_argument("--variant", default=None, help="Model variant (uses default if omitted)")
    parser.add_argument("--package-name", default=None, help="Package name (default: e2e_<model>)")
    parser.add_argument("--ros-distro", default=ROS_DISTRO, help=f"ROS2 distro (default: {ROS_DISTRO})")
    parser.add_argument("--duration", type=float, default=30.0, help="Inference duration in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("./e2e-output"),
                        help="Directory for report output")

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--rosbag", type=Path, help="Path to rosbag directory")
    input_group.add_argument("--usb-cam", action="store_true",
                             help="Use usb_cam package as input (live webcam)")

    # Camera device (only relevant with --usb-cam)
    parser.add_argument("--video-device", default="/dev/video0",
                        help="Video device path for usb_cam (default: /dev/video0)")

    # Visualization
    parser.add_argument("--rosboard", action="store_true",
                        help="Start rosboard web UI on port 8888")

    args = parser.parse_args()

    package_name = args.package_name or f"e2e_{args.model}"
    require_gpu = args.backend in ("cuda", "rocm", "tensorrt")
    input_source = "usb_cam" if args.usb_cam else "rosbag"

    print("=" * 60)
    print("  vision-factory E2E runner")
    print("=" * 60)
    print(f"  model:        {args.model}")
    print(f"  backend:      {args.backend}")
    print(f"  variant:      {args.variant or '(default)'}")
    print(f"  package_name: {package_name}")
    print(f"  ros_distro:   {args.ros_distro}")
    print(f"  input_source: {input_source}")
    print(f"  rosbag:       {args.rosbag or 'N/A'}")
    print(f"  usb_cam:      {args.usb_cam}")
    print(f"  video_device: {args.video_device}")
    print(f"  rosboard:     {args.rosboard}")
    print(f"  duration:     {args.duration}s")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  base_image:   {BASE_IMAGE_TAG}")
    print(f"  cuda_version: {CUDA_VERSION}")
    print(f"  require_gpu:  {require_gpu}")
    print("=" * 60)

    # Validate rosbag path
    if args.rosbag and not args.rosbag.exists():
        print(f"ERROR: Rosbag path does not exist: {args.rosbag}", file=sys.stderr)
        sys.exit(1)

    # Prerequisites
    _check_prerequisites(require_gpu)

    # Determine output topic from manifest
    gen = Generator(MODELS_DIR, CORE_TEMPLATES_DIR)
    if args.model not in gen.manifests:
        print(f"ERROR: Unknown model '{args.model}'", file=sys.stderr)
        print(f"  Available models: {list(gen.manifests.keys())}")
        sys.exit(1)
    manifest = gen.manifests[args.model]
    output_topic = f"/{manifest.ros.publishers[0].topic}" if manifest.ros.publishers else "/output"
    print(f"\nResolved output_topic: {output_topic}")

    variant = args.variant or manifest.model.default_variant
    node_name = f"{package_name}_node"
    print(f"Resolved variant: {variant}")
    print(f"Resolved node_name: {node_name}")

    stages: list[StageResult] = []
    harness_data = None

    # Create a timestamped run directory so each run is preserved
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{run_stamp}_{args.model}_{args.backend}"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir / "build_context"
    work_dir.mkdir()
    print(f"\nRun directory: {run_dir}")
    print(f"Work directory: {work_dir}")

    # Ensure base image exists (build if needed)
    _ensure_base_image()

    # Stage 1: Generate
    print(f"\n[1/2] Generating package: {package_name} ({args.model}/{variant}/{args.backend})")
    s, pkg_dir = stage_generate(args.model, args.backend, variant, package_name, work_dir)
    stages.append(s)
    print(f"      {s.status} ({s.duration:.1f}s)")
    if s.status != "PASS":
        _finish(stages, harness_data, args, variant, input_source, run_dir)
        return

    # Save a copy of the generated package for inspection
    pkg_copy = run_dir / "package"
    shutil.copytree(work_dir / package_name, pkg_copy)
    print(f"      Package saved to: {pkg_copy}")

    # Stage 2: Build package + run inference (everything in one container)
    mode_desc = "live webcam" if args.usb_cam else "rosbag"
    if args.rosboard:
        mode_desc += " + rosboard"
    print(f"\n[2/2] Building & running ({mode_desc}, {args.duration}s)...")
    s, harness_data = stage_docker_run(
        work_dir=work_dir / package_name,
        package_name=package_name,
        node_name=node_name,
        output_topic=output_topic,
        input_source=input_source,
        rosbag_path=args.rosbag,
        ros_distro=args.ros_distro,
        duration=args.duration,
        require_gpu=require_gpu,
        enable_rosboard=args.rosboard,
        video_device=args.video_device,
    )
    stages.append(s)
    print(f"      {s.status} ({s.duration:.1f}s)")
    if harness_data:
        print(f"      Messages received: {harness_data.get('messages_received', 0)}")

    _finish(stages, harness_data, args, variant, input_source, run_dir)


def _finish(stages, harness_data, args, variant, input_source, run_dir: Path):
    """Write the report and exit."""
    print(f"\n=== Generating report ===")
    report = generate_report(
        stages, harness_data,
        model=args.model,
        backend=args.backend,
        variant=variant,
        ros_distro=args.ros_distro,
        input_source=input_source,
        enable_rosboard=args.rosboard,
    )

    # Write report into the run directory
    report_path = run_dir / "e2e-report.md"
    report_path.write_text(report)
    print(f"Report written to: {report_path}")

    # Also write a copy to the top-level output dir for quick access
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "e2e-report.md"
    latest_path.write_text(report)
    print(f"Latest report: {latest_path}")

    overall = all(s.status == "PASS" for s in stages)
    print(f"\n{'=' * 60}")
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    for s in stages:
        dur = f"{s.duration:.1f}s" if s.duration is not None else "-"
        err = f" — {s.error}" if s.error else ""
        print(f"    {s.name}: {s.status} ({dur}){err}")
    print(f"{'=' * 60}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
