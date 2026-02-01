"""E2E test harness — runs inside the Docker container.

Launches the vision node, feeds input (rosbag or usb_cam), optionally
starts rosboard, captures output messages, and prints a JSON report to stdout.

Everything runs in a single container — camera, viz, rosboard, and the vision
node are all started as local processes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time


def _log(msg: str):
    """Print a timestamped log line to stderr (so it doesn't pollute the JSON report)."""
    ts = time.strftime("%H:%M:%S")
    print(f"[harness {ts}] {msg}", file=sys.stderr, flush=True)


def _run_show(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, log it, and return the result."""
    _log(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            _log(f"  stdout: {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            _log(f"  stderr: {line}")
    _log(f"  exit code: {result.returncode}")
    return result


def _popen_show(cmd: list[str], **kwargs) -> subprocess.Popen:
    """Start a subprocess and log the command."""
    _log(f"$ {' '.join(cmd)}  (background)")
    return subprocess.Popen(cmd, **kwargs)


def _stream_pipe(pipe, label: str):
    """Read lines from a pipe and log them. Run in a daemon thread."""
    try:
        for line in pipe:
            _log(f"[{label}] {line.rstrip()}")
    except (ValueError, OSError):
        pass  # pipe closed


def _wait_for_node(node_name: str, timeout: float = 60.0) -> float:
    """Wait until the node appears in `ros2 node list`. Returns startup time."""
    start = time.monotonic()
    deadline = start + timeout
    _log(f"Waiting for node /{node_name} (timeout {timeout}s)...")
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["ros2", "node", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if f"/{node_name}" in result.stdout:
            elapsed = time.monotonic() - start
            _log(f"Node /{node_name} appeared after {elapsed:.1f}s")
            return elapsed
        time.sleep(0.5)
    raise TimeoutError(f"Node /{node_name} did not appear within {timeout}s")


def _count_messages(topic: str, duration: float, first_msg_timeout: float = 300.0) -> dict:
    """Subscribe to a topic for `duration` seconds and count messages.

    first_msg_timeout is generous (default 5 min) to allow for first-run
    model compilation (TensorRT engine export, torch.compile, etc.).
    """
    # First, echo one message to confirm the topic is active and show its content
    _log(f"Waiting for first message on {topic} (timeout {first_msg_timeout}s)...")
    result = subprocess.run(
        ["ros2", "topic", "echo", topic, "--once"],
        capture_output=True, text=True, timeout=first_msg_timeout,
    )
    got_first = result.returncode == 0
    _log(f"First message received: {got_first}")
    if got_first and result.stdout.strip():
        # Log the first message (truncated)
        msg_preview = result.stdout.strip()[:2000]
        _log(f"--- First message on {topic} ---")
        for line in msg_preview.splitlines():
            _log(f"  {line}")
        _log(f"--- End first message ---")
    elif result.stderr.strip():
        _log(f"Echo stderr: {result.stderr.strip()[:500]}")

    # Count messages by echoing for the duration
    _log(f"Counting messages on {topic} for {duration}s...")
    proc = subprocess.Popen(
        ["ros2", "topic", "echo", topic],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    time.sleep(duration)
    proc.terminate()
    try:
        stdout, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()

    # Count messages by counting the "---" separators in ros2 topic echo output
    if stdout.strip():
        # Each message in `ros2 topic echo` is separated by "---"
        messages = [m.strip() for m in stdout.split("---") if m.strip()]
        count = len(messages)
    else:
        count = 0

    _log(f"Messages counted: {count}")
    return {
        "first_message_received": got_first,
        "message_count": count,
    }


def _start_input_source(args) -> subprocess.Popen | None:
    """Start the input source (rosbag or usb_cam). Returns the process."""
    if args.input_source == "rosbag":
        return _popen_show(
            ["ros2", "bag", "play", args.bag, "--loop"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    elif args.input_source == "usb_cam":
        video_device = getattr(args, "video_device", "/dev/video0")
        _log(f"Starting usb_cam node (device: {video_device})")
        proc = _popen_show(
            ["ros2", "run", "usb_cam", "usb_cam_node_exe", "--ros-args",
             "-p", f"video_device:={video_device}",
             "-r", "/image_raw:=/camera/image_raw"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        threading.Thread(
            target=_stream_pipe, args=(proc.stderr, "usb_cam"),
            daemon=True,
        ).start()
        return proc
    return None


def _start_rosboard() -> subprocess.Popen | None:
    """Launch rosboard web UI on port 8888."""
    proc = _popen_show(
        ["python3", "/opt/rosboard/run"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    threading.Thread(
        target=_stream_pipe, args=(proc.stderr, "rosboard"),
        daemon=True,
    ).start()
    return proc


def main():
    parser = argparse.ArgumentParser(description="E2E test harness for vision nodes")
    parser.add_argument("--package", required=True, help="ROS2 package name")
    parser.add_argument("--node", required=True, help="ROS2 node executable name")
    parser.add_argument("--input-source", choices=["rosbag", "usb_cam"], default="rosbag",
                        help="Input source for the vision node")
    parser.add_argument("--bag", default=None, help="Path to rosbag directory (required if input-source=rosbag)")
    parser.add_argument("--video-device", default="/dev/video0", help="Video device for usb_cam")
    parser.add_argument("--output-topic", required=True, help="Topic to monitor for output")
    parser.add_argument("--rosboard", action="store_true", help="Start rosboard web UI on port 8888")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")
    args = parser.parse_args()

    if args.input_source == "rosbag" and not args.bag:
        print("ERROR: --bag is required when using rosbag input", file=sys.stderr)
        sys.exit(1)

    # --- Diagnostics ---
    _log("=== E2E Test Harness ===")
    _log(f"Package: {args.package}")
    _log(f"Node:    {args.node}")
    _log(f"Input:   {args.input_source}")
    _log(f"Topic:   {args.output_topic}")
    _log(f"Duration: {args.duration}s")
    if args.input_source == "usb_cam":
        _log(f"Video:   {args.video_device}")
    if args.rosboard:
        _log(f"rosboard: enabled (port 8888)")
    _log("")

    # Show environment
    _log("--- Environment ---")
    for var in ["ROS_DISTRO", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH",
                "LD_LIBRARY_PATH", "PYTHONPATH", "PATH"]:
        _log(f"  {var}={os.environ.get(var, '<unset>')}")
    _log("")

    # Verify package is visible to ROS
    _log("--- Package check ---")
    _run_show(["ros2", "pkg", "list"], timeout=10)
    _log("")

    # List executables in the package
    _log(f"--- Executables in {args.package} ---")
    _run_show(["ros2", "pkg", "executables", args.package], timeout=10)
    _log("")

    # Check install directory
    _log("--- Install directory ---")
    _run_show(["ls", "-la", f"/ros_ws/install/{args.package}/lib/{args.package}/"], timeout=5)
    _log("")

    report = {
        "node_started": False,
        "node_startup_time_s": None,
        "input_source": args.input_source,
        "input_active": False,
        "rosboard_launched": args.rosboard,
        "messages_received": 0,
        "first_message_received": False,
        "output_topic": args.output_topic,
        "duration_s": args.duration,
        "errors": [],
    }

    node_proc = None
    input_proc = None
    viz_proc = None
    rosboard_proc = None

    try:
        # 1. Launch the vision node — stderr is streamed live so we see ROS logs
        #    Use python3 -m to launch instead of ros2 run, because ros2 run
        #    uses the colcon-generated entry point script whose shebang points
        #    at /usr/bin/python3 (system Python, can't see venv packages).
        _log("--- Launching vision node ---")
        node_proc = _popen_show(
            ["python3", "-c",
             f"from {args.package}.node import main; main()"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        # Stream node stderr (ROS logs) in a background thread
        node_log_thread = threading.Thread(
            target=_stream_pipe, args=(node_proc.stderr, "node"),
            daemon=True,
        )
        node_log_thread.start()

        # Give the process a moment, then check if it crashed immediately
        time.sleep(1.0)
        if node_proc.poll() is not None:
            node_log_thread.join(timeout=2)
            stdout = node_proc.stdout.read() if node_proc.stdout else ""
            _log(f"Node crashed immediately with exit code {node_proc.returncode}")
            if stdout.strip():
                _log(f"Node stdout:\n{stdout[:3000]}")
            report["errors"].append(
                f"Node crashed on startup (exit {node_proc.returncode})")
            _emit(report)
            return

        # 2. Wait for node to come up
        try:
            startup_time = _wait_for_node(args.node)
            report["node_started"] = True
            report["node_startup_time_s"] = round(startup_time, 2)
        except TimeoutError as e:
            report["errors"].append(str(e))
            node_proc.terminate()
            _, stderr = node_proc.communicate(timeout=5)
            if stderr:
                _log(f"Node stderr:\n{stderr[:2000]}")
                report["errors"].append(f"Node stderr: {stderr[:2000]}")
            _emit(report)
            return

        # 3. Start input source (rosbag or usb_cam — both run locally)
        _log("--- Starting input source ---")
        input_proc = _start_input_source(args)
        time.sleep(2.0)
        report["input_active"] = True

        # 4. Launch viz node (draws bounding boxes on camera images)
        if os.path.exists("/ros_ws/viz_node.py"):
            _log("--- Launching viz node ---")
            viz_proc = _popen_show(
                ["python3", "/ros_ws/viz_node.py"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            threading.Thread(
                target=_stream_pipe, args=(viz_proc.stderr, "viz"),
                daemon=True,
            ).start()
            time.sleep(1.0)
        else:
            _log("viz_node.py not found, skipping visualization overlay")

        # 5. Launch rosboard if requested (all topics visible in web UI)
        if args.rosboard:
            _log("--- Launching rosboard (http://localhost:8888) ---")
            rosboard_proc = _start_rosboard()
            time.sleep(1.0)

        # 6. Monitor the output topic
        _log("--- Monitoring output ---")

        # Check what topics are available
        _run_show(["ros2", "topic", "list"], timeout=10)

        # Check topic info (non-blocking)
        _log("Checking input topic /camera/image_raw...")
        _run_show(["ros2", "topic", "info", "/camera/image_raw"], timeout=5)
        _log(f"Checking output topic {args.output_topic}...")
        _run_show(["ros2", "topic", "info", args.output_topic], timeout=5)

        msg_info = _count_messages(args.output_topic, args.duration)
        report["first_message_received"] = msg_info["first_message_received"]
        report["messages_received"] = msg_info["message_count"]

    except Exception as e:
        _log(f"Unexpected error: {e}")
        report["errors"].append(f"Unexpected error: {e}")
    finally:
        for proc in [rosboard_proc, viz_proc, input_proc, node_proc]:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        # Capture node exit info
        if node_proc and node_proc.returncode and node_proc.returncode != -15:
            _log(f"Node exited with code {node_proc.returncode}")
            report["errors"].append(f"Node exited with code {node_proc.returncode}")

    _emit(report)


def _emit(report: dict):
    """Print the JSON report to stdout."""
    _log("=== Emitting report ===")
    print("---E2E_REPORT_START---")
    print(json.dumps(report, indent=2))
    print("---E2E_REPORT_END---")
    has_messages = report["messages_received"] != 0  # -1 counts as success (interactive)
    sys.exit(0 if not report["errors"] and has_messages else 1)


if __name__ == "__main__":
    main()
