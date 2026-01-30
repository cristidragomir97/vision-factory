"""E2E test harness — runs inside the Docker container.

Launches the vision node, feeds input (rosbag or usb_cam), optionally
opens rqt, captures output messages, and prints a JSON report to stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def _wait_for_node(node_name: str, timeout: float = 60.0) -> float:
    """Wait until the node appears in `ros2 node list`. Returns startup time."""
    start = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["ros2", "node", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if f"/{node_name}" in result.stdout:
            return time.monotonic() - start
        time.sleep(0.5)
    raise TimeoutError(f"Node /{node_name} did not appear within {timeout}s")


def _count_messages(topic: str, duration: float) -> dict:
    """Subscribe to a topic for `duration` seconds and count messages."""
    result = subprocess.run(
        ["ros2", "topic", "echo", topic, "--once"],
        capture_output=True, text=True, timeout=duration,
    )
    got_first = result.returncode == 0

    proc = subprocess.Popen(
        ["ros2", "topic", "echo", topic, "--no-arr", "--csv"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    time.sleep(duration)
    proc.terminate()
    try:
        stdout, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()

    lines = [l for l in stdout.strip().split("\n") if l.strip()] if stdout.strip() else []
    return {
        "first_message_received": got_first,
        "message_count": len(lines),
    }


def _start_input_source(args) -> subprocess.Popen | None:
    """Start the input source (rosbag or usb_cam). Returns the process."""
    if args.input_source == "rosbag":
        return subprocess.Popen(
            ["ros2", "bag", "play", args.bag, "--loop"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    elif args.input_source == "usb_cam":
        return subprocess.Popen(
            ["ros2", "run", "usb_cam", "usb_cam_node_exe",
             "--ros-args", "-p", "camera_name:=camera",
             "-r", "/image_raw:=/camera/image_raw"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    return None


def _start_rqt(output_topic: str) -> subprocess.Popen | None:
    """Launch rqt_image_view on the output topic."""
    return subprocess.Popen(
        ["ros2", "run", "rqt_image_view", "rqt_image_view",
         "--ros-args", "-r", f"image:={output_topic}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def main():
    parser = argparse.ArgumentParser(description="E2E test harness for vision nodes")
    parser.add_argument("--package", required=True, help="ROS2 package name")
    parser.add_argument("--node", required=True, help="ROS2 node executable name")
    parser.add_argument("--input-source", choices=["rosbag", "usb_cam"], default="rosbag",
                        help="Input source for the vision node")
    parser.add_argument("--bag", default=None, help="Path to rosbag directory (required if input-source=rosbag)")
    parser.add_argument("--output-topic", required=True, help="Topic to monitor for output")
    parser.add_argument("--rqt", action="store_true", help="Open rqt_image_view on the output topic")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")
    args = parser.parse_args()

    if args.input_source == "rosbag" and not args.bag:
        print("ERROR: --bag is required when using rosbag input", file=sys.stderr)
        sys.exit(1)

    report = {
        "node_started": False,
        "node_startup_time_s": None,
        "input_source": args.input_source,
        "input_active": False,
        "rqt_launched": args.rqt,
        "messages_received": 0,
        "first_message_received": False,
        "output_topic": args.output_topic,
        "duration_s": args.duration,
        "errors": [],
    }

    node_proc = None
    input_proc = None
    rqt_proc = None

    try:
        # 1. Launch the vision node
        node_proc = subprocess.Popen(
            ["ros2", "run", args.package, args.node],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

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
                report["errors"].append(f"Node stderr: {stderr[:2000]}")
            _emit(report)
            return

        # 3. Start input source
        input_proc = _start_input_source(args)
        time.sleep(2.0)
        report["input_active"] = True

        # 4. Launch rqt if requested
        if args.rqt:
            rqt_proc = _start_rqt(args.output_topic)
            time.sleep(1.0)

        # 5. Monitor the output topic
        if args.rqt and args.input_source == "usb_cam":
            # Interactive mode: just wait for duration, user watches rqt
            print(f"Running for {args.duration}s — view output in rqt_image_view...")
            time.sleep(args.duration)
            report["messages_received"] = -1  # Not counted in interactive mode
            report["first_message_received"] = True
        else:
            msg_info = _count_messages(args.output_topic, args.duration)
            report["first_message_received"] = msg_info["first_message_received"]
            report["messages_received"] = msg_info["message_count"]

    except Exception as e:
        report["errors"].append(f"Unexpected error: {e}")
    finally:
        for proc in [rqt_proc, input_proc, node_proc]:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        # Capture node exit info
        if node_proc and node_proc.returncode and node_proc.returncode != -15:
            report["errors"].append(f"Node exited with code {node_proc.returncode}")

    _emit(report)


def _emit(report: dict):
    """Print the JSON report to stdout."""
    print("---E2E_REPORT_START---")
    print(json.dumps(report, indent=2))
    print("---E2E_REPORT_END---")
    has_messages = report["messages_received"] != 0  # -1 counts as success (interactive)
    sys.exit(0 if not report["errors"] and has_messages else 1)


if __name__ == "__main__":
    main()
