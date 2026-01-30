# E2E Tests

End-to-end tests that generate a ROS2 vision package, build it in Docker, and run full inference. Supports rosbag replay or live webcam input, with optional rqt visualization.

## Prerequisites

- Docker
- NVIDIA GPU + `nvidia-smi` (for `cuda`/`rocm` backends)
- For `--usb-cam`: a webcam at `/dev/video0`
- For `--rqt`: X11 display (run `xhost +local:docker` first)

## Usage

### Rosbag mode (automated test)

```bash
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --rosbag /path/to/camera_bag \
    --ros-distro jazzy
```

### Live webcam mode

```bash
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --usb-cam \
    --ros-distro jazzy
```

### With rqt visualization

```bash
# Rosbag + rqt
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --rosbag /path/to/camera_bag \
    --rqt

# Live webcam + rqt (interactive demo)
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --usb-cam \
    --rqt \
    --duration 60
```

## Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--model` | yes | - | Model architecture (e.g. `yolo`, `depth_anything`) |
| `--backend` | yes | - | Hardware backend (`cuda`, `rocm`, `openvino`) |
| `--rosbag` | one of | - | Path to rosbag directory with camera images |
| `--usb-cam` | one of | - | Use `usb_cam` package as live input |
| `--rqt` | no | off | Open `rqt_image_view` on the output topic |
| `--variant` | no | model default | Model variant (e.g. `yolo_v8s`) |
| `--package-name` | no | `e2e_<model>` | Name for the generated package |
| `--ros-distro` | no | `jazzy` | ROS2 distribution |
| `--duration` | no | `30.0` | Test duration in seconds |
| `--output-dir` | no | `./e2e-output` | Directory for the report |

`--rosbag` and `--usb-cam` are mutually exclusive; one must be provided.

## Modes

### Automated test (rosbag)

The default mode. Replays a rosbag, counts output messages, and produces a pass/fail report. Good for CI or validating that a generated package works end-to-end.

```bash
python e2e/run.py --model yolo --backend cuda --rosbag /path/to/bag
```

The container runs headless — no display needed. The test harness subscribes to the output topic, counts messages for `--duration` seconds, and exits with code 0 if messages were received.

### Live webcam (usb_cam)

Reads from a USB camera instead of a rosbag. The container needs access to the video device. Still runs the automated message-counting harness unless combined with `--rqt`.

```bash
python e2e/run.py --model yolo --backend cuda --usb-cam
```

Make sure your camera is at `/dev/video0`. If it's at a different device, you'll need to modify the device passthrough in `run.py`.

### Interactive demo (usb_cam + rqt)

Opens an `rqt_image_view` window showing the vision node's output in real time. This mode requires X11 forwarding from the host into the container.

```bash
# 1. Allow Docker to access your X11 display
xhost +local:docker

# 2. Run with webcam + rqt
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --usb-cam \
    --rqt \
    --duration 120

# 3. Revoke X11 access when done
xhost -local:docker
```

In this mode the harness doesn't count messages — it just keeps the node running for `--duration` seconds while you watch the rqt window. The report will still be generated with stage timings.

**Wayland users:** If you're on Wayland instead of X11, you may need to set `DISPLAY=:0` and ensure XWayland is running, or use `xdg-open` with an alternative viewer.

### Rosbag + rqt

You can also visualize rosbag replay in rqt. Same X11 setup as above:

```bash
xhost +local:docker
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --rosbag /path/to/bag \
    --rqt
```

In this mode the harness still counts messages (since rosbag is finite/looped), so you get both the visual output and an automated report.

## What It Does

1. **Generate** -- Calls the generator to produce a ROS2 package zip
2. **Build** -- Renders an e2e Dockerfile (with conditional deps for rosbag/usb_cam/rqt), builds the Docker image
3. **Run** -- Starts the container, launches the vision node, feeds input, monitors the output topic

Docker run flags adapt to the mode:
- `--gpus all` for cuda/rocm backends
- `--device /dev/video0` for usb_cam
- X11 forwarding (`-e DISPLAY`, `-v /tmp/.X11-unix`) for rqt

## Output

A markdown report at `<output-dir>/e2e-report.md` with:

- Pass/fail status per stage
- Timing for each stage
- Input source and rqt status
- Inference summary (message count, startup time)
- Errors (if any)
- Collapsible build/run logs
