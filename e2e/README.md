# E2E Tests

End-to-end tests that generate a ROS2 vision package, build it in Docker, and run full inference. Supports rosbag replay or live webcam input, with optional rosboard visualization.

## Prerequisites

- Docker
- NVIDIA GPU + `nvidia-smi` (for `cuda`/`rocm` backends)
- For `--usb-cam`: a webcam at `/dev/video0`
- For `--rosboard`: rosboard web UI (accessible at `http://localhost:8888`)

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

### With rosboard visualization

```bash
# Rosbag + rosboard
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --rosbag /path/to/camera_bag \
    --rosboard

# Live webcam + rosboard (interactive demo)
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --usb-cam \
    --rosboard \
    --duration 60
```

## Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--model` | yes | - | Model architecture (e.g. `yolo`, `depth_anything`) |
| `--backend` | yes | - | Hardware backend (`cuda`, `rocm`, `openvino`) |
| `--rosbag` | one of | - | Path to rosbag directory with camera images |
| `--usb-cam` | one of | - | Use `usb_cam` package as live input |
| `--rosboard` | no | off | Start rosboard web UI on port 8888 |
| `--variant` | no | model default | Model variant (e.g. `yolo_v8s`) |
| `--package-name` | no | `e2e_<model>` | Name for the generated package |
| `--ros-distro` | no | `jazzy` | ROS2 distribution |
| `--duration` | no | `30.0` | Test duration in seconds |
| `--video-device` | no | `/dev/video0` | Video device path for usb_cam |
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

Reads from a USB camera instead of a rosbag. The container needs access to the video device. Still runs the automated message-counting harness unless combined with `--rosboard`.

```bash
python e2e/run.py --model yolo --backend cuda --usb-cam
```

Make sure your camera is at `/dev/video0`, or specify the device with `--video-device /dev/video2`.

### Interactive demo (usb_cam + rosboard)

Starts rosboard, a web-based visualization tool, showing the vision node's output in real time. No X11 forwarding required — just open `http://localhost:8888` in your browser.

```bash
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --usb-cam \
    --rosboard \
    --duration 120
```

In this mode the harness doesn't count messages — it just keeps the node running for `--duration` seconds while you watch the rosboard UI. The report will still be generated with stage timings.

### Rosbag + rosboard

You can also visualize rosbag replay in rosboard:

```bash
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --rosbag /path/to/bag \
    --rosboard
```

Open `http://localhost:8888` to view the output. In this mode the harness still counts messages (since rosbag is finite/looped), so you get both the visual output and an automated report.

## What It Does

1. **Generate** -- Calls the generator to produce a ROS2 package zip
2. **Build & Run** -- Starts a single container from the pre-built base image, builds the package with colcon, launches the vision node + input source + viz + rosboard (if enabled), monitors the output topic

Docker run flags adapt to the mode:
- `--gpus all` for cuda/rocm backends
- `--device /dev/video0` for usb_cam
- Port mapping (`-p 8888:8888`) for rosboard

## Output

A markdown report at `<output-dir>/e2e-report.md` with:

- Pass/fail status per stage
- Timing for each stage
- Input source and rosboard status
- Inference summary (message count, startup time)
- Errors (if any)
- Collapsible build/run logs
