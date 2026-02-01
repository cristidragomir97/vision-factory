# vision-factory

A code generator that produces self-contained ROS2 vision packages.

## Why a Generator?

A single universal vision node that supports every model and backend would accumulate the union of all dependencies — PyTorch, TensorRT, ONNX Runtime, OpenVINO, Ultralytics, Transformers, and more. That image would be enormous, most of it unused for any given deployment, and these libraries frequently conflict with each other.

Vision-factory generates standalone packages instead. Each output contains only the code and dependencies it actually needs. A YOLO + CUDA package pulls in `ultralytics` and `torch+cu124`. A Depth Anything + ONNX package pulls in `onnxruntime` and `transformers`. Neither drags in the other's stack.

The generated packages are small, self-documenting, and easy to audit — you can read every line of code that will run on your robot.

## Supported Models

| Model | Output | Variants |
|-------|--------|----------|
| YOLO | Object detection (xyxy) | v8n, v8s, v8m, v8l, v8x, v11n, v11s, v11m, v11l |
| Depth Anything | Depth map | vits, vitb, vitl (v1 & v2) |
| Grounding DINO | Detection + text prompts | tiny, base |
| Segment Anything | Segmentation masks | hiera_t/s/b+/l, large, huge |
| Florence | Multi-task VLM | base, base-ft, large, large-ft |
| RTMPose | Keypoint detection | body-s/m/l, hand, face, wholebody-s/m/l |
| ZoeDepth | Monocular depth | nyu, kitti, nyu-kitti |
| ByteTrack | Multi-object tracking | - |

## Hardware Platforms

| Backend | Device | Notes |
|---------|--------|-------|
| `cuda` | NVIDIA GPU | PyTorch + `torch.compile`. Broadest model support. |
| `tensorrt` | NVIDIA GPU | Optimized TensorRT engines. Currently supports YOLO. |
| `rocm` | AMD GPU | Same PyTorch code paths as CUDA, ROCm stack. |
| `openvino` | Intel CPU/iGPU/NPU | Intel-optimized inference. Good for NUCs and edge PCs. |
| `onnx` | Any (CPU, CUDA, ROCm, OpenVINO, NPU) | Portable ONNX Runtime with selectable execution providers. |

See [docs/hardware.md](docs/hardware.md) for per-platform details, example commands, and NPU guidance.

## Quick Start

```bash
pip install -e .
```

Generate a package:

```bash
vision-factory generate \
  --model yolo \
  --backend cuda \
  --variant yolo_v8s \
  --package-name my_detector \
  --output ./output
```

List available models:

```bash
vision-factory list-models
```

## Documentation

- [Hardware Platforms](docs/hardware.md) — per-device guidance, example commands, NPU support
- [Backends](docs/backends.md) — detailed backend descriptions (cuda, rocm, tensorrt, onnx, openvino)
- [Architecture](docs/architecture.md) — pipeline diagram, template layers, call chain, project structure
- [Development](docs/development.md) — adding new models, testing, installation
