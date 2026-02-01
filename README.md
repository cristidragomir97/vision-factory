# vision-factory

Pick a model, pick a backend, get a ROS2 package — ready to `colcon build` and deploy. No dependency soup, no 30 GB images, no "works on my laptop."

Vision-factory is a code generator that produces **self-contained ROS2 vision packages**. Instead of one monolithic node that drags in PyTorch, TensorRT, ONNX Runtime, OpenVINO, Ultralytics, and Transformers all at once, each generated package ships only what it needs. YOLO on TensorRT? You get Ultralytics + TensorRT. Depth Anything on ONNX? You get `onnxruntime` + `transformers`. Clean, auditable, conflict-free.

## Supported Models

| Model | What it does | Variants |
|-------|--------------|----------|
| **YOLO** | Object detection | v8n/s/m/l/x, v11n/s/m/l |
| **Grounding DINO** | Open-vocabulary detection (text prompts) | tiny, base |
| **Depth Anything** | Monocular depth estimation | vits, vitb, vitl (v1 & v2) |
| **ZoeDepth** | Metric monocular depth | nyu, kitti, nyu-kitti |
| **Segment Anything** | Promptable segmentation | hiera_t/s/b+/l, large, huge |
| **Florence** | Multi-task vision-language | base, base-ft, large, large-ft |
| **RTMPose** | Keypoint / pose estimation | body-s/m/l, hand, face, wholebody-s/m/l |
| **ByteTrack** | Multi-object tracking | — |

## Runners & Hardware

Every generated package includes a **runner** — a thin layer that wires up device setup, model loading, and the `preprocess -> forward -> postprocess` loop. You choose the runner at generation time with `--backend`, and the generated code is specialized for that backend. No runtime dispatch, no unused code paths.

| Backend | Runner | Device | What you get |
|---------|--------|--------|--------------|
| `cuda` | `CudaRunner` | NVIDIA GPU | PyTorch + `torch.compile` for fused ops after warmup. Broadest model support. |
| `tensorrt` | `TensorRtRunner` | NVIDIA GPU | Exports and caches TensorRT engines (FP16/INT8). Fastest inference on NVIDIA hardware. |
| `rocm` | `RocmRunner` | AMD GPU | Same PyTorch code paths as CUDA, built against the ROCm stack. |
| `openvino` | `OpenVinoRunner` | Intel CPU / iGPU / NPU | Intel-optimized inference via OpenVINO Runtime. Great for NUCs and edge PCs. |
| `onnx` | `OnnxRunner` | Any | Portable ONNX Runtime with selectable execution providers (CPU, CUDA, ROCm, OpenVINO, NPU). |

The CUDA and ROCm runners optionally apply `torch.compile` for graph-level fusion — the first frame compiles, then every subsequent frame runs the optimized graph. The TensorRT runner goes further: it exports a TensorRT engine on first launch, caches it to disk, and reloads it instantly on subsequent runs. The ONNX runner follows the same caching pattern for its exported `.onnx` models.

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
