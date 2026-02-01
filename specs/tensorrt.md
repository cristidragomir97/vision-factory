# TensorRT Backend

## Overview
TensorRT backend for vision-factory that exports and caches TensorRT engines for optimized GPU inference on embedded devices (Jetson). YOLO is the first model to support it, using Ultralytics' native TRT export.

## Architecture

### Manifest-driven backend compatibility
Backend compatibility is declared per-model in `manifest.yaml` via the `backends` field:

```yaml
model:
  name: yolo
  backends: [cuda, rocm, openvino, tensorrt]
```

The `validate_selection()` function in `resolver.py` checks this list. Models without the field default to `[cuda, rocm, openvino]`.

### Engine caching
The TensorRT runner declares two ROS parameters:
- `precision` (default: `fp16`) — `fp16`, `fp32`, or `int8`
- `engine_cache_dir` (default: `/tmp/trt_engines`)

On init:
1. Check if `{cache_dir}/{variant}_{precision}.engine` exists
2. If yes → `model.load_engine(path)`
3. If no → `model.load(device)` → `model.export_tensorrt(...)` → `model.load_engine(path)`

### CUDA memory safety
Every `infer()` path calls `_ensure_contiguous(image)` → `np.ascontiguousarray(image)` if needed. This is critical because `cv_bridge.imgmsg_to_cv2` can return non-contiguous arrays and TensorRT is strict about memory layout.

## Model support

| Model | TensorRT | Notes |
|-------|----------|-------|
| yolo | Yes | Ultralytics native export |
| depth_anything | No | HF model, no TRT path yet |
| grounding_dino | No | HF model, no TRT path yet |
| segment_anything | No | CUDA-only (SAM2 requirements) |
| florence | No | HF encoder-decoder |
| rtmpose | No | ONNX Runtime internally |
| zoedepth | No | HF model |
| bytetrack | No | CPU-based tracking logic |

## Files changed
- `generator/manifest.py` — `backends` field on `ModelInfo`
- `generator/resolver.py` — `tensorrt` backend config, `supported_backends` param in `validate_selection()`
- `generator/engine.py` — passes `manifest.model.backends` to validation
- `generator/context.py` — `TensorRtRunner` in `RUNNER_CLASS_MAP`
- `generator/cli.py` — `tensorrt` in CLI choices
- `models/yolo/model.py` — `export_tensorrt()` and `load_engine()` methods
- `generator/templates/runners/tensorrt.py.j2` — new runner template
- All 8 `models/*/manifest.yaml` — `backends:` field added
