# ONNX Backend Research

## Overview

Adding an `onnx` backend to vision-factory would provide portable, CPU-first inference via ONNX Runtime. Combined with the existing backends, this covers the full spectrum: PyTorch (cuda/rocm), optimized hardware (tensorrt/openvino), and portable inference (onnx).

## Per-model feasibility

| Model | ONNX Feasibility | Effort | Notes |
|-------|-----------------|--------|-------|
| **yolo** | Easy | Low | `model.export(format='onnx')` native in Ultralytics |
| **rtmpose** | Already done | — | Uses ONNX Runtime via rtmlib internally |
| **depth_anything** | Doable | Medium | HF Optimum or `torch.onnx.export` |
| **grounding_dino** | Doable | Medium | HF Optimum supports it, some custom ops |
| **zoedepth** | Partial | Medium | Niche model, may not be in Optimum |
| **florence** | Challenging | High | Multimodal + generation, needs validation |
| **segment_anything** | Hard | High | Custom decoder, scatter ops |
| **bytetrack** | N/A | — | Pure Python tracking, no neural net |

## ONNX Runtime execution providers

ONNX Runtime dispatches inference through "execution providers" (EPs). The runner would default to CPU and expose an `execution_provider` ROS parameter:

- `CPUExecutionProvider` — works everywhere, no drivers needed
- `CUDAExecutionProvider` — NVIDIA GPU acceleration
- `TensorrtExecutionProvider` — NVIDIA via TensorRT through ORT
- `RocmExecutionProvider` — AMD GPU
- `OpenVINOExecutionProvider` — Intel CPU/iGPU

```python
import onnxruntime as ort

providers = [('CPUExecutionProvider', {})]
session = ort.InferenceSession(model_path, providers=providers)
output = session.run(output_names, {input_name: input_array})
```

## Implementation plan

### Infrastructure changes

**resolver.py** — register backend:
```python
"onnx": BackendConfig(
    torch_index_url=None,
    extra_deps=["onnxruntime>=1.17.0"],
),
```

**context.py** — runner class:
```python
"onnx": "OnnxRunner",
```

**cli.py** — add `"onnx"` to `click.Choice`.

**onnx.py.j2** — new runner template. Follows the TensorRT pattern: export + cache on first run, load from cache on subsequent runs. Closest reference is `openvino.py.j2` since inference is non-PyTorch.

### YOLO (Ultralytics native export)

Add to `models/yolo/model.py`:
```python
def export_onnx(self, opset=14, cache_dir="/tmp/onnx_models", model_name="model.onnx"):
    export_path = self.model.export(format='onnx', opset=opset)
    # Cache logic identical to export_tensorrt()
    return cached_path

def load_onnx(self, onnx_path):
    self.model = YOLO(str(onnx_path))
```

Update manifest: `backends: [cuda, rocm, openvino, tensorrt, onnx]`

### RTMPose (already ONNX)

RTMPose via `rtmlib` already uses ONNX Runtime. The `onnxruntime` dependency is already declared in `MODEL_DEPS`. Just add `onnx` to its manifest backends — the runner template would recognize the `onnx` model family and skip export.

### HuggingFace models (depth_anything, grounding_dino, etc.)

Two export paths:

**Option A: HF Optimum (preferred)**
```python
from optimum.onnxruntime import ORTModelForDepthEstimation

model = ORTModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Base-hf",
    export=True,
)
```
Advantages: official HF path, handles architecture-specific optimizations. Limitation: not all architectures supported.

**Option B: torch.onnx.export (fallback)**
```python
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=14,
                  dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}})
```
Advantages: works for any PyTorch model. Limitation: manual, needs per-model tuning.

Pre/post processing still requires the HF processor (tokenizer, image transforms), so torch isn't fully eliminated for HF models.

## Technical challenges

### Dynamic shapes
ONNX Runtime is strict about input shapes. Models with variable-size inputs need explicit dynamic axes during export and careful preprocessing to match expected dimensions.

### Operator coverage
Not all PyTorch ops have ONNX equivalents. Risk areas:
- Grounding DINO: complex attention mechanisms
- Florence: beam search generation
- SAM: mask decoder scatter operations
- Depth Anything: generally safe (backbone + decoder)

### Quantization
Unlike TensorRT (automatic FP16/INT8), ONNX quantization requires explicit calibration data and per-model validation. This is a follow-on concern, not needed for MVP.

## Recommended rollout

**Phase 1 (MVP):** YOLO + RTMPose. Both have native ONNX support. Validates the runner/backend infrastructure.

**Phase 2:** Depth Anything via HF Optimum. Simplest HF model, proves the Optimum export path.

**Phase 3:** Grounding DINO, ZoeDepth. Expand HF coverage based on Optimum compatibility.

**Phase 4 (stretch):** Florence, SAM. Complex architectures, may need custom export logic or may not be feasible.
