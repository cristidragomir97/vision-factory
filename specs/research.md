# Vision Node Research

## Model Framework & Format Analysis

All target models use **PyTorch** as their primary framework. The key differentiation is how they're loaded and what ecosystem they belong to.

### Loading Ecosystem Breakdown

| Model | Loading Path | HuggingFace | ONNX Export | Notes |
|-------|-------------|-------------|-------------|-------|
| YOLO v8/v11 | Ultralytics API | No | Official | `YOLO("yolov8n.pt")` |
| Grounding DINO | HF Transformers | Yes | Partial | Use HF version, not original repo |
| Depth Anything v2 | HF Transformers | Yes | Convertible | `pipeline("depth-estimation")` |
| SAM2 | Custom (Meta) | No | Partial (decoder only) | Two-stage: encoder cached, decoder per prompt |
| Florence-2 | HF Transformers | Yes | Community | Skip flash-attn, use SDPA |
| RTMPose | rtmlib (ONNX) | No | Via MMDeploy | rtmlib avoids mmcv/mmpose deps |
| ZoeDepth | HF Transformers | Yes | Via OpenVINO | Needs patch for torch 2.1+ |
| ByteTrack | Pure Python | N/A | N/A | Tracking algorithm, not a model |

### HuggingFace Transformers API (4 models)

```python
# Common pattern for DINO, Depth Anything, ZoeDepth, Florence-2:
from transformers import AutoProcessor, AutoModelForXxx
processor = AutoProcessor.from_pretrained("repo/name")
model = AutoModelForXxx.from_pretrained("repo/name").to("cuda")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
result = processor.post_process_xxx(outputs)
```

### Ultralytics API (YOLO)

```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
results = model(image, conf=0.25, iou=0.45)
# results[0].boxes.xyxy, .conf, .cls
```

### Custom Loading (SAM, RTMPose)

```python
# SAM2 - two stage
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus")
predictor.set_image(image)  # Encodes once (heavy)
masks, scores, _ = predictor.predict(point_coords=pts)  # Decodes (light)

# RTMPose via rtmlib - ONNX runtime, no torch needed
from rtmlib import RTMPOSE
pose = RTMPOSE(model="rtmpose-m", backend="onnxruntime")
keypoints, scores = pose(image, bboxes=person_boxes)
```

---

## Preprocessing Commonality

### Shared Across Nearly All Models

| Step | Models Using It | Configurable Params |
|------|----------------|-------------------|
| BGR→RGB conversion | All | - |
| ImageNet normalize | All except YOLO | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| Divide by 255 | YOLO only | scale=255.0 |
| Letterbox resize | YOLO | target_size |
| Resize longest | DINO, SAM | max_size |
| Direct resize | Depth models, Florence | target_size |
| HWC→CHW | All | - |
| Add batch dim | All | - |

### Model-Specific Preprocessing

| Model | Extra Preprocessing |
|-------|-------------------|
| Grounding DINO | BERT text tokenization (handled by HF processor) |
| Florence-2 | Task prompt text encoding (handled by HF processor) |
| SAM | Prompt encoding (points, boxes, masks) |
| RTMPose | Crop image to person bounding boxes |

**Key insight**: HuggingFace `AutoProcessor` handles all preprocessing for its models. We only need custom preprocessing for YOLO, SAM, and RTMPose.

---

## Postprocessing Per Model

### Detection Models

| Model | Output Format | Postprocessing |
|-------|--------------|----------------|
| YOLO v8/v11 | (1, 84, 8400) xywh + class scores | Transpose, NMS, xywh→xyxy, class extraction |
| YOLO v5 | (1, 25200, 85) xywh + obj_conf + classes | NMS, obj_conf × class_scores |
| Grounding DINO | cxcywh normalized + logits | Sigmoid, box threshold, text threshold, phrase match |
| Florence-2 (OD) | Text tokens → parsed boxes + labels | Text decoding, coordinate parsing |

### Depth Models

| Model | Output Format | Postprocessing |
|-------|--------------|----------------|
| Depth Anything v2 | (1, H, W) relative depth | Resize to original, normalize 0-1 |
| ZoeDepth | (1, H, W) metric depth (meters) | Clip range, domain-specific scaling |

### Segmentation

| Model | Output Format | Postprocessing |
|-------|--------------|----------------|
| SAM2 | (1, N, H, W) masks + (1, N) IoU scores | Resize to original, select best mask, binarize |

### Pose

| Model | Output Format | Postprocessing |
|-------|--------------|----------------|
| RTMPose | (N, K, 2) keypoints + (N, K) scores | Score thresholding, coordinate transform back to full image |

### Tracking

| Model | Output Format | Postprocessing |
|-------|--------------|----------------|
| ByteTrack | Tracked boxes + IDs | Track lifecycle management, ID persistence |

---

## Dependency Version Research

### Baseline Compatible Environment

```
python >= 3.10              # SAM2 requirement (most restrictive)
torch == 2.5.1              # SAM2 minimum, works for all others
torchvision == 0.20.1       # Matches torch 2.5.1
CUDA 12.4 or 12.6           # Best for torch 2.5.1
numpy < 2.0, >= 1.21.6     # ByteTrack incompatible with numpy 2.0
transformers >= 4.40, < 5.0 # 5.0.0 requires torch >= 2.6
```

### Per-Model Version Requirements

#### YOLO v8/v11 (Ultralytics)
- Python: >= 3.8
- PyTorch: >= 1.8 (very flexible)
- `ultralytics >= 8.2.70`
- No specific CUDA version (follows PyTorch)

#### Grounding DINO (via HuggingFace)
- `transformers >= 4.40`
- Uses `GroundingDinoProcessor` + `GroundingDinoForObjectDetection`
- Original repo requires CUDA_HOME for custom ops - **avoid, use HF version**

#### Depth Anything v2
- No strict version pins
- Works with any modern torch 2.x
- Deps: torch, torchvision, timm, opencv, einops

#### SAM2 (segment-anything-2) ⚠️ Most Restrictive
- **Python >= 3.10**
- **torch >= 2.5.1**
- **torchvision >= 0.20.1**
- Benchmarked on CUDA 12.4

#### Florence-2
- `transformers >= 4.40` (native support since Aug 2024)
- **flash-attention NOT required** - use `attn_implementation="sdpa"` workaround
- The transformers lib mistakenly lists flash_attn as required

#### RTMPose
- **Option A (recommended)**: `rtmlib` - No PyTorch needed, ONNX only
  - Deps: numpy, opencv, onnxruntime
- **Option B**: Full MMPose - Complex version matrix (mmcv must match torch exactly)

#### ZoeDepth
- Official env: torch ~1.13, timm 0.6.12
- **Works on torch 2.5.1** with a patch for `interpolate` type signature
- Use HF version: `Intel/zoedepth-nyu-kitti`

#### ByteTrack
- **numpy < 2.0** (cython_bbox binary incompatibility)
- scipy, lap==0.4.0, Cython==0.29.34
- No GPU required (CPU tracking algorithm)

### Known Conflicts & Resolutions

| Conflict | Impact | Resolution |
|----------|--------|-----------|
| SAM2 torch>=2.5.1 vs ZoeDepth ~1.13 | Can't use both | ZoeDepth works on 2.5.1 with HF version + minor fix |
| transformers 5.0 needs torch>=2.6 | Incompatible with SAM2's torch 2.5.1 | Pin transformers < 5.0 |
| ByteTrack needs numpy<2.0 | Many libs moving to numpy 2.0 | Pin globally, ByteTrack is the bottleneck |
| Florence-2 "requires" flash-attn | flash-attn is painful to install | Use SDPA attention instead |
| RTMPose MMPose needs exact mmcv version | mmcv version matrix is fragile | Use rtmlib (ONNX) instead, zero mmcv deps |

### Can All Models Coexist in One Environment?

**Yes.** The compatible set is:

```
python 3.10+
torch 2.5.1 + CUDA 12.4
torchvision 0.20.1
numpy >= 1.21.6, < 2.0
transformers >= 4.40, < 5.0
timm >= 0.6.7
ultralytics >= 8.2.70
rtmlib (for pose, no torch conflict)
scipy, lap==0.4.0 (for ByteTrack)
```

---

## Packaging Strategy: Core + Plugin Packages

### Problem

Different robots need different models. A warehouse robot needs YOLO + depth. A humanoid needs pose + segmentation. Shipping all dependencies to all robots wastes space and creates conflict risk.

### Solution: Separate Colcon Packages

```
the-vision-node/                    # Colcon workspace
├── vision_node_core/               # Always required
│   ├── package.xml                 # Minimal: rclpy, sensor_msgs, cv_bridge
│   ├── runners/                    # CUDA, ROCm, OpenVINO
│   ├── loaders/                    # Memory management
│   └── models/base.py             # BaseModel interface
│
├── vision_node_yolo/               # Selective
│   ├── package.xml                 # depends: core + ultralytics
│   └── models/yolo.py
│
├── vision_node_dino/               # Selective
│   ├── package.xml                 # depends: core + transformers
│   └── models/grounding_dino.py
│
├── vision_node_depth/              # Selective
│   ├── package.xml                 # depends: core + transformers + timm
│   └── models/depth_anything.py, zoedepth.py
│
├── vision_node_sam/                # Selective
│   ├── package.xml                 # depends: core + segment-anything-2
│   └── models/sam.py
│
├── vision_node_florence/           # Selective
│   ├── package.xml                 # depends: core + transformers
│   └── models/florence.py
│
├── vision_node_pose/               # Selective
│   ├── package.xml                 # depends: core + rtmlib + onnxruntime
│   └── models/rtmpose.py
│
├── vision_node_tracking/           # Selective
│   ├── package.xml                 # depends: core + scipy + lap
│   └── trackers/bytetrack.py
│
└── vision_node_bringup/            # Meta-package (all plugins)
    └── package.xml
```

### User Workflow

```bash
# Build only what you need
colcon build --packages-select vision_node_core vision_node_yolo vision_node_depth

# Build everything
colcon build --packages-up-to vision_node_bringup

# Add a model later
colcon build --packages-select vision_node_sam
```

### Why Not a Single Package?

- colcon doesn't handle conditional Python deps
- `extras_require` isn't respected by colcon build
- Separate packages = separate `package.xml` = clean dependency declarations
- Follows the Nav2 pattern (separate planner packages, loaded at runtime)
- Each plugin can be versioned independently

### Runtime Model Discovery

The core node discovers available models at runtime by checking which plugin packages are installed:

```python
# In vision_node_core, discover available models:
import importlib

PLUGIN_MODULES = {
    "yolo": "vision_node_yolo.models.yolo",
    "grounding_dino": "vision_node_dino.models.grounding_dino",
    "depth_anything": "vision_node_depth.models.depth_anything",
    # ...
}

available_models = {}
for name, module_path in PLUGIN_MODULES.items():
    try:
        mod = importlib.import_module(module_path)
        available_models[name] = mod.get_model_class()
    except ImportError:
        pass  # Plugin not installed, skip
```

---

## Shared vs Per-Model Code Summary

### Shared (in vision_node_core)

- **Preprocessing**: resize, normalize, format conversion (~200 lines)
- **Runner interface**: load/unload/infer for CUDA/ROCm/OpenVINO (~800 lines)
- **Loader strategies**: preload/dynamic with VRAM management (~400 lines)
- **Box utilities**: format conversion, NMS, coordinate scaling (~150 lines)
- **Visualization**: box drawing, mask overlay, depth colormap (~200 lines)
- **ROS node**: topic management, service handling (~300 lines)

**Total shared: ~2000 lines**

### Per-Model (in plugin packages)

Each model needs ~50-150 lines of custom code, primarily postprocessing:

| Model | Pre | Post | Total | What's Custom |
|-------|-----|------|-------|--------------|
| YOLO | 0 | 100 | ~100 | NMS, output shape handling |
| Grounding DINO | 0* | 80 | ~80 | Phrase matching (*HF processor handles pre) |
| Depth Anything | 0* | 50 | ~50 | Depth normalization |
| ZoeDepth | 0* | 60 | ~60 | Metric scaling, domain routing |
| Florence-2 | 0* | 120 | ~120 | Multi-task text decoding |
| SAM2 | 30 | 120 | ~150 | Prompt encoding, two-stage, mask selection |
| RTMPose | 40 | 60 | ~100 | Box cropping, keypoint mapping |
| ByteTrack | 0 | 50 | ~50 | Track state management |

**Total per-model: ~710 lines across 8 models**

The ratio is roughly **3:1 shared-to-custom**, meaning the architecture provides significant reuse.
