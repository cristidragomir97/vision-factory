# Vision Node - Architecture & Packaging Plan

## Decisions Made

1. **HuggingFace Transformers**: Yes, core dependency for compatible models
2. **Model Format**: PyTorch native (ONNX as optional optimization later)
3. **Pipeline Models**: User provides boxes (models stay decoupled)

---

## Package Architecture: Core + Plugins

Following the Nav2 pattern (pluginlib), split into separate ROS packages so colcon only builds what the user needs:

```
the-vision-node/                    # Monorepo (colcon workspace)
├── vision_node_core/               # Base package - always required
│   ├── package.xml
│   ├── setup.py
│   ├── runners/                    # CUDA, ROCm, OpenVINO
│   ├── loaders/                    # Preload, dynamic
│   └── models/base.py             # BaseModel interface
│
├── vision_node_yolo/               # YOLO plugin package
│   ├── package.xml                 # depends: vision_node_core, ultralytics
│   ├── setup.py
│   ├── manifests/yolo.yaml
│   └── models/yolo.py
│
├── vision_node_dino/               # Grounding DINO plugin
│   ├── package.xml                 # depends: vision_node_core, transformers
│   └── models/grounding_dino.py
│
├── vision_node_depth/              # Depth models plugin
│   ├── package.xml                 # depends: vision_node_core, transformers
│   ├── models/depth_anything.py
│   └── models/zoedepth.py
│
├── vision_node_sam/                # SAM plugin
│   ├── package.xml                 # depends: vision_node_core, segment-anything-2
│   └── models/sam.py
│
├── vision_node_florence/           # Florence-2 plugin
│   ├── package.xml                 # depends: vision_node_core, transformers
│   └── models/florence.py
│
├── vision_node_pose/               # Pose plugin
│   ├── package.xml                 # depends: vision_node_core, rtmlib
│   └── models/rtmpose.py
│
├── vision_node_tracking/           # Tracking plugin
│   ├── package.xml                 # depends: vision_node_core, scipy, lap
│   └── trackers/bytetrack.py
│
└── vision_node_bringup/            # Meta-package for convenience
    └── package.xml                 # depends on all plugins (optional)
```

### User Workflow

```bash
# User only wants YOLO + depth on their robot:
colcon build --packages-select vision_node_core vision_node_yolo vision_node_depth

# User wants everything:
colcon build --packages-select vision_node_bringup

# User adds SAM later:
colcon build --packages-select vision_node_sam
```

### Why This Works

- **Isolated dependencies**: SAM needing torch>=2.5.1 doesn't affect YOLO
- **Selective builds**: Only compile what you deploy
- **Independent versioning**: Each plugin can pin its own deps
- **colcon native**: Uses `COLCON_IGNORE` or `--packages-select`, no hacks

---

## Dependency Version Matrix

### Baseline Environment (driven by SAM2 being the most restrictive)

```
python >= 3.10          # SAM2 requirement
torch == 2.5.1          # SAM2 minimum, works for all others
torchvision == 0.20.1   # Matches torch 2.5.1
CUDA 12.4 or 12.6       # Best for torch 2.5.1
numpy < 2.0             # ByteTrack incompatible with numpy 2.0
transformers >= 4.40, < 5.0  # 5.0 requires torch 2.6
```

### Per-Plugin Dependencies

| Plugin | Extra Dependencies | Version Constraints |
|--------|-------------------|-------------------|
| **core** | torch, torchvision, numpy, opencv, pyyaml | torch>=2.5.1, numpy<2.0 |
| **yolo** | ultralytics | >=8.2.70 |
| **dino** | transformers | >=4.40,<5.0 |
| **depth** | transformers, timm | timm>=0.6.7 |
| **sam** | segment-anything-2 | torch>=2.5.1 |
| **florence** | transformers | >=4.40,<5.0 (NO flash-attn, use SDPA) |
| **pose** | rtmlib, onnxruntime | No torch required (ONNX-based) |
| **tracking** | scipy, lap | lap==0.4.0, numpy<2.0 |

### Known Conflicts & Resolutions

| Conflict | Resolution |
|----------|-----------|
| SAM2 needs torch>=2.5.1, ZoeDepth tested on ~1.13 | ZoeDepth works on 2.5.1 with minor patch (interpolate type fix) |
| ByteTrack needs numpy<2.0 | Pin numpy<2.0 globally |
| transformers 5.0 needs torch>=2.6 | Stay on transformers 4.x |
| Florence-2 lists flash-attn as required | Use `attn_implementation="sdpa"` instead |
| RTMPose (MMPose) has complex mmcv deps | Use rtmlib instead (ONNX-only, no mmcv) |

### Can All Models Coexist?

**Yes**, with the baseline above. The key insight is:
- torch 2.5.1 satisfies everyone
- transformers 4.x works for all HF models
- rtmlib avoids the MMPose/mmcv dependency hell
- numpy<2.0 is the only "annoying" pin (ByteTrack)

---

## Model Abstraction Analysis

### What's Repeatable (shared in core)

| Layer | Shared % | Details |
|-------|----------|---------|
| Preprocessing | 95% | BGR→RGB, ImageNet normalize, resize, HWC→CHW |
| Runner interface | 100% | load/unload/infer/memory management |
| Box utilities | 70% | xyxy↔xywh↔cxcywh, NMS, coordinate rescaling |
| Visualization | 80% | Box drawing, mask overlay, depth colormap |

### What Needs Per-Model Code (~50-150 lines each)

| Model | Custom Pre | Custom Post | Lines Est. |
|-------|-----------|-------------|------------|
| YOLO | None | NMS, xywh→xyxy, class scores | ~100 |
| Grounding DINO | Text tokenization | Phrase matching | ~80 |
| Depth Anything | None | Resize to original | ~50 |
| ZoeDepth | None | Metric scaling, domain routing | ~60 |
| Florence-2 | Task prompt | Text decoding per task | ~120 |
| SAM | Prompt encoding | Multi-mask selection, two-stage | ~150 |
| RTMPose | Crop to boxes | Keypoint mapping | ~100 |
| ByteTrack | N/A (not a model) | Track ID management | ~50 |

### Three Loading Paths

```python
# Path 1: HuggingFace (DINO, Depth Anything, ZoeDepth, Florence)
processor = AutoProcessor.from_pretrained(repo)
model = AutoModelFor*.from_pretrained(repo).to(device)

# Path 2: Ultralytics (YOLO)
model = YOLO(weights_path)

# Path 3: Custom (SAM, RTMPose via rtmlib)
# Model-specific loading code
```

---

## Extensibility: Adding New Models

### Problem
New vision models appear constantly. Adding a model should NOT require:
- Modifying core code
- Touching any __init__.py import lists
- Rebuilding existing packages

### Solution: Registry + Python Entry Points

Two mechanisms working together:

#### 1. Registry Pattern (MMDetection-style decorator)

Each model class registers itself with a decorator:

```python
# In vision_node_core/registry.py
class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, name=None):
        def decorator(model_cls):
            key = name or model_cls.__name__
            cls._models[key] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name):
        return cls._models[name]

    @classmethod
    def available(cls):
        return list(cls._models.keys())
```

```python
# In a plugin package (vision_node_yolo/models/yolo.py)
from vision_node_core.registry import ModelRegistry
from vision_node_core.models.base import BaseModel

@ModelRegistry.register("yolo_v8")
class YOLOv8Model(BaseModel):
    def preprocess(self, image): ...
    def postprocess(self, outputs): ...
```

#### 2. Entry Points for Cross-Package Discovery

Decorators only fire when imported. Entry points solve discovery across packages:

```python
# Plugin's setup.py
setup(
    name='vision_node_yolo',
    entry_points={
        'the_vision_node.models': [
            'yolo_v8 = vision_node_yolo.models.yolo:YOLOv8Model',
            'yolo_v11 = vision_node_yolo.models.yolo:YOLOv11Model',
        ],
    },
)
```

```python
# Core discovers all installed plugins at startup
from importlib.metadata import entry_points

def discover_models():
    """Find all model plugins from any installed package."""
    discovered = entry_points(group='the_vision_node.models')
    for ep in discovered:
        model_class = ep.load()  # This import triggers @register
    # Now ModelRegistry._models has everything
```

#### How It Works Together

```
1. User installs vision_node_yolo package
2. Its setup.py declares entry_points under 'the_vision_node.models'
3. Core node starts, calls discover_models()
4. importlib.metadata finds all packages with that entry point group
5. ep.load() imports the module → triggers @ModelRegistry.register()
6. Registry now knows about yolo_v8, yolo_v11
7. Node can instantiate any discovered model by name
```

### What a Contributor Needs to Do (Add New Model)

**Touch 3 files, never touch core:**

```
my_new_model_package/
├── package.xml              # ROS deps + depends on vision_node_core
├── setup.py                 # entry_points declaration
├── manifests/
│   └── my_model.yaml        # I/O spec (follows manifest schema)
└── models/
    └── my_model.py          # BaseModel subclass with @register
```

**Step 1**: Subclass `BaseModel`:
```python
from vision_node_core.models.base import BaseModel
from vision_node_core.registry import ModelRegistry

@ModelRegistry.register("my_model")
class MyModel(BaseModel):
    model_type = "detection"  # detection, depth, segmentation, pose, etc.

    def load(self, device="cuda"):
        # Load weights however your model needs
        ...

    def preprocess(self, image):
        # Use shared utilities or custom
        resized, transform = self._resize_image(image, (640, 640), "letterbox")
        normalized = self._normalize_image(resized)
        return {"pixel_values": self._to_tensor_format(normalized)}

    def postprocess(self, outputs):
        # Decode model-specific outputs into standard ModelOutput
        return ModelOutput(detections=[...])
```

**Step 2**: Declare entry point in `setup.py`:
```python
entry_points={
    'the_vision_node.models': [
        'my_model = my_package.models.my_model:MyModel',
    ],
}
```

**Step 3**: Write manifest YAML (optional but recommended):
```yaml
model:
  name: my_model
  type: detection
input:
  type: image
  preprocessing:
    resize: { method: letterbox, target_size: [640, 640] }
    normalize: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] }
output:
  type: detections
  fields:
    - { name: boxes, dtype: float32, shape: [N, 4] }
    - { name: scores, dtype: float32, shape: [N] }
```

**That's it. No core code touched. `colcon build --packages-select my_new_model_package`, restart the node, model appears.**

### BaseModel Contract

Every model plugin must implement this interface:

```python
class BaseModel(ABC):
    model_type: str  # "detection", "depth", "segmentation", "pose", "tracking", "vlm"

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load model weights to device."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Image → model input tensors."""

    @abstractmethod
    def postprocess(self, outputs: dict[str, np.ndarray]) -> ModelOutput:
        """Raw outputs → structured results."""

    def unload(self) -> None:
        """Free memory. Default: del self._model."""

    # Shared utilities available (don't rewrite):
    # _resize_image(), _normalize_image(), _to_tensor_format()
    # _scale_boxes_to_original(), _nms()
```

### Standard Output Types

Models return a `ModelOutput` with typed fields. This lets the ROS node publish correct message types without knowing which model produced them:

```python
@dataclass
class ModelOutput:
    detections: list[Detection] | None = None      # boxes, scores, classes
    depth_map: np.ndarray | None = None             # (H, W) float
    masks: list[np.ndarray] | None = None           # list of (H, W) bool
    keypoints: np.ndarray | None = None             # (N, K, 2)
    tracks: list[TrackedObject] | None = None       # boxes + persistent IDs
    text: str | None = None                         # captions, VQA answers
    embeddings: np.ndarray | None = None            # feature vectors
```

### Maintenance Story

| Scenario | What Changes |
|----------|-------------|
| New YOLO version (v12) | Add variant to vision_node_yolo, update manifest |
| Entirely new model | New plugin package, 3 files |
| New HW backend (e.g. Qualcomm) | New runner in vision_node_core |
| Model API breaks | Only that plugin package changes |
| PyTorch major upgrade | Each plugin tests independently |
| Someone forks and adds a model | They just make their own package with entry_points |

### Comparison to Other Approaches

| Approach | Files to Add Model | Touch Core? | Third-party Extensible? |
|----------|-------------------|-------------|------------------------|
| **Hardcoded imports** | 1 + edit __init__.py | Yes | No |
| **importlib (current)** | 1 + edit dict in core | Yes | No |
| **Registry + entry_points** | 3 (self-contained) | No | Yes |
| **YAML-only config** | 1 yaml | No | Limited (no custom code) |

---

## Files to Create/Modify

### Restructure from current monolith to plugin packages:

**vision_node_core/** (refactor from current root):
- `registry.py` - NEW: ModelRegistry with decorator + entry_point discovery
- `runners/base.py` - keep as-is
- `runners/cuda.py` - keep as-is
- `runners/rocm.py` - keep as-is
- `runners/openvino.py` - keep as-is
- `loaders/base.py` - keep as-is
- `loaders/preload.py` - keep as-is
- `loaders/dynamic.py` - keep as-is
- `models/base.py` - refactor: enforce contract, add model_type, add load()/unload()

**Per-plugin packages** (new):
- Each gets: `package.xml`, `setup.py` (with entry_points), manifest YAML, model .py file
