# the-vision-node: ROS Vision Package Generator

## Concept Shift

Not a single ROS package. A **generator platform** that produces customized, single ROS packages based on user selections. Think Spring Initializr for ROS vision.

### User Flow

```
1. User visits web app (or CLI)
2. Selects: model architecture (YOLO, SAM, Depth Anything, etc.)
         + backend (CUDA, ROCm, OpenVINO)
         + variant (yolov8s, yolov11m, etc.)
         + optional: custom weights path (fine-tuned model)
3. Generator produces a ready-to-build colcon package (.zip)
4. User extracts, `colcon build`, runs
```

**Fine-tuning support**: Since generated packages target architectures (not specific weights), users with fine-tuned models just point to their own weights. The architecture code is identical.

---

## Decisions Made

1. **Interface**: Web app (Next.js frontend + FastAPI backend). Product potential, analytics, discoverability.
2. **Templates**: Jinja2 - conditional sections for backends, models, configs
3. **Community registry**: Anyone can submit model architecture manifests
4. **Output**: Single ROS package (zip download)
5. **HuggingFace Transformers**: Core dependency for compatible models
6. **Model Format**: PyTorch native
7. **Pipeline Models**: User provides boxes (decoupled)
8. **VLMs**: Separate model category with vLLM server support from day one

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│            Web App (Next.js)                     │
│  Model picker, backend picker, config form      │
│  Preview generated files, download .zip         │
├─────────────────────────────────────────────────┤
│            API (FastAPI)                         │
│  POST /api/generate                             │
│  GET  /api/models  (list available)             │
│  GET  /api/manifests/:model  (model info)       │
│  POST /api/manifests  (community submit)        │
├─────────────────────────────────────────────────┤
│         Generator Engine                         │
│  1. Validate selections (dep conflicts)         │
│  2. Resolve versions (dep matrix)               │
│  3. Render Jinja2 templates                     │
│  4. Pack into zip, stream response              │
├─────────────────────────────────────────────────┤
│         Template Store                           │
│  Jinja2 templates for:                          │
│  - package.xml, setup.py, CMakeLists.txt        │
│  - model.py (per architecture)                  │
│  - runner selection (cuda/rocm/openvino)        │
│  - node.py (ROS entry point)                    │
│  - launch files, config yaml                    │
│  - Dockerfile                                   │
├─────────────────────────────────────────────────┤
│         Manifest Registry                        │
│  YAML files describing each model architecture: │
│  - input/output spec                            │
│  - preprocessing steps                          │
│  - postprocessing logic                         │
│  - dependency versions                          │
│  - VRAM estimates                               │
│  - HF repo / weight source                      │
└─────────────────────────────────────────────────┘
```

---

## What Gets Generated (Output Package)

When a user selects e.g. "YOLOv8 + CUDA + detection":

```
my_vision_package/
├── package.xml                 # ROS deps (generated)
├── setup.py                    # Python deps with pinned versions
├── config/
│   └── params.yaml             # Model config (variant, thresholds, topics)
├── launch/
│   └── vision.launch.py        # ROS launch file
├── my_vision_package/
│   ├── __init__.py
│   ├── node.py                 # ROS node (subscribe image, publish results)
│   ├── model.py                # Model pre/postprocessing (from template)
│   └── runner.py               # Backend-specific inference (CUDA only)
├── manifests/
│   └── yolo.yaml               # Model I/O manifest (copied from registry)
├── Dockerfile                  # Ready-to-build container
├── requirements.txt            # Pinned Python deps
└── README.md                   # Usage instructions (generated)
```

**Key**: This is a **self-contained** package. No dependency on vision_node_core or plugins. Everything needed is generated into the package.

### For fine-tuned models

The generated `params.yaml` includes:

```yaml
model:
  architecture: yolo_v8        # Fixed by generation
  weights: yolov8s.pt          # Default weights
  # User changes this to their fine-tuned weights:
  # weights: /path/to/my_finetuned_yolov8.pt
  custom_classes:              # Optional: override class names
    # - forklift
    # - pallet
    # - person_with_vest
```

Same code, different weights. Works because the architecture (pre/postprocessing, output format) is identical.

---

## Template Structure

```
templates/
├── base/                       # Always included
│   ├── package.xml.j2
│   ├── setup.py.j2
│   ├── __init__.py.j2
│   ├── node.py.j2              # ROS node with subscribe/publish
│   ├── config/
│   │   └── params.yaml.j2
│   ├── launch/
│   │   └── vision.launch.py.j2
│   ├── Dockerfile.j2
│   ├── requirements.txt.j2
│   └── README.md.j2
│
├── runners/                    # Backend-specific (one selected)
│   ├── cuda.py.j2
│   ├── rocm.py.j2
│   └── openvino.py.j2
│
├── models/                     # Model architecture (one selected)
│   ├── yolo.py.j2
│   ├── grounding_dino.py.j2
│   ├── depth_anything.py.j2
│   ├── segment_anything.py.j2
│   ├── zoedepth.py.j2
│   ├── rtmpose.py.j2
│   └── vlm.py.j2              # VLM base (Qwen, Florence, PaliGemma)
│
├── vlm/                        # VLM-specific extras (if VLM selected)
│   ├── vllm_client.py.j2
│   └── output_parser.py.j2
│
└── tracking/                   # If tracking selected
    └── bytetrack.py.j2
```

### Jinja2 Conditionals Example

```jinja2
{# node.py.j2 #}
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

{% if model_type == "vlm" %}
from concurrent.futures import ThreadPoolExecutor
{% endif %}

from .model import {{ model_class }}
from .runner import {{ runner_class }}

class VisionNode(Node):
    def __init__(self):
        super().__init__('{{ package_name }}_node')
        self.bridge = CvBridge()
        self.model = {{ model_class }}()
        self.runner = {{ runner_class }}(device_id={{ device_id }})

{% if model_type == "vlm" %}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.busy = False
        self.latest_image = None
{% endif %}

        self.image_sub = self.create_subscription(
            Image, '{{ input_topic }}', self.image_callback, 10)

{% if model_type == "detection" %}
        # Detection output
        self.det_pub = self.create_publisher(...)
{% elif model_type == "depth" %}
        # Depth output
        self.depth_pub = self.create_publisher(...)
{% elif model_type == "vlm" %}
        # VLM text output
        self.text_pub = self.create_publisher(...)
{% endif %}

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
{% if model_type == "vlm" %}
        self.latest_image = image
        if not self.busy:
            self.busy = True
            self.executor.submit(self._run_vlm)
{% else %}
        inputs = self.model.preprocess(image)
        outputs = self.runner.infer(inputs)
        result = self.model.postprocess(outputs)
        self._publish(result)
{% endif %}
```

---

## Manifest Registry (Community Contributions)

### Manifest Schema

Each model architecture is described by a YAML manifest:

```yaml
# manifests/yolo.yaml
meta:
  name: yolo
  display_name: YOLO (Ultralytics)
  description: Real-time object detection
  category: detection          # detection, depth, segmentation, pose, vlm, tracking
  author: ultralytics
  url: https://github.com/ultralytics/ultralytics
  license: AGPL-3.0

variants:
  - id: yolo_v8n
    display_name: YOLOv8 Nano
    hf_repo: null              # Uses ultralytics API
    weights_url: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
    vram_mb: 200
  - id: yolo_v8s
    display_name: YOLOv8 Small
    vram_mb: 400
  # ... more variants

dependencies:
  python: ">=3.10"
  torch: ">=2.0"
  extra:
    - ultralytics>=8.2.70

input:
  type: image
  format: rgb
  preprocessing:
    resize: { method: letterbox, target_size: [640, 640] }
    normalize: { scale: 255.0 }

output:
  type: detections
  format: xyxy
  fields:
    - { name: boxes, dtype: float32, shape: [N, 4] }
    - { name: scores, dtype: float32, shape: [N] }
    - { name: class_ids, dtype: int32, shape: [N] }

postprocessing:
  nms: { iou_threshold: 0.45, score_threshold: 0.25 }

ros:
  publishers:
    - { topic: detections, msg_type: Detection2DArray }
  parameters:
    - { name: confidence_threshold, type: float, default: 0.25 }
    - { name: classes, type: "list[str]", default: [] }

# Fine-tuning support
fine_tuning:
  supported: true
  weights_format: ".pt"
  custom_classes: true         # Can override class names
  notes: "Train with `yolo train data=custom.yaml model=yolov8s.pt`"
```

### Community Contribution Flow

```
1. Contributor creates a manifest YAML following the schema
2. Optionally includes a model.py.j2 template for custom pre/postprocessing
3. Submits PR to manifest registry (GitHub repo)
4. Maintainers review (schema validation + basic testing)
5. Merged → appears on web app as available model
```

### If a model uses standard patterns, the manifest alone is enough:

- Standard preprocessing (resize + normalize) → generated from manifest fields
- Standard postprocessing (NMS for detection, resize for depth) → generated from output.type
- Only models with unusual logic need a custom `model.py.j2` template

---

## Dependency Version Matrix

### Resolution Engine

The generator resolves dependency versions based on selections:

```python
# dependency_matrix.py
BACKENDS = {
    "cuda": {
        "torch_index": "https://download.pytorch.org/whl/cu124",
        "extra_deps": [],
    },
    "rocm": {
        "torch_index": "https://download.pytorch.org/whl/rocm6.2",
        "extra_deps": [],
    },
    "openvino": {
        "torch_index": None,  # CPU torch
        "extra_deps": ["openvino>=2024.0"],
    },
}

MODEL_DEPS = {
    "yolo": {"ultralytics>=8.2.70": None},
    "grounding_dino": {"transformers>=4.40,<5.0": None},
    "depth_anything": {"transformers>=4.40,<5.0": None, "timm>=0.6.7": None},
    "segment_anything": {"segment-anything-2": None},
    "zoedepth": {"transformers>=4.40,<5.0": None, "timm>=0.6.7": None},
    "rtmpose": {"rtmlib": None, "onnxruntime": None},
    "bytetrack": {"scipy": None, "lap==0.4.0": None},
    "vlm": {"transformers>=4.40,<5.0": None, "openai": None},  # openai for vLLM client
}

BASE_DEPS = {
    "torch>=2.5.1": None,
    "torchvision>=0.20.1": None,
    "numpy>=1.21.6,<2.0": None,
    "opencv-python": None,
    "pyyaml": None,
}

CONFLICTS = [
    # (dep_a, dep_b, resolution)
    ("transformers>=5.0", "torch<2.6", "Pin transformers<5.0"),
]
```

### Validation at generation time

Web form disables incompatible combinations:
- OpenVINO backend + SAM2 → warning (SAM2 needs torch>=2.5.1, OpenVINO runner is CPU)
- Multiple models → show combined VRAM estimate
- VLM selected → show vLLM server recommendation

---

## VLM Handling

When user selects a VLM model:

1. **Extra files generated**: `vllm_client.py`, `output_parser.py`
2. **Node uses async pattern**: ThreadPoolExecutor, non-blocking inference
3. **Config includes vLLM settings**:

```yaml
vlm:
  backend: auto          # auto, vllm, local
  vllm_url: http://localhost:8000
  model: Qwen/Qwen2.5-VL-3B-Instruct
  max_new_tokens: 256
  default_prompt: "Describe what you see"
```

4. **Dockerfile includes vLLM** as optional service:

```dockerfile
# Generated Dockerfile includes:
# Option to run vLLM alongside or separately
```

---

## Tech Stack

### Web App
- **Frontend**: Next.js (TypeScript, React)
  - Model picker with search/filter
  - Backend selector with compatibility indicators
  - Live preview of generated files
  - Zip download
  - Analytics (Vercel Analytics or PostHog)
- **Backend**: FastAPI (Python)
  - `/api/generate` - POST, returns zip stream
  - `/api/models` - GET, list available models from registry
  - `/api/validate` - POST, check compatibility before generating
- **Template Engine**: Jinja2
- **Deployment**: Vercel (frontend) + Railway/Fly.io (FastAPI backend)

### Manifest Registry
- **GitHub repo**: `the-vision-node/manifest-registry`
- **Schema validation**: JSON Schema or Pydantic
- **CI**: Auto-validate manifest PRs, run template generation test

---

## Project Structure

```
the-vision-node/
├── specs/                      # Research & specs (current)
│   ├── basics.md
│   ├── supported_models.md
│   └── research.md
│
├── generator/                  # FastAPI backend + Jinja2 engine
│   ├── main.py                # FastAPI app
│   ├── engine.py              # Template rendering + zip packing
│   ├── deps.py                # Dependency resolution matrix
│   ├── validate.py            # Compatibility validation
│   ├── templates/             # Jinja2 templates (from above)
│   │   ├── base/
│   │   ├── runners/
│   │   ├── models/
│   │   ├── vlm/
│   │   └── tracking/
│   ├── tests/
│   │   └── test_generate.py   # Generate each combo, verify valid Python
│   └── requirements.txt
│
├── web/                        # Next.js frontend
│   ├── app/
│   │   ├── page.tsx           # Landing / generator form
│   │   ├── api/               # API routes (proxy to FastAPI or edge)
│   │   └── models/            # Model browser page
│   ├── components/
│   │   ├── ModelPicker.tsx
│   │   ├── BackendSelector.tsx
│   │   ├── ConfigForm.tsx
│   │   └── FilePreview.tsx
│   └── package.json
│
├── registry/                   # Manifest registry
│   ├── schema.json            # Manifest JSON Schema
│   ├── manifests/             # The manifests (yolo.yaml, sam.yaml, etc.)
│   └── validate.py            # Schema validation script
│
└── examples/                   # Pre-generated example packages
    ├── yolo_cuda/
    ├── depth_anything_cuda/
    └── qwen_vl_vllm/
```

---

## Implementation Order

1. **Manifest registry** - Define schema, write manifests for all models
2. **Generator engine** - Jinja2 templates + zip generation (testable standalone)
3. **FastAPI backend** - API endpoints wrapping the engine
4. **Next.js frontend** - Web form + file preview + download
5. **Community features** - Manifest submission, validation CI
6. **Analytics + product** - Usage tracking, popular models dashboard

---

## What This Changes From Previous Plan

| Before | Now |
|--------|-----|
| Multiple colcon packages (core + plugins) | Single generated package per user |
| Users install plugins manually | Users generate exactly what they need |
| Registry + entry_points for discovery | Manifest registry drives code generation |
| Complex local setup | Download zip, extract, `colcon build` |
| One-size-fits-all | Customized per robot/use-case |
| Pure ROS package | Platform: web app + generator + registry |