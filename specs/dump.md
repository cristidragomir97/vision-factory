# the-vision-node — Build Summary

## What We Built

A **ROS2 vision package generator** — like Spring Initializr but for robotics vision nodes. Users pick a model architecture + hardware backend + variant, and get a self-contained colcon package as a zip.

## Architecture

Two-repo design:

- **the-vision-node** — Generator tool (CLI + FastAPI server + core templates)
- **model_templates/** — Per-model Jinja2 templates (future: separate community repo)

## What Was Created

### Generator Engine (Python)

| File | Purpose |
|------|---------|
| `generator/manifest.py` | Pydantic models for parsing model manifest YAMLs (handles varied input schemas) |
| `generator/resolver.py` | Dependency resolution: backend configs (cuda/rocm/openvino), base deps, model-specific deps |
| `generator/context.py` | Builds Jinja2 template context dict from manifest + user selections |
| `generator/engine.py` | Core `Generator` class: load manifests → validate → resolve → render → zip |
| `generator/cli.py` | Click CLI: `generate`, `list-models`, `info` commands |
| `generator/__main__.py` | Entry point for `python -m generator` |

### Core Templates (`generator/templates/`)

| Template | Purpose |
|----------|---------|
| `base/node.py.j2` | ROS2 node — branches on `output_type` (detections/depth_map) and `has_text_input` |
| `base/package.xml.j2` | ROS2 package manifest |
| `base/setup.py.j2` | setuptools config with entry points |
| `base/params.yaml.j2` | ROS2 parameter file |
| `base/vision.launch.py.j2` | Launch file |
| `base/requirements.txt.j2` | pip deps with optional torch index URL |
| `base/Dockerfile.j2` | Docker build for ros:humble |
| `base/README.md.j2` | Generated package docs |
| `base/__init__.py.j2` | Package init |
| `runners/cuda.py.j2` | CUDA runner — branches on `model_family` (ultralytics vs huggingface) |
| `runners/rocm.py.j2` | ROCm runner (same API as CUDA) |
| `runners/openvino.py.j2` | OpenVINO runner (CPU/iGPU) |

### Model Templates (`model_templates/models/`)

| Template | Pattern |
|----------|---------|
| `yolo.py.j2` | Ultralytics API: `model.predict()` → `postprocess()` |
| `depth_anything.py.j2` | HF Transformers: `AutoImageProcessor` + `AutoModelForDepthEstimation` |
| `grounding_dino.py.j2` | HF Transformers: `AutoProcessor` + `AutoModelForZeroShotObjectDetection` + text input |

### Model Manifests (`manifests/`) — 8 YAML files

yolo, depth_anything, grounding_dino, segment_anything, florence, rtmpose, zoedepth, bytetrack

Each defines: model info, source, input schema, output schema, postprocessing, ROS config (publishers/subscribers/parameters), VRAM estimates.

### FastAPI Server (`server/main.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/models` | GET | List all model architectures |
| `/api/models/{name}` | GET | Get single model details |
| `/api/generate` | POST | Generate + stream zip download |
| `/api/preview` | POST | Generate + return file contents as JSON |
| `/api/backends` | GET | List available backends |

CORS enabled for frontend access.

### Frontend (`robotic-venture-hub/src/pages/Generator.tsx`)

React page at `/generator` route integrated into the existing website. Uses shadcn/ui components (Select, Input, Button, Card, Badge, Tabs, ScrollArea) matching the site's dark theme + teal accents.

Two-column layout:
- **Left**: Configuration card (model picker, backend selector, variant picker, package name input, Preview + Download buttons)
- **Right**: File preview card with tabbed code viewer

### Tests — 71 passing

| File | Count | Coverage |
|------|-------|----------|
| `tests/test_manifest.py` | 15 | All 8 manifests parse, field validation, error cases |
| `tests/test_resolver.py` | 21 | Backend configs, model deps, validation, context building |
| `tests/test_engine.py` | 28 | YOLO/DepthAnything/GroundingDINO generation, ast.parse syntax checks, XML validation |
| `tests/test_cli.py` | 7 | list-models, info, generate commands |

## Key Design Decisions

1. **`model_family` drives runner branching** — ultralytics uses `model(image)`, huggingface uses `preprocess → forward → postprocess`
2. **`output_type` drives node branching** — detections → Detection2DArray, depth_map → Image/32FC1
3. **`has_text_input` flag** — Grounding DINO needs a prompt subscriber, detected from manifest input schema
4. **`input: dict[str, Any]`** — Manifests have varied input structures (flat, multi-input, keyed), kept as raw dict
5. **Generated code has zero dependency on generator** — Output zip is fully self-contained ROS2 package

---

## Repo Separation Guide

The project is designed as two repos. Right now everything lives in `the-vision-node/`, but the split is clean:

### Repo 1: `the-vision-node` (The Generator Tool)

What stays here — the stable infrastructure:

```
the-vision-node/
├── pyproject.toml
├── generator/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── engine.py
│   ├── manifest.py
│   ├── resolver.py
│   ├── context.py
│   └── templates/            # CORE templates (always needed)
│       ├── base/
│       │   ├── node.py.j2
│       │   ├── package.xml.j2
│       │   ├── setup.py.j2
│       │   ├── __init__.py.j2
│       │   ├── params.yaml.j2
│       │   ├── vision.launch.py.j2
│       │   ├── requirements.txt.j2
│       │   ├── Dockerfile.j2
│       │   └── README.md.j2
│       └── runners/
│           ├── cuda.py.j2
│           ├── rocm.py.j2
│           └── openvino.py.j2
├── server/
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_manifest.py
│   ├── test_resolver.py
│   ├── test_engine.py
│   └── test_cli.py
└── specs/
```

This repo owns:
- The generator engine (manifest parsing, dep resolution, template rendering, zip packing)
- The CLI (`python -m generator generate --model yolo --backend cuda ...`)
- The FastAPI server (wraps the engine for the web UI)
- Core templates (ROS node scaffolding, runner backends) — these are model-agnostic
- Tests

### Repo 2: `vision-models` (Community Registry)

What gets extracted — community-maintained model definitions:

```
vision-models/
├── manifests/
│   ├── yolo.yaml
│   ├── depth_anything.yaml
│   ├── grounding_dino.yaml
│   ├── segment_anything.yaml
│   ├── florence.yaml
│   ├── rtmpose.yaml
│   ├── zoedepth.yaml
│   └── bytetrack.yaml
├── templates/
│   ├── models/
│   │   ├── yolo.py.j2
│   │   ├── depth_anything.py.j2
│   │   └── grounding_dino.py.j2
├── CONTRIBUTING.md
└── README.md
```

This repo owns:
- Model manifest YAMLs (input/output specs, variants, deps, VRAM estimates)
- Per-model Jinja2 templates (the actual model inference code: preprocessing, forward, postprocessing)
- Contribution guidelines for adding new models

The generator tool clones/fetches this registry at runtime (or uses a local path via `--manifests` and `--model-templates` CLI flags).

### Repo 3: `robotic-venture-hub` (Website)

The only file we touched:

```
src/
├── App.tsx                  # Added /generator route
└── pages/
    └── Generator.tsx        # New page — calls the FastAPI server
```

---

## Tutorials

### Tutorial 1: Generate a Package via CLI

```bash
# Setup
cd the-vision-node
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# List available models
python -m generator list-models

# Output:
#   bytetrack                 tracked_objects      (1 variants)
#   depth_anything            depth_map            (6 variants)
#   florence                  multi_task           (4 variants)
#   grounding_dino            detections           (2 variants)
#   rtmpose                   keypoints            (7 variants)
#   segment_anything          segmentation         (9 variants)
#   yolo                      detections           (9 variants)
#   zoedepth                  depth_map            (3 variants)

# Get info on a specific model
python -m generator info yolo

# Output:
#   Model:    yolo
#   Family:   ultralytics
#   Source:   ultralytics (ultralytics/ultralytics)
#   Output:   detections (xyxy)
#   Default:  yolo_v8s
#   Variants:
#     - yolo_v8n  [200 MB VRAM]
#     - yolo_v8s (default)  [400 MB VRAM]
#     - yolo_v8m  [800 MB VRAM]
#     ...

# Generate a package
python -m generator generate \
  --model yolo \
  --backend cuda \
  --variant yolo_v8s \
  --package-name my_detector \
  --output ./output

# Output: Generated: output/my_detector.zip

# Unzip and inspect
unzip output/my_detector.zip -d output/
ls output/my_detector/
# Dockerfile  README.md  config/  launch/  my_detector/  package.xml  requirements.txt  setup.py
```

The generated `my_detector/` is a complete ROS2 colcon package. No dependency on the generator at runtime.

### Tutorial 2: Generate a Depth Estimation Package

```bash
python -m generator generate \
  --model depth_anything \
  --backend cuda \
  --variant depth_anything_v2_vitb \
  --package-name my_depth

# Or use the default variant (depth_anything_v2_vitb):
python -m generator generate \
  --model depth_anything \
  --backend cuda \
  --package-name my_depth
```

The generated node publishes:
- `depth` — raw 32FC1 depth image
- `depth_colored` — colorized RGB visualization

### Tutorial 3: Generate an Open-Vocab Detection Package

```bash
python -m generator generate \
  --model grounding_dino \
  --backend cuda \
  --variant grounding_dino_base \
  --package-name my_dino
```

This generates a node that:
- Subscribes to `/camera/image_raw` (Image)
- Subscribes to `prompt` (String) — for dynamic text prompts
- Publishes `detections` (Detection2DArray)

Change what to detect at runtime:
```bash
ros2 topic pub /prompt std_msgs/String "data: 'fire hydrant . stop sign'"
```

### Tutorial 4: Use a Different Backend

```bash
# AMD GPU (ROCm)
python -m generator generate \
  --model yolo --backend rocm --package-name yolo_amd

# Intel CPU/iGPU (OpenVINO)
python -m generator generate \
  --model yolo --backend openvino --package-name yolo_intel
```

The only difference in the generated code is `runner.py`:
- `cuda` → `torch.device('cuda:0')`, pip index `cu124`
- `rocm` → `torch.device('cuda:0')` (ROCm maps to cuda API), pip index `rocm6.2`
- `openvino` → no torch device, adds `openvino>=2024.0` dep

### Tutorial 5: Run the Generated Package in ROS2

```bash
# After generating and unzipping my_detector:
cd ~/ros_ws/src
cp -r /path/to/output/my_detector .

# Install Python deps
pip install -r my_detector/requirements.txt

# Build
cd ~/ros_ws
colcon build --packages-select my_detector
source install/setup.bash

# Run
ros2 launch my_detector vision.launch.py

# Or run the node directly:
ros2 run my_detector my_detector_node
```

Configure via `config/params.yaml`:
```yaml
my_detector_node:
  ros__parameters:
    input_topic: "/camera/image_raw"
    confidence_threshold: 0.25
    iou_threshold: 0.45
    classes: []  # empty = all classes
```

### Tutorial 6: Run with Docker

```bash
cd output/my_detector
docker build -t my_detector .
docker run --gpus all --rm my_detector
```

The generated `Dockerfile` uses `ros:humble-ros-base`, installs pip deps, builds with colcon, and launches the node.

### Tutorial 7: Run the Web UI

```bash
# Terminal 1: Start the generator API
cd the-vision-node
source .venv/bin/activate
pip install -e ".[server]"
uvicorn server.main:app --reload --port 8000

# Terminal 2: Start the website
cd robotic-venture-hub
npm install
npm run dev

# Open http://localhost:8081/generator
```

The web UI:
1. Pick a model from the dropdown (shows output type)
2. Pick a hardware backend (CUDA / ROCm / OpenVINO)
3. Pick a variant (shows VRAM estimate)
4. Type a package name (snake_case)
5. Click **Preview** to see all generated files in a tabbed code viewer
6. Click **Download** to get the zip

### Tutorial 8: Use the API Directly

```bash
# List models
curl http://localhost:8000/api/models | python -m json.tool

# List backends
curl http://localhost:8000/api/backends

# Preview generated files (returns JSON)
curl -X POST http://localhost:8000/api/preview \
  -H "Content-Type: application/json" \
  -d '{"model":"yolo","backend":"cuda","variant":"yolo_v8s","package_name":"my_pkg"}'

# Download zip
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"yolo","backend":"cuda","variant":"yolo_v8s","package_name":"my_pkg"}' \
  -o my_pkg.zip
```

Swagger docs at: `http://localhost:8000/docs`

### Tutorial 9: Run Tests

```bash
cd the-vision-node
source .venv/bin/activate
pip install -e ".[dev]"

# Run all 71 tests
pytest

# Run specific test files
pytest tests/test_manifest.py     # 15 tests — manifest parsing
pytest tests/test_resolver.py     # 21 tests — dependency resolution + context building
pytest tests/test_engine.py       # 28 tests — end-to-end generation + validation
pytest tests/test_cli.py          # 7 tests — CLI commands

# Run with verbose output
pytest -v
```

### Tutorial 10: Add a New Model (Contributing to vision-models)

To add a new model (e.g. RT-DETR), you need two files:

**1. Manifest (`manifests/rt_detr.yaml`):**

```yaml
model:
  name: rt_detr
  family: ultralytics          # determines runner branching
  variants:
    - rt_detr_l
    - rt_detr_x
  default_variant: rt_detr_l

source:
  type: ultralytics
  repo: ultralytics/ultralytics

input:
  type: image
  format: rgb
  preprocessing:
    resize:
      method: letterbox
      target_size: [640, 640]

output:
  type: detections              # determines node publisher logic
  format: xyxy
  fields:
    - name: boxes
      dtype: float32
      shape: [N, 4]
    - name: scores
      dtype: float32
      shape: [N]
    - name: class_ids
      dtype: int32
      shape: [N]
    - name: class_names
      dtype: string
      shape: [N]

ros:
  publishers:
    - topic: detections
      msg_type: DetectionArray
      frame_id: camera_frame

  parameters:
    - name: confidence_threshold
      type: float
      default: 0.25
    - name: iou_threshold
      type: float
      default: 0.45

resources:
  vram:
    rt_detr_l: 1500
    rt_detr_x: 3000
```

**2. Model template (`model_templates/models/rt_detr.py.j2`):**

```python
"""RT-DETR model via Ultralytics API."""

from ultralytics import RTDETR


class {{ model_class }}:
    WEIGHTS_MAP = {
        "rt_detr_l": "rtdetr-l.pt",
        "rt_detr_x": "rtdetr-x.pt",
    }

    def __init__(self, node):
        self.node = node
        self.confidence = node.get_parameter('confidence_threshold').value
        self.iou = node.get_parameter('iou_threshold').value

    def load(self, device):
        weights = self.WEIGHTS_MAP.get("{{ variant }}", "{{ variant }}.pt")
        self.model = RTDETR(weights)
        self.model.to(device)

    def predict(self, image):
        return self.model(image, conf=self.confidence, iou=self.iou, verbose=False)

    def postprocess(self, results):
        result = results[0]
        return {
            'boxes': result.boxes.xyxy.cpu().numpy(),
            'scores': result.boxes.conf.cpu().numpy(),
            'class_ids': result.boxes.cls.cpu().numpy().astype(int),
            'class_names': [result.names[int(c)] for c in result.boxes.cls.cpu().numpy()],
        }
```

Then it just works:
```bash
python -m generator generate --model rt_detr --backend cuda --package-name my_rtdetr
```

**Key rules for new models:**
- `model.family` must match an entry in `context.py:MODEL_CLASS_MAP` (or add one)
- `source.type` must match `context.py:FAMILY_MAP` (ultralytics/huggingface/github/mmpose)
- `output.type` drives the node's `_publish()` method (detections/depth_map are built-in, others get a TODO stub)
- Model deps must be added to `resolver.py:MODEL_DEPS`
- Template must expose: `__init__(node)`, `load(device)`, and either `predict(image)` (ultralytics) or `preprocess()`/`forward()`/`postprocess()` (huggingface)
