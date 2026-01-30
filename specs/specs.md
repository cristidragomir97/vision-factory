# vision-factory -- Handoff Specs

This document is for a Claude instance running on a CUDA-enabled system.
It describes how to set up, run, test, and extend vision-factory.

## What This Is

vision-factory is a CLI tool that generates complete, buildable ROS2 vision packages from YAML manifests. You give it a model name (e.g. `yolo`) and a backend (e.g. `cuda`), and it produces a zip containing a full ROS2 package: node, launch file, Dockerfile, params, requirements, etc.

The generated packages are standalone -- they don't depend on vision-factory at runtime.

## Directory Layout

```
vision-factory/
├── pyproject.toml                  # Package config. Name: vision-factory. CLI: vision-gen
├── README.md                       # Project README with Mermaid diagrams
│
├── generator/                      # Core Python package
│   ├── __init__.py
│   ├── __main__.py                 # python -m generator
│   ├── cli.py                      # Click CLI (vision-gen generate/list-models/info)
│   ├── engine.py                   # Generator class -- orchestrates the 5-stage pipeline
│   ├── manifest.py                 # Pydantic models for YAML manifest parsing
│   ├── resolver.py                 # Dependency resolution (pip deps, ROS deps per backend)
│   ├── context.py                  # Builds Jinja2 template context dict
│   └── templates/
│       ├── base/                   # 9 core templates (package.xml, setup.py, node.py, etc.)
│       │   ├── __init__.py.j2
│       │   ├── Dockerfile.j2
│       │   ├── README.md.j2
│       │   ├── node.py.j2
│       │   ├── package.xml.j2
│       │   ├── params.yaml.j2
│       │   ├── requirements.txt.j2
│       │   ├── setup.py.j2
│       │   └── vision.launch.py.j2
│       └── runners/                # Backend-specific inference runners
│           ├── cuda.py.j2
│           ├── rocm.py.j2
│           └── openvino.py.j2
│
├── model_templates/                # Per-model Jinja2 templates
│   └── models/
│       ├── yolo.py.j2
│       ├── depth_anything.py.j2
│       └── grounding_dino.py.j2
│
├── manifests/                      # YAML model definitions (8 models)
│   ├── yolo.yaml
│   ├── depth_anything.yaml
│   ├── grounding_dino.yaml
│   ├── segment_anything.yaml
│   ├── florence.yaml
│   ├── rtmpose.yaml
│   ├── zoedepth.yaml
│   └── bytetrack.yaml
│
├── tests/                          # Unit tests (pytest)
│   ├── __init__.py
│   ├── conftest.py
│   └── test_manifest.py
│
├── e2e/                            # End-to-end Docker-based tests
│   ├── run.py                      # Host-side orchestrator
│   ├── test_harness.py             # Runs inside the Docker container
│   ├── Dockerfile.e2e.j2           # Jinja2 Dockerfile template
│   └── README.md                   # E2E usage docs
│
└── specs/                          # Design docs and specs
    ├── basics.md
    ├── supported_models.md
    ├── research.md
    ├── structure.md
    ├── generator.md
    ├── dump.md
    └── specs.md                    # This file
```

## Setup

```bash
cd vision-factory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Dependencies: jinja2, pyyaml, pydantic, click. Dev extras: pytest, pytest-tmp-files, httpx.

## Running the Generator CLI

```bash
# List available models
vision-gen list-models

# Get info about a specific model
vision-gen info yolo

# Generate a ROS2 package
vision-gen generate \
    --model yolo \
    --backend cuda \
    --variant yolo_v8s \
    --package-name my_yolo_pkg \
    --output-dir ./output
```

This produces a zip at `./output/my_yolo_pkg.zip` containing a complete ROS2 package.

## Running Unit Tests

```bash
pytest
```

Note: after a recent directory rename, only `test_manifest.py` is present. The other test files (`test_engine.py`, `test_resolver.py`, `test_cli.py`) may have been lost in the move. Check if they need to be restored from git history.

## Running E2E Tests

The e2e tests generate a package, build it in Docker, and run full inference. **This is the main thing you're here to test.**

### Prerequisites

- Docker
- NVIDIA GPU with `nvidia-smi` working
- A rosbag containing camera images (topic publishing `sensor_msgs/Image`)

### Basic run (rosbag mode)

```bash
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --rosbag /path/to/your/camera_bag \
    --ros-distro jazzy
```

This will:
1. Call `Generator.generate()` to produce a ROS2 package zip
2. Render `Dockerfile.e2e.j2` and build a Docker image
3. Run the container with `--gpus all`
4. Inside the container: launch the vision node, play the rosbag on loop, count output messages
5. Produce a markdown report at `./e2e-output/e2e-report.md`

Exit code 0 = all stages passed and messages were received on the output topic.

### USB camera mode

```bash
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --usb-cam
```

Passes `/dev/video0` into the container. Expects a webcam at that path.

### With rqt visualization (requires X11)

```bash
xhost +local:docker
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --usb-cam \
    --rqt \
    --duration 60
xhost -local:docker
```

### E2E arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--model` | yes | - | Model architecture (e.g. `yolo`) |
| `--backend` | yes | - | Backend (`cuda`, `rocm`, `openvino`) |
| `--rosbag` | one of | - | Path to rosbag directory |
| `--usb-cam` | one of | - | Use webcam as input |
| `--rqt` | no | off | Open rqt_image_view (needs X11) |
| `--variant` | no | manifest default | Model variant |
| `--package-name` | no | `e2e_<model>` | Generated package name |
| `--ros-distro` | no | `jazzy` | ROS2 distribution |
| `--duration` | no | `30.0` | Test duration in seconds |
| `--output-dir` | no | `./e2e-output` | Report output directory |

### E2E internals

The e2e system has three parts:

1. **`run.py`** (host) -- Orchestrates generation, Docker build, Docker run. Parses the JSON report from stdout. Writes markdown report. Uses `StageResult` objects for per-stage pass/fail/timing.

2. **`Dockerfile.e2e.j2`** (Jinja2 template) -- Renders a Dockerfile based on `ros_distro`, `input_source` (rosbag/usb_cam), and `enable_rqt`. Conditionally installs rosbag2 packages, usb_cam, or rqt. Copies the generated ROS2 package, builds with colcon, copies the test harness.

3. **`test_harness.py`** (runs inside container) -- Launches the vision node via `ros2 run`, waits for it with `ros2 node list`, starts the input source (rosbag play or usb_cam), optionally launches rqt, then counts messages on the output topic. Emits JSON between `---E2E_REPORT_START---` / `---E2E_REPORT_END---` markers.

## Architecture: The Generation Pipeline

The generator runs a 5-stage pipeline:

1. **Load manifests** -- Parse YAML files from `manifests/` into Pydantic `ModelManifest` objects
2. **Validate** -- Check model + backend combo is valid (via `resolver.validate_selection()`)
3. **Resolve dependencies** -- Compute pip deps, ROS deps, base image per backend (via `resolver.resolve_dependencies()`)
4. **Build context** -- Assemble the Jinja2 template context dict (via `context.build_context()`)
5. **Render templates** -- Render all templates, write files into a zip

Key classes/functions:
- `generator/engine.py`: `Generator` class, `Generator.generate()`, `GenerationError`
- `generator/manifest.py`: `ModelManifest`, `ModelInfo`, `SourceInfo`, `OutputConfig`, `RosConfig`
- `generator/resolver.py`: `BACKENDS` dict, `MODEL_DEPS`, `resolve_dependencies()`, `validate_selection()`
- `generator/context.py`: `build_context()`, `MODEL_CLASS_MAP`, `RUNNER_CLASS_MAP`, `FAMILY_MAP`
- `generator/cli.py`: Click CLI with `generate`, `list-models`, `info` commands

### Template layers

Templates are organized in three layers:
1. **Base templates** (`generator/templates/base/`) -- Package structure, node skeleton, Dockerfile, launch file
2. **Runner templates** (`generator/templates/runners/`) -- Backend-specific inference code (cuda.py.j2, rocm.py.j2, openvino.py.j2)
3. **Model templates** (`model_templates/models/`) -- Per-model inference logic (yolo.py.j2, depth_anything.py.j2, etc.)

## Supported Models and Backends

| Model | Backends | Default Variant |
|-------|----------|-----------------|
| yolo | cuda, rocm, openvino | Check manifest |
| depth_anything | cuda, rocm, openvino | Check manifest |
| grounding_dino | cuda | Check manifest |
| segment_anything | cuda | Check manifest |
| florence | cuda | Check manifest |
| rtmpose | cuda, openvino | Check manifest |
| zoedepth | cuda | Check manifest |
| bytetrack | cuda | Check manifest |

To see exact variants and defaults: `vision-gen info <model>` or read `manifests/<model>.yaml`.

## How to Extend

### Adding a new model

1. Create `manifests/<model_name>.yaml` following the schema in existing manifests
2. Create `model_templates/models/<model_name>.py.j2` with the inference logic
3. Add entries to `generator/resolver.py`: `MODEL_DEPS` dict
4. Add entries to `generator/context.py`: `MODEL_CLASS_MAP`, `FAMILY_MAP`
5. Run `pytest` to validate
6. Test with e2e: `python e2e/run.py --model <model_name> --backend cuda --rosbag /path/to/bag`

### Adding a new backend

1. Create `generator/templates/runners/<backend>.py.j2`
2. Add to `generator/resolver.py`: `BACKENDS` dict (base image, pip suffix, etc.)
3. Update manifests that support the backend

## Known Issues

1. **Missing test files** -- After renaming `robot-vision-tool/` to `vision-factory/`, only `test_manifest.py` survived in `tests/`. The tests `test_engine.py`, `test_resolver.py`, `test_cli.py` may need to be restored from git history. Run `pytest` to see what's currently passing.

2. **Only 3 model templates exist** -- `model_templates/models/` has templates for yolo, depth_anything, and grounding_dino only. The other 5 models (segment_anything, florence, rtmpose, zoedepth, bytetrack) have manifests but no model templates yet.

3. **Sibling repo: robot-vision-api** -- There's a FastAPI server at `../robot-vision-api/` that wraps the generator. Its `main.py` has a hardcoded path issue: `ROOT = Path(__file__).parent.parent` resolves incorrectly after the repo split. Not relevant to e2e testing but worth knowing about.

## Quick Start for E2E Testing on CUDA

```bash
# 1. Setup
cd vision-factory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Verify GPU
nvidia-smi

# 3. Run unit tests first
pytest

# 4. Run e2e with YOLO + CUDA
python e2e/run.py \
    --model yolo \
    --backend cuda \
    --rosbag /path/to/your/camera_bag \
    --ros-distro jazzy \
    --duration 30

# 5. Check the report
cat e2e-output/e2e-report.md
```

If the e2e test fails, the report will show which stage failed (generation, Docker build, or inference) with logs in collapsible sections.
