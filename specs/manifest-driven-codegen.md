# Plan: Real Python Model Files (No Jinja2)

## Goal
Model files become plain `.py` files — lintable, testable, no Jinja2. The generator just copies them. Only ROS wiring (node, runner, package metadata) stays as templates.

## Research: How Other Projects Handle This

We looked at ESPHome, MMDetection, HuggingFace Transformers, Home Assistant, Cookiecutter, and TinyGSM. The pattern that wins across all of them: **real Python files for model logic + declarative metadata for discovery/wiring**. Not code generation of model logic from YAML/templates.

- **MMDetection**: `@MODELS.register_module()` decorator on real Python classes, referenced by name in config
- **HuggingFace**: `AutoModel.register()` on real Python classes, config is JSON metadata
- **ESPHome**: YAML config → generated code, but component implementations are real code
- **TinyGSM**: Deep inheritance was fragile and hard to extend — cautionary tale

## Key Design Decisions

1. **Copy base classes into each package** (not a shared pip library) — keeps packages self-contained
2. **Standardize model export**: every model file ends with `Model = ConcreteClassName` — runners always do `from .model import Model`
3. **Variant becomes a ROS parameter** (default set at generation time) — model reads it at init via `__init__(self, node, variant)`
4. **Only HuggingFace gets a base class** — YOLO stays standalone (it's already simple)
5. **Backward-compatible**: engine checks for `.py` first, falls back to `.py.j2`

## New/Modified Files

### NEW: `model_templates/base_model.py` (real Python)
```python
class HuggingFaceModel:
    VARIANT_MAP = {}  # subclass overrides

    def __init__(self, node, variant, processor_cls, model_cls):
        self.node = node
        self.logger = node.get_logger()
        self.variant = variant
        self._processor_cls = processor_cls
        self._model_cls = model_cls

    def load(self, device):
        repo = self.VARIANT_MAP[self.variant]
        self.processor = self._processor_cls.from_pretrained(repo)
        self.model = self._model_cls.from_pretrained(repo).to(device)
        self.model.eval()
        self.device = device

    def preprocess(self, image, **kwargs):
        import cv2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, **kwargs, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, inputs):
        return self.model(**inputs)

    def postprocess(self, outputs, original_size=None):
        raise NotImplementedError
```

### NEW: `model_templates/models/yolo.py` (replaces `yolo.py.j2`)
- Standalone class `YoloModel` — no base class needed
- `__init__(self, node, variant)` — reads params, stores variant
- `load(self, device)` — `WEIGHTS_MAP[self.variant]` → `YOLO(weights).to(device)`
- `predict()`, `postprocess()` — same logic as current template
- Bottom: `Model = YoloModel`

### NEW: `model_templates/models/grounding_dino.py` (replaces `.py.j2`)
- `class GroundingDinoModel(HuggingFaceModel)` with `VARIANT_MAP`
- `__init__` passes `AutoProcessor`, `AutoModelForZeroShotObjectDetection` to super
- Overrides `preprocess` (text handling) and `postprocess`
- Bottom: `Model = GroundingDinoModel`

### NEW: `model_templates/models/depth_anything.py` (replaces `.py.j2`)
- `class DepthAnythingModel(HuggingFaceModel)` with `VARIANT_MAP`
- Overrides `postprocess` (depth normalization + colormap)
- Bottom: `Model = DepthAnythingModel`

### MODIFY: `generator/engine.py` — `_render_all()`
```python
# Copy base_model.py into package (always, harmless if unused)
base_model_src = self.model_templates_dir / "base_model.py"
if base_model_src.exists():
    files[f"{pkg}/base_model.py"] = base_model_src.read_text()

# Model: prefer .py (real Python), fall back to .py.j2, then placeholder
model_py = self.model_templates_dir / "models" / f"{model}.py"
if model_py.exists():
    files[f"{pkg}/model.py"] = model_py.read_text()
else:
    # legacy .j2 template path
    ...
```

### MODIFY: `generator/templates/runners/cuda.py.j2` (+ rocm, openvino)
```python
# Before:
from .model import {{ model_class }}
self.model = {{ model_class }}(node)

# After:
from .model import Model
variant = node.get_parameter('variant').value
self.model = Model(node, variant)
```
Keep `{{ model_family }}` and `{{ has_text_input }}` for `infer()` branching — those stay as Jinja2.

### MODIFY: `generator/templates/base/node.py.j2`
- Remove: `from .model import {{ model_class }}` and `self.model = {{ model_class }}(self)`
- Add: `self.declare_parameter('variant', '{{ variant }}')` to parameter declarations
- The runner already creates the model — node doesn't need to

### DELETE (after migration):
- `model_templates/models/yolo.py.j2`
- `model_templates/models/grounding_dino.py.j2`
- `model_templates/models/depth_anything.py.j2`

## Adding a New Model (After This Change)

To add a new HuggingFace model, a contributor creates:

1. **`manifests/new_model.yaml`** — declares variants, ROS params, publishers, output type
2. **`model_templates/models/new_model.py`** — plain Python file:
   ```python
   from transformers import SomeProcessor, SomeAutoModel
   from .base_model import HuggingFaceModel

   class NewModel(HuggingFaceModel):
       VARIANT_MAP = {"variant_a": "org/model-a", "variant_b": "org/model-b"}

       def __init__(self, node, variant):
           super().__init__(node, variant, SomeProcessor, SomeAutoModel)

       def postprocess(self, outputs, original_size=None):
           # model-specific extraction logic
           return {...}

   Model = NewModel
   ```

No Jinja2 knowledge required. The file is testable in isolation.

## Implementation Order

1. Create `base_model.py` (the HuggingFace base class)
2. Convert `depth_anything.py.j2` → `.py` (simplest HF model)
3. Convert `grounding_dino.py.j2` → `.py` (HF with text input)
4. Convert `yolo.py.j2` → `.py` (standalone, no base class)
5. Update `engine.py` to prefer `.py` over `.py.j2` and copy `base_model.py`
6. Update runner templates (`from .model import Model`, pass variant)
7. Update `node.py.j2` (remove model instantiation, add variant param)
8. Delete the old `.py.j2` files

## Verification

1. Generate `e2e_yolo` package → verify `model.py` is valid Python (no `{{ }}`)
2. Run `python3 e2e/run.py --verbose --model yolo --backend cuda --usb-cam --rqt --duration 30`
3. Generate grounding_dino + depth_anything packages → verify valid Python
4. `python -c "import ast; ast.parse(open('model.py').read())"` on each generated model
