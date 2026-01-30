# Plan: Reduce Model Template Boilerplate via Base Classes

## Goal
Extract shared infrastructure (load, preprocess, forward) into base classes so per-model templates only define `postprocess()` — the truly unique part. Add `codegen` manifest fields to drive the shared parts declaratively.

## What Changes

### 1. New base class templates

Create two base class templates that handle the repetitive parts:

**`generator/templates/base/base_model_hf.py.j2`** — HuggingFace base class:
```python
"""Base class for HuggingFace Transformers models."""
import cv2
import torch

class HuggingFaceModelBase:
    VARIANT_MAP = {}  # Override in subclass

    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()
        # Read all ROS parameters into self.<name>
        {% for param in parameters %}
        self.{{ param.name }} = node.get_parameter('{{ param.name }}').value
        {% endfor %}

    def load(self, device):
        from transformers import {{ codegen.loading.processor_class }}, {{ codegen.loading.model_class }}
        repo = self.VARIANT_MAP.get("{{ variant }}")
        self.processor = {{ codegen.loading.processor_class }}.from_pretrained(repo)
        self.model = {{ codegen.loading.model_class }}.from_pretrained(repo).to(device)
        self.model.eval()
        self.device = device

    def preprocess(self, image, **kwargs):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # text input handled if has_text_input
        inputs = self.processor(images=rgb, **kwargs, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, inputs):
        return self.model(**inputs)
```

**`generator/templates/base/base_model_ultralytics.py.j2`** — Ultralytics base class:
```python
"""Base class for Ultralytics models."""
from ultralytics import YOLO

class UltralyticsModelBase:
    VARIANT_MAP = {}

    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()
        {% for param in parameters %}
        self.{{ param.name }} = node.get_parameter('{{ param.name }}').value
        {% endfor %}

    def load(self, device):
        weights = self.VARIANT_MAP.get("{{ variant }}", "{{ variant }}.pt")
        self.model = YOLO(weights)
        self.model.to(device)

    def predict(self, image):
        return self.model(image, verbose=False, **self._predict_kwargs())
```

### 2. Add `codegen` section to manifests

Minimal — just enough to drive the base class:

```yaml
# In each manifest YAML:
codegen:
  variant_map:
    variant_name: "weight_or_repo_id"
  loading:
    pattern: huggingface        # or: ultralytics
    processor_class: AutoProcessor        # HF only
    model_class: AutoModelForZeroShotObjectDetection  # HF only
```

### 3. Simplify per-model templates

Each model template shrinks to just the unique part — `postprocess()` and any model-specific extras:

**yolo.py.j2** (before: 71 lines, after: ~20 lines):
```python
from .base_model import UltralyticsModelBase

class {{ model_class }}(UltralyticsModelBase):
    VARIANT_MAP = { {{ codegen variant_map entries }} }

    def _predict_kwargs(self):
        return dict(conf=self.confidence_threshold, iou=self.iou_threshold,
                    classes=self.classes if self.classes else None)

    def postprocess(self, results):
        result = results[0]
        # ... extract boxes/scores/class_ids/class_names ...
```

**grounding_dino.py.j2** (before: 77 lines, after: ~30 lines):
```python
from .base_model import HuggingFaceModelBase

class {{ model_class }}(HuggingFaceModelBase):
    VARIANT_MAP = { {{ codegen variant_map entries }} }

    def preprocess(self, image, text=None):
        prompt = text if text is not None else self.default_prompt
        inputs = super().preprocess(image, text=prompt)
        self._last_input_ids = inputs.get("input_ids")  # needed for postprocess
        return inputs

    def postprocess(self, outputs, original_size=None):
        # ... grounding DINO specific post-processing ...
```

**depth_anything.py.j2** (before: 78 lines, after: ~25 lines):
```python
from .base_model import HuggingFaceModelBase

class {{ model_class }}(HuggingFaceModelBase):
    VARIANT_MAP = { {{ codegen variant_map entries }} }

    def postprocess(self, outputs, original_size=None):
        # ... depth normalization + colormap ...
```

### 4. Files to modify

| File | Change |
|------|--------|
| `generator/manifest.py` | Add `CodegenConfig` Pydantic model with `variant_map` and `loading` fields. Add optional `codegen` field to `ModelManifest` |
| `generator/context.py` | Pass `codegen` dict into template context |
| `generator/engine.py` | Render `base_model.py` from the appropriate base class template, add it to the generated package files |
| `generator/templates/base/base_model_hf.py.j2` | **NEW** — HuggingFace base class template |
| `generator/templates/base/base_model_ultralytics.py.j2` | **NEW** — Ultralytics base class template |
| `model_templates/models/yolo.py.j2` | Simplify — inherit from UltralyticsModelBase, only define postprocess + predict kwargs |
| `model_templates/models/grounding_dino.py.j2` | Simplify — inherit from HuggingFaceModelBase, only define postprocess + preprocess override |
| `model_templates/models/depth_anything.py.j2` | Simplify — inherit from HuggingFaceModelBase, only define postprocess |
| `manifests/yolo.yaml` | Add `codegen` section with variant_map + loading |
| `manifests/grounding_dino.yaml` | Add `codegen` section with variant_map + loading |
| `manifests/depth_anything.yaml` | Add `codegen` section with variant_map + loading |
| `generator/templates/runners/cuda.py.j2` | No change needed — runner still calls model.predict/preprocess/forward/postprocess |

### 5. Generated package structure (after)

```
e2e_yolo/
├── e2e_yolo/
│   ├── __init__.py
│   ├── node.py          # unchanged
│   ├── base_model.py    # NEW — generated from base class template
│   ├── model.py         # simplified — just postprocess + overrides
│   └── runner.py        # unchanged
├── ...
```

## Verification

1. Generate `e2e_yolo` package, verify `model.py` imports from `base_model` and inference works
2. Run `python3 e2e/run.py --verbose --model yolo --backend cuda --usb-cam --rqt --duration 30`
3. Generate grounding_dino and depth_anything packages, verify valid Python output
