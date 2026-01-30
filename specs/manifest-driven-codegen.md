# Plan: Eliminate Per-Model Templates via Manifest-Driven Codegen

## Goal
Replace individual `model_templates/models/{model}.py.j2` files with a **single generic `model.py.j2`** template driven by a new `codegen` section in each manifest YAML. Adding a new model should only require writing a manifest — no new `.py.j2` file.

## Key Insight
The three existing model templates differ in:
1. **Imports** — ultralytics vs various `transformers.Auto*` classes
2. **Variant map** — weight filenames vs HF repo IDs
3. **Load pattern** — `YOLO(weights).to(device)` vs `AutoProcessor + AutoModel.from_pretrained`
4. **Pipeline shape** — ultralytics `predict→postprocess` vs HF `preprocess→forward→postprocess`
5. **Postprocessing logic** — extracting boxes from ultralytics results vs HF processor post-processing vs depth normalization

All of these can be parameterized via manifest fields and a finite set of **postprocess patterns** in the template.

## New Manifest `codegen` Section

Each manifest gets a `codegen` block. Example (grounding_dino):

```yaml
codegen:
  variant_map:
    grounding_dino_tiny: "IDEA-Research/grounding-dino-tiny"
    grounding_dino_base: "IDEA-Research/grounding-dino-base"

  imports:
    - module: cv2
    - module: numpy
      alias: np
    - module: torch
    - module: transformers
      names: [AutoProcessor, AutoModelForZeroShotObjectDetection]

  loading:
    pattern: huggingface            # huggingface | ultralytics
    processor_class: AutoProcessor
    model_class: AutoModelForZeroShotObjectDetection

  pipeline: preprocess_forward      # preprocess_forward | ultralytics_predict
  has_text_input: true
  preprocess_converts_bgr: true

  postprocess:
    pattern: hf_grounded_detection  # finite enum of patterns
```

## Postprocess Patterns (finite set)

| Pattern | Models | What it does |
|---------|--------|-------------|
| `ultralytics_detect` | YOLO | `result.boxes.xyxy/conf/cls` extraction |
| `hf_grounded_detection` | Grounding DINO | `processor.post_process_grounded_object_detection(...)` |
| `hf_depth` | Depth Anything, ZoeDepth | `outputs.predicted_depth` → normalize → optional colormap |

New patterns (for future models not yet implemented) would be added to the same template file — one file to edit, not N.

## Files to Modify

### 1. `generator/manifest.py` — Add Pydantic models for `codegen`
- `ImportSpec(module, names=[], alias="")`
- `LoadingConfig(pattern, processor_class="", model_class="")`
- `PostprocessConfig(pattern)`
- `CodegenConfig(variant_map, imports, loading, pipeline, has_text_input, preprocess_converts_bgr, postprocess, constants={})`
- Add `codegen: CodegenConfig | None = None` to `ModelManifest`

### 2. `generator/context.py` — Pass codegen to template context
- Add `"codegen": manifest.codegen.model_dump() if manifest.codegen else None` to the context dict
- When `codegen` is present, derive `has_text_input` and `model_family` from it instead of heuristics

### 3. `generator/engine.py` — Fallback to generic template
- Change `_render_all` model template resolution:
  - If manifest has `codegen` → render `base/model.py.j2` (the new generic template)
  - Else → try `models/{model}.py.j2` (legacy per-model template)
  - Else → placeholder

### 4. `generator/templates/base/model.py.j2` — NEW generic template
- Renders imports from `codegen.imports`
- Generates `VARIANT_MAP` from `codegen.variant_map`
- Generates `__init__` reading all `ros.parameters`
- Generates `load()` branching on `codegen.loading.pattern`
- Generates `predict()` for ultralytics or `preprocess()/forward()` for HF
- Generates `postprocess()` branching on `codegen.postprocess.pattern`
- Optional `constants` block (e.g. COLORMAPS for depth models)

### 5. `generator/templates/runners/cuda.py.j2` (and rocm, openvino)
- Change branching from `model_family == "ultralytics"` to `codegen.pipeline == "ultralytics_predict"` (when codegen is present, fall back to old logic otherwise)

### 6. Manifests — Add `codegen` to the 3 implemented models
- `manifests/yolo.yaml`
- `manifests/grounding_dino.yaml`
- `manifests/depth_anything.yaml`

### 7. Delete old per-model templates
- `model_templates/models/yolo.py.j2`
- `model_templates/models/grounding_dino.py.j2`
- `model_templates/models/depth_anything.py.j2`

## Migration Strategy

Do it all at once for the 3 existing models (they're small). The generated `model.py` output should be functionally equivalent to what the old per-model templates produced.

## Verification

1. Generate the `e2e_yolo` package before and after, diff the output `model.py` — should be functionally equivalent
2. Run `python3 e2e/run.py --verbose --model yolo --backend cuda --usb-cam --rqt --duration 30` to verify YOLO still works end-to-end
3. Generate grounding_dino and depth_anything packages, verify they produce valid Python