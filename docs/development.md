# Development

## Adding a New Model

1. Create `models/model_name/manifest.yaml` defining the model's I/O schema, variants, and ROS config
2. Create `models/model_name/model.py` with the model loading/inference logic
3. Add an entry to `MODEL_DEPS` in `generator/resolver.py` with the model's pip dependencies
4. Run `pytest` to validate

## Installation

```bash
pip install -e .

# For development (includes pytest)
pip install -e ".[dev]"
```

## Testing

```bash
pytest
pytest -v                      # verbose
pytest tests/test_engine.py    # specific file
```
