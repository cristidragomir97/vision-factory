"""Base class for HuggingFace Transformers vision models."""

import cv2


class HuggingFaceModel:
    """Common load/preprocess/forward pattern for HF models.

    Subclasses override VARIANT_MAP and postprocess().
    """

    VARIANT_MAP = {}  # subclass overrides: variant_id -> HF repo

    def __init__(self, node, variant, processor_cls, model_cls):
        self.node = node
        self.logger = node.get_logger()
        self.variant = variant
        self._processor_cls = processor_cls
        self._model_cls = model_cls

    def load(self, device):
        repo = self.VARIANT_MAP.get(self.variant)
        if repo is None:
            raise ValueError(
                f"Unknown variant '{self.variant}'. "
                f"Available: {list(self.VARIANT_MAP.keys())}"
            )
        self.processor = self._processor_cls.from_pretrained(repo)
        self.model = self._model_cls.from_pretrained(repo).to(device)
        self.model.eval()
        self.device = device

    def preprocess(self, image, **kwargs):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, **kwargs, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, inputs):
        return self.model(**inputs)

    def postprocess(self, outputs, original_size=None):
        raise NotImplementedError
