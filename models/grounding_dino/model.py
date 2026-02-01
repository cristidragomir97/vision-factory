"""Grounding DINO model via HuggingFace Transformers."""

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from .base_model import HuggingFaceModel


class GroundingDinoModel(HuggingFaceModel):
    """Open-vocabulary object detection using text prompts.

    Detects objects described by natural language phrases.
    Phrases in the prompt are separated by ' . ' (e.g. "person . car . dog").
    """

    VARIANT_MAP = {
        "grounding_dino_tiny": "IDEA-Research/grounding-dino-tiny",
        "grounding_dino_base": "IDEA-Research/grounding-dino-base",
    }

    def __init__(self, node, variant):
        super().__init__(node, variant, AutoProcessor, AutoModelForZeroShotObjectDetection)
        self.box_threshold = node.get_parameter('box_threshold').value
        self.text_threshold = node.get_parameter('text_threshold').value
        self.default_prompt = node.get_parameter('default_prompt').value

    def preprocess(self, image, text=None):
        """Process image and text prompt for the model."""
        import cv2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prompt = text if text is not None else self.default_prompt

        inputs = self.processor(images=rgb, text=prompt, return_tensors="pt")
        self._last_input_ids = inputs.get("input_ids")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def postprocess(self, outputs, original_size=None):
        """Extract detections with phrase matching."""
        if original_size is not None:
            h, w = original_size
            target_sizes = torch.tensor([[h, w]], device=self.device)
        else:
            target_sizes = None

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=self._last_input_ids.to(self.device),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

        r = results[0]
        boxes = r["boxes"].cpu().numpy()
        scores = r["scores"].cpu().numpy()
        labels = r["labels"]

        return {
            'boxes': boxes,
            'scores': scores,
            'class_ids': np.zeros(len(scores), dtype=np.int32),
            'class_names': labels,
        }


Model = GroundingDinoModel
