"""CUDA inference runner for grounding_dino."""

import time
import torch
from .model import GroundingDinoModel


class CudaRunner:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()
        self.logger.info('CudaRunner: initializing...')

        self.logger.info('CudaRunner: detecting CUDA device...')
        self.device = torch.device('cuda:0')
        self.logger.info(f'CudaRunner: using device {self.device}')
        self.logger.info(f'CudaRunner: CUDA available={torch.cuda.is_available()}, '
                         f'device_count={torch.cuda.device_count()}')
        if torch.cuda.is_available():
            self.logger.info(f'CudaRunner: GPU name={torch.cuda.get_device_name(0)}')

        self.logger.info('CudaRunner: creating model instance...')
        t0 = time.monotonic()
        self.model = GroundingDinoModel(node)
        self.logger.info(f'CudaRunner: model instance created in {time.monotonic() - t0:.2f}s')

        self.logger.info('CudaRunner: loading weights to device...')
        t0 = time.monotonic()
        self.model.load(self.device)
        self.logger.info(f'CudaRunner: weights loaded in {time.monotonic() - t0:.2f}s')

        self.logger.info('CudaRunner: ready.')

    def infer(self, image, text=None):
        """Run inference with image and text input."""
        inputs = self.model.preprocess(image, text=text)
        with torch.no_grad():
            outputs = self.model.forward(inputs)
        return self.model.postprocess(outputs, original_size=image.shape[:2])
