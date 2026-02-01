"""YOLO model with ONNX export and session loading."""

import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from .model_base import YoloModel as _YoloModelBase


class YoloModel(_YoloModelBase):

    def export_onnx(self, opset=14, cache_dir="/tmp/onnx_models", model_name="model.onnx"):
        """Export model to ONNX format.

        Args:
            opset: ONNX opset version.
            cache_dir: Directory to cache the ONNX model.
            model_name: Filename for the cached model.

        Returns:
            Path to the exported ONNX file.
        """
        self.logger.info(f'YoloModel: exporting ONNX model (opset={opset})...')
        t0 = time.monotonic()
        export_path = self.model.export(format='onnx', opset=opset)
        self.logger.info(f'YoloModel: ONNX export done in {time.monotonic() - t0:.2f}s')

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        dest = cache_path / model_name
        shutil.move(str(export_path), str(dest))
        self.logger.info(f'YoloModel: ONNX model cached at {dest}')
        return dest

    def load_onnx(self, onnx_path):
        """Load an ONNX model through Ultralytics API.

        Replaces the current model with the ONNX model.

        Args:
            onnx_path: Path to the .onnx file.
        """
        self.logger.info(f'YoloModel: loading ONNX model from {onnx_path}...')
        t0 = time.monotonic()
        self.model = YOLO(str(onnx_path))
        self.logger.info(f'YoloModel: ONNX model loaded in {time.monotonic() - t0:.2f}s')


Model = YoloModel
