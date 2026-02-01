"""YOLO model with TensorRT export and engine loading."""

import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from .model_base import YoloModel as _YoloModelBase


class YoloModel(_YoloModelBase):

    def export_tensorrt(self, half=True, int8=False, cache_dir="/tmp/trt_engines", engine_name="model.engine"):
        """Export model to TensorRT engine format.

        Args:
            half: Use FP16 precision.
            int8: Use INT8 precision (requires calibration).
            cache_dir: Directory to cache the engine file.
            engine_name: Filename for the cached engine.

        Returns:
            Path to the exported engine file.
        """
        self.logger.info(f'YoloModel: exporting TensorRT engine (half={half}, int8={int8})...')
        t0 = time.monotonic()
        export_path = self.model.export(format='engine', half=half, int8=int8)
        self.logger.info(f'YoloModel: TensorRT export done in {time.monotonic() - t0:.2f}s')

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        dest = cache_path / engine_name
        shutil.move(str(export_path), str(dest))
        self.logger.info(f'YoloModel: engine cached at {dest}')
        return dest

    def load_engine(self, engine_path):
        """Load a TensorRT engine through Ultralytics API.

        Replaces the current model with the TRT engine.

        Args:
            engine_path: Path to the .engine file.
        """
        self.logger.info(f'YoloModel: loading TensorRT engine from {engine_path}...')
        t0 = time.monotonic()
        self.model = YOLO(str(engine_path))
        self.logger.info(f'YoloModel: TensorRT engine loaded in {time.monotonic() - t0:.2f}s')


Model = YoloModel
