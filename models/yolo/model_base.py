"""YOLO model via Ultralytics API."""

import time

from ultralytics import YOLO


class YoloModel:
    """YOLO object detection model.

    Ultralytics handles preprocessing internally — we just call predict()
    and extract structured results from the output.
    """

    # Map variant IDs to Ultralytics weight filenames
    WEIGHTS_MAP = {
        "yolo_v8n": "yolov8n.pt",
        "yolo_v8s": "yolov8s.pt",
        "yolo_v8m": "yolov8m.pt",
        "yolo_v8l": "yolov8l.pt",
        "yolo_v8x": "yolov8x.pt",
        "yolo_v11n": "yolo11n.pt",
        "yolo_v11s": "yolo11s.pt",
        "yolo_v11m": "yolo11m.pt",
        "yolo_v11l": "yolo11l.pt",
    }

    def __init__(self, node, variant):
        self.node = node
        self.logger = node.get_logger()
        self.variant = variant
        self.confidence = node.get_parameter('confidence_threshold').value
        self.iou = node.get_parameter('iou_threshold').value
        classes_param = node.get_parameter('classes').value
        self.classes = classes_param if classes_param else None
        self.logger.info(f'YoloModel: conf={self.confidence}, iou={self.iou}, classes={self.classes}')

    def load(self, device):
        weights = self.WEIGHTS_MAP.get(self.variant, f"{self.variant}.pt")
        self.logger.info(f'YoloModel: loading weights "{weights}"...')
        t0 = time.monotonic()
        self.model = YOLO(weights)
        self.logger.info(f'YoloModel: YOLO("{weights}") loaded in {time.monotonic() - t0:.2f}s')
        self.logger.info(f'YoloModel: moving model to {device}...')
        t0 = time.monotonic()
        self.model.to(device)
        self.logger.info(f'YoloModel: model.to({device}) done in {time.monotonic() - t0:.2f}s')

    def predict(self, image):
        """Run Ultralytics inference (handles preprocessing internally)."""
        return self.model(
            image,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
        )

    def postprocess(self, results):
        """Extract structured data from Ultralytics results."""
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [result.names[int(cid)] for cid in class_ids]

        return {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'class_names': class_names,
        }


Model = YoloModel
