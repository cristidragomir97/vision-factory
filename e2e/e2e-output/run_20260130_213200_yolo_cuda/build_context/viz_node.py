"""Visualization node — draws bounding boxes on camera images.

Subscribes to an image topic and a Detection2DArray topic, overlays
bounding boxes with class labels and scores, publishes the annotated image.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np

# Colors for different classes (BGR)
COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 255, 128), (128, 255, 255),
]


class VizNode(Node):
    def __init__(self):
        super().__init__('e2e_viz_node')
        self.bridge = CvBridge()
        self._latest_image = None
        self._latest_detections = None
        self._frame_count = 0

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('detection_topic', '/detections')
        self.declare_parameter('output_topic', '/detections_viz')

        image_topic = self.get_parameter('image_topic').value
        det_topic = self.get_parameter('detection_topic').value
        out_topic = self.get_parameter('output_topic').value

        self.image_sub = self.create_subscription(
            Image, image_topic, self._image_cb, 10)
        self.det_sub = self.create_subscription(
            Detection2DArray, det_topic, self._det_cb, 10)
        self.viz_pub = self.create_publisher(Image, out_topic, 10)

        self.get_logger().info(
            f'VizNode ready: {image_topic} + {det_topic} -> {out_topic}')

    def _image_cb(self, msg: Image):
        self._latest_image = msg
        self._draw_and_publish()

    def _det_cb(self, msg: Detection2DArray):
        self._latest_detections = msg
        self._draw_and_publish()

    def _draw_and_publish(self):
        if self._latest_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self._latest_image, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        if self._latest_detections is not None:
            for i, det in enumerate(self._latest_detections.detections):
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                w = det.bbox.size_x
                h = det.bbox.size_y

                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)

                color = COLORS[i % len(COLORS)]

                # Draw box
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                if det.results:
                    r = det.results[0]
                    label = f'{r.hypothesis.class_id} {r.hypothesis.score:.2f}'
                else:
                    label = f'det_{i}'

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(cv_image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(cv_image, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)

        self._frame_count += 1
        if self._frame_count <= 3 or self._frame_count % 100 == 0:
            n_det = len(self._latest_detections.detections) if self._latest_detections else 0
            self.get_logger().info(f'Viz frame {self._frame_count}: {n_det} boxes drawn')

        viz_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        viz_msg.header = self._latest_image.header
        self.viz_pub.publish(viz_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
