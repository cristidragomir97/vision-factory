"""ROS2 vision node for yolo (yolo_v8s)."""

import time
import traceback
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, Pose2D

from .model import YoloModel
from .runner import CudaRunner


class VisionNode(Node):
    def __init__(self):
        super().__init__('e2e_yolo_node')
        self.get_logger().info('Initializing e2e_yolo_node...')
        self.bridge = CvBridge()
        self._frame_count = 0

        # Declare parameters
        self.get_logger().info('Declaring parameters...')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('classes', [])
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.get_logger().info('Parameters declared.')

        # Initialize model and runner
        self.get_logger().info('Creating model (YoloModel)...')
        t0 = time.monotonic()
        self.model = YoloModel(self)
        self.get_logger().info(f'Model created in {time.monotonic() - t0:.2f}s')

        self.get_logger().info('Creating runner (CudaRunner)...')
        t0 = time.monotonic()
        self.runner = CudaRunner(self)
        self.get_logger().info(f'Runner created in {time.monotonic() - t0:.2f}s')

        # Image subscriber
        input_topic = self.get_parameter('input_topic').value
        self.get_logger().info(f'Subscribing to image topic: {input_topic}')
        self.image_sub = self.create_subscription(
            Image, input_topic, self._image_callback, 10)


        # Publishers
        self.detections_pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.get_logger().info('Publisher created: detections (Detection2DArray)')
        self.visualization_pub = self.create_publisher(Image, 'visualization', 10)
        self.get_logger().info('Publisher created: visualization (Image)')

        self.get_logger().info(
            'e2e_yolo_node ready — yolo:yolo_v8s on cuda')

    def _image_callback(self, msg: Image):
        self._frame_count += 1
        frame_id = self._frame_count
        try:
            if frame_id <= 3 or frame_id % 100 == 0:
                self.get_logger().info(
                    f'Frame {frame_id}: received image '
                    f'{msg.width}x{msg.height} enc={msg.encoding}')

            t0 = time.monotonic()
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            t_bridge = time.monotonic() - t0

            t0 = time.monotonic()
            result = self.runner.infer(cv_image)
            t_infer = time.monotonic() - t0

            t0 = time.monotonic()
            self._publish(result, msg.header)
            t_publish = time.monotonic() - t0

            if frame_id <= 3 or frame_id % 100 == 0:
                self.get_logger().info(
                    f'Frame {frame_id}: bridge={t_bridge:.3f}s '
                    f'infer={t_infer:.3f}s publish={t_publish:.3f}s')
        except Exception as e:
            self.get_logger().error(
                f'Frame {frame_id}: exception in callback:\n'
                f'{traceback.format_exc()}')

    def _publish(self, result, header):
        det_array = Detection2DArray()
        det_array.header = header

        boxes = result.get('boxes', [])
        scores = result.get('scores', [])
        class_names = result.get('class_names', [''] * len(scores))

        for i in range(len(scores)):
            det = Detection2D()
            det.header = header

            # Bounding box center + size
            x1, y1, x2, y2 = boxes[i]
            det.bbox.center = Pose2D(x=(x1 + x2) / 2, y=(y1 + y2) / 2, theta=0.0)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            # Hypothesis
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(class_names[i])
            hyp.hypothesis.score = float(scores[i])
            det.results.append(hyp)

            det_array.detections.append(det)

        self.detections_pub.publish(det_array)
        if self._frame_count <= 3 or self._frame_count % 100 == 0:
            self.get_logger().info(
                f'Published {len(det_array.detections)} detections on /detections')


def main(args=None):
    rclpy.init(args=args)
    try:
        node = VisionNode()
    except Exception:
        import traceback as tb
        tb.print_exc()
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
