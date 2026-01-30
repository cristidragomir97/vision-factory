"""ROS2 vision node for depth_anything (depth_anything_v2_vitb)."""

import time
import traceback
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

from .model import DepthModel
from .runner import CudaRunner


class VisionNode(Node):
    def __init__(self):
        super().__init__('e2e_depth_anything_node')
        self.get_logger().info('Initializing e2e_depth_anything_node...')
        self.bridge = CvBridge()
        self._frame_count = 0

        # Declare parameters
        self.get_logger().info('Declaring parameters...')
        self.declare_parameter('publish_colored', True)
        self.declare_parameter('publish_pointcloud', False)
        self.declare_parameter('colormap', 'turbo')
        self.declare_parameter('min_depth', 0.0)
        self.declare_parameter('max_depth', 1.0)
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.get_logger().info('Parameters declared.')

        # Initialize model and runner
        self.get_logger().info('Creating model (DepthModel)...')
        t0 = time.monotonic()
        self.model = DepthModel(self)
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
        self.depth_pub = self.create_publisher(Image, 'depth', 10)
        self.get_logger().info('Publisher created: depth (Image)')
        self.depth_colored_pub = self.create_publisher(Image, 'depth_colored', 10)
        self.get_logger().info('Publisher created: depth_colored (Image)')
        # TODO: Add publisher for sensor_msgs/PointCloud2
        # self.pointcloud_pub = self.create_publisher(..., 'pointcloud', 10)

        self.get_logger().info(
            'e2e_depth_anything_node ready — depth_anything:depth_anything_v2_vitb on cuda')

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
        depth = result['depth']
        depth_msg = self.bridge.cv2_to_imgmsg(
            depth.astype(np.float32), encoding='32FC1')
        depth_msg.header = header
        self.depth_pub.publish(depth_msg)

        if 'depth_colored' in result:
            color_msg = self.bridge.cv2_to_imgmsg(result['depth_colored'], encoding='rgb8')
            color_msg.header = header
            self.depth_colored_pub.publish(color_msg)

        if self._frame_count <= 3 or self._frame_count % 100 == 0:
            self.get_logger().info('Published depth map')


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
