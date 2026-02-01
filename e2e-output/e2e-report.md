# E2E Test Report

**Model:** yolo | **Backend:** tensorrt | **Variant:** yolo_v8s
**ROS Distro:** jazzy | **Input:** usb_cam | **rqt:** yes
**Date:** 2026-02-01 00:16:27 UTC | **Result:** FAIL

## Stages

| Stage | Status | Duration |
|-------|--------|----------|
| Package generation | PASS | 0.0s |
| Build & Inference | FAIL | 306.9s |

## Inference Summary

- **Node started:** True
- **Node startup time:** 0.83s
- **Bag played:** False
- **Messages received:** 0
- **Output topic:** /detections

## Errors

- **Build & Inference:** Unexpected error: Command '['ros2', 'topic', 'echo', '/detections', '--once']' timed out after 300.0 seconds; Node exited with code 1
- **Harness:** Unexpected error: Command '['ros2', 'topic', 'echo', '/detections', '--once']' timed out after 300.0 seconds
- **Harness:** Node exited with code 1

<details><summary>Build & Inference log</summary>

```
 topic /detections...
[harness 00:11:26] $ ros2 topic info /detections
[harness 00:11:26] [node] [INFO] [1769904686.340461772] [e2e_yolo_node]: YoloModel: model.to(cuda:0) done in 0.29s
[harness 00:11:26] [node] [INFO] [1769904686.340660121] [e2e_yolo_node]: YoloModel: exporting TensorRT engine (half=True, int8=False)...
[harness 00:11:26]   stdout: Type: vision_msgs/msg/Detection2DArray
[harness 00:11:26]   stdout: Publisher count: 0
[harness 00:11:26]   stdout: Subscription count: 1
[harness 00:11:26]   exit code: 0
[harness 00:11:26] Waiting for first message on /detections (timeout 300.0s)...
[harness 00:13:55] [node] [INFO] [1769904835.361604723] [e2e_yolo_node]: YoloModel: TensorRT export done in 149.02s
[harness 00:13:55] [node] [INFO] [1769904835.361918894] [e2e_yolo_node]: YoloModel: engine cached at /tmp/trt_engines/yolo_v8s_fp16.engine
[harness 00:13:55] [node] [INFO] [1769904835.362077419] [e2e_yolo_node]: YoloModel: loading TensorRT engine from /tmp/trt_engines/yolo_v8s_fp16.engine...
[harness 00:13:55] [node] [INFO] [1769904835.362795665] [e2e_yolo_node]: YoloModel: TensorRT engine loaded in 0.00s
[harness 00:13:55] [node] [INFO] [1769904835.362978381] [e2e_yolo_node]: TensorRtRunner: ready.
[harness 00:13:55] [node] [INFO] [1769904835.363129397] [e2e_yolo_node]: Runner created in 151.85s
[harness 00:13:55] [node] [INFO] [1769904835.363278751] [e2e_yolo_node]: Subscribing to image topic: /camera/image_raw
[harness 00:13:55] [node] [INFO] [1769904835.373268528] [e2e_yolo_node]: Publisher created: detections (Detection2DArray)
[harness 00:13:55] [node] [INFO] [1769904835.373676448] [e2e_yolo_node]: Publisher created: visualization (Image)
[harness 00:13:55] [node] [INFO] [1769904835.373847188] [e2e_yolo_node]: e2e_yolo_node ready — yolo:yolo_v8s on tensorrt
[harness 00:16:26] Unexpected error: Command '['ros2', 'topic', 'echo', '/detections', '--once']' timed out after 300.0 seconds
[harness 00:16:26] [node] Traceback (most recent call last):
[harness 00:16:26] [node]   File "/ros_ws/install/e2e_yolo/lib/python3.12/site-packages/e2e_yolo/node.py", line 124, in main
[harness 00:16:26] [node]     rclpy.spin(node)
[harness 00:16:26] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 247, in spin
[harness 00:16:26] [node]     executor.spin_once()
[harness 00:16:26] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 923, in spin_once
[harness 00:16:26] [node]     self._spin_once_impl(timeout_sec)
[harness 00:16:26] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 904, in _spin_once_impl
[harness 00:16:26] [node]     handler, entity, node = self.wait_for_ready_callbacks(
[harness 00:16:26] [node]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[harness 00:16:26] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 874, in wait_for_ready_callbacks
[harness 00:16:26] [node]     return next(self._cb_iter)
[harness 00:16:26] [node]            ^^^^^^^^^^^^^^^^^^^
[harness 00:16:26] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 782, in _wait_for_ready_callbacks
[harness 00:16:26] [node]     raise ExternalShutdownException()
[harness 00:16:26] [node] rclpy.executors.ExternalShutdownException
[harness 00:16:26] [node] 
[harness 00:16:26] [node] During handling of the above exception, another exception occurred:
[harness 00:16:26] [node] 
[harness 00:16:26] [node] Traceback (most recent call last):
[harness 00:16:26] [node]   File "<string>", line 1, in <module>
[harness 00:16:26] [node]   File "/ros_ws/install/e2e_yolo/lib/python3.12/site-packages/e2e_yolo/node.py", line 129, in main
[harness 00:16:26] [node]     rclpy.shutdown()
[harness 00:16:26] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 134, in shutdown
[harness 00:16:26] [node]     _shutdown(context=context)
[harness 00:16:26] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/utilities.py", line 82, in shutdown
[harness 00:16:26] [node]     context.shutdown()
[harness 00:16:26] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/context.py", line 129, in shutdown
[harness 00:16:26] [node]     self.__context.shutdown()
[harness 00:16:26] [node] rclpy._rclpy_pybind11.RCLError: failed to shutdown: rcl_shutdown already called on the given context, at ./src/rcl/init.c:333
[harness 00:16:27] Node exited with code 1
[harness 00:16:27] === Emitting report ===
---E2E_REPORT_START---
{
  "node_started": true,
  "node_startup_time_s": 0.83,
  "input_source": "usb_cam",
  "input_active": true,
  "rqt_launched": false,
  "messages_received": 0,
  "first_message_received": false,
  "output_topic": "/detections",
  "duration_s": 30.0,
  "errors": [
    "Unexpected error: Command '['ros2', 'topic', 'echo', '/detections', '--once']' timed out after 300.0 seconds",
    "Node exited with code 1"
  ]
}
---E2E_REPORT_END---


```

</details>
