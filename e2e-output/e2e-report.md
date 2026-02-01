# E2E Test Report

**Model:** yolo | **Backend:** tensorrt | **Variant:** yolo_v8s
**ROS Distro:** jazzy | **Input:** usb_cam | **rosboard:** yes
**Date:** 2026-02-01 12:56:08 UTC | **Result:** FAIL

## Stages

| Stage | Status | Duration |
|-------|--------|----------|
| Package generation | PASS | 0.1s |
| Build & Inference | FAIL | 257.0s |

## Inference Summary

- **Node started:** True
- **Node startup time:** 0.82s
- **Input active:** True
- **Messages received:** 6
- **Output topic:** /detections

## Errors

- **Build & Inference:** Node exited with code 1
- **Harness:** Node exited with code 1

<details><summary>Build & Inference log</summary>

```
on3.12/site-packages/rclpy/executors.py", line 923, in spin_once
[harness 12:56:06] [viz]     self._spin_once_impl(timeout_sec)
[harness 12:56:06] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 904, in _spin_once_impl
[harness 12:56:06] [viz]     handler, entity, node = self.wait_for_ready_callbacks(
[harness 12:56:06] [viz]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[harness 12:56:06] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 874, in wait_for_ready_callbacks
[harness 12:56:06] [viz]     return next(self._cb_iter)
[harness 12:56:06] [viz]            ^^^^^^^^^^^^^^^^^^^
[harness 12:56:06] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 782, in _wait_for_ready_callbacks
[harness 12:56:06] [viz]     raise ExternalShutdownException()
[harness 12:56:06] [viz] rclpy.executors.ExternalShutdownException
[harness 12:56:06] [viz] 
[harness 12:56:06] [viz] During handling of the above exception, another exception occurred:
[harness 12:56:06] [viz] 
[harness 12:56:06] [viz] Traceback (most recent call last):
[harness 12:56:06] [viz]   File "/ros_ws/viz_node.py", line 119, in <module>
[harness 12:56:06] [viz]     main()
[harness 12:56:06] [viz]   File "/ros_ws/viz_node.py", line 115, in main
[harness 12:56:06] [viz]     rclpy.shutdown()
[harness 12:56:06] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 134, in shutdown
[harness 12:56:06] [viz]     _shutdown(context=context)
[harness 12:56:06] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/utilities.py", line 82, in shutdown
[harness 12:56:06] [viz]     context.shutdown()
[harness 12:56:06] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/context.py", line 129, in shutdown
[harness 12:56:06] [viz]     self.__context.shutdown()
[harness 12:56:06] [viz] rclpy._rclpy_pybind11.RCLError: failed to shutdown: rcl_shutdown already called on the given context, at ./src/rcl/init.c:333
[harness 12:56:07] [node] Traceback (most recent call last):
[harness 12:56:07] [node]   File "/ros_ws/install/e2e_yolo/lib/python3.12/site-packages/e2e_yolo/node.py", line 124, in main
[harness 12:56:07] [node]     rclpy.spin(node)
[harness 12:56:07] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 247, in spin
[harness 12:56:07] [node]     executor.spin_once()
[harness 12:56:07] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 923, in spin_once
[harness 12:56:07] [node]     self._spin_once_impl(timeout_sec)
[harness 12:56:07] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 904, in _spin_once_impl
[harness 12:56:07] [node]     handler, entity, node = self.wait_for_ready_callbacks(
[harness 12:56:07] [node]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[harness 12:56:07] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 874, in wait_for_ready_callbacks
[harness 12:56:07] [node]     return next(self._cb_iter)
[harness 12:56:07] [node]            ^^^^^^^^^^^^^^^^^^^
[harness 12:56:07] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 782, in _wait_for_ready_callbacks
[harness 12:56:07] [node]     raise ExternalShutdownException()
[harness 12:56:07] [node] rclpy.executors.ExternalShutdownException
[harness 12:56:07] [node] 
[harness 12:56:07] [node] During handling of the above exception, another exception occurred:
[harness 12:56:07] [node] 
[harness 12:56:07] [node] Traceback (most recent call last):
[harness 12:56:07] [node]   File "<string>", line 1, in <module>
[harness 12:56:07] [node]   File "/ros_ws/install/e2e_yolo/lib/python3.12/site-packages/e2e_yolo/node.py", line 129, in main
[harness 12:56:07] [node]     rclpy.shutdown()
[harness 12:56:07] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 134, in shutdown
[harness 12:56:07] [node]     _shutdown(context=context)
[harness 12:56:07] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/utilities.py", line 82, in shutdown
[harness 12:56:07] [node]     context.shutdown()
[harness 12:56:07] [node]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/context.py", line 129, in shutdown
[harness 12:56:07] [node]     self.__context.shutdown()
[harness 12:56:07] [node] rclpy._rclpy_pybind11.RCLError: failed to shutdown: rcl_shutdown already called on the given context, at ./src/rcl/init.c:333
[harness 12:56:07] Node exited with code 1
[harness 12:56:07] === Emitting report ===
---E2E_REPORT_START---
{
  "node_started": true,
  "node_startup_time_s": 0.82,
  "input_source": "usb_cam",
  "input_active": true,
  "rosboard_launched": true,
  "messages_received": 6,
  "first_message_received": true,
  "output_topic": "/detections",
  "duration_s": 30.0,
  "errors": [
    "Node exited with code 1"
  ]
}
---E2E_REPORT_END---


```

</details>
