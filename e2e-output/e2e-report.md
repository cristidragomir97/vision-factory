# E2E Test Report

**Model:** yolo | **Backend:** tensorrt | **Variant:** yolo_v8s
**ROS Distro:** jazzy | **Input:** usb_cam | **rosboard:** yes
**Date:** 2026-02-01 13:08:46 UTC | **Result:** PASS

## Stages

| Stage | Status | Duration |
|-------|--------|----------|
| Package generation | PASS | 0.0s |
| Build & Inference | PASS | 203.7s |

## Inference Summary

- **Node started:** True
- **Node startup time:** 0.82s
- **Input active:** True
- **Messages received:** 6
- **Output topic:** /detections

## Errors

None

<details><summary>Build & Inference log</summary>

```
e]: Viz frame 4500: 10 boxes drawn
[harness 13:08:27] [viz] [INFO] [1769951307.500122120] [e2e_viz_node]: Viz frame 4600: 9 boxes drawn
[harness 13:08:27] [node] [INFO] [1769951307.580265008] [e2e_yolo_node]: Frame 300: received image 640x480 enc=yuv422_yuy2
[harness 13:08:27] [node] [INFO] [1769951307.589308832] [e2e_yolo_node]: Published 8 detections on /detections
[harness 13:08:27] [node] [INFO] [1769951307.590414192] [e2e_yolo_node]: Frame 300: bridge=0.000s infer=0.006s publish=0.002s
[harness 13:08:29] [viz] [INFO] [1769951309.580751787] [e2e_viz_node]: Viz frame 4700: 9 boxes drawn
[harness 13:08:31] [viz] [INFO] [1769951311.646009643] [e2e_viz_node]: Viz frame 4800: 10 boxes drawn
[harness 13:08:31] [node] [INFO] [1769951311.737198371] [e2e_yolo_node]: Frame 400: received image 640x480 enc=yuv422_yuy2
[harness 13:08:31] [node] [INFO] [1769951311.745301394] [e2e_yolo_node]: Published 10 detections on /detections
[harness 13:08:31] [node] [INFO] [1769951311.745735831] [e2e_yolo_node]: Frame 400: bridge=0.001s infer=0.006s publish=0.001s
[harness 13:08:33] [viz] [INFO] [1769951313.725187010] [e2e_viz_node]: Viz frame 4900: 7 boxes drawn
[harness 13:08:35] [viz] [INFO] [1769951315.825106156] [e2e_viz_node]: Viz frame 5000: 7 boxes drawn
[harness 13:08:35] [node] [INFO] [1769951315.895775194] [e2e_yolo_node]: Frame 500: received image 640x480 enc=yuv422_yuy2
[harness 13:08:35] [node] [INFO] [1769951315.909296884] [e2e_yolo_node]: Published 7 detections on /detections
[harness 13:08:35] [node] [INFO] [1769951315.910102429] [e2e_yolo_node]: Frame 500: bridge=0.001s infer=0.010s publish=0.002s
[harness 13:08:37] [viz] [INFO] [1769951317.908823324] [e2e_viz_node]: Viz frame 5100: 9 boxes drawn
[harness 13:08:39] [viz] [INFO] [1769951319.994130435] [e2e_viz_node]: Viz frame 5200: 10 boxes drawn
[harness 13:08:40] [node] [INFO] [1769951320.064220794] [e2e_yolo_node]: Frame 600: received image 640x480 enc=yuv422_yuy2
[harness 13:08:40] [node] [INFO] [1769951320.071752855] [e2e_yolo_node]: Published 10 detections on /detections
[harness 13:08:40] [node] [INFO] [1769951320.072135689] [e2e_yolo_node]: Frame 600: bridge=0.001s infer=0.006s publish=0.001s
[harness 13:08:42] [viz] [INFO] [1769951322.066453405] [e2e_viz_node]: Viz frame 5300: 10 boxes drawn
[harness 13:08:44] [viz] [INFO] [1769951324.159061526] [e2e_viz_node]: Viz frame 5400: 9 boxes drawn
[harness 13:08:44] [node] [INFO] [1769951324.224001646] [e2e_yolo_node]: Frame 700: received image 640x480 enc=yuv422_yuy2
[harness 13:08:44] [node] [INFO] [1769951324.235070441] [e2e_yolo_node]: Published 8 detections on /detections
[harness 13:08:44] [node] [INFO] [1769951324.235933147] [e2e_yolo_node]: Frame 700: bridge=0.001s infer=0.008s publish=0.003s
[harness 13:08:45] Messages counted: 6
[harness 13:08:45] [rosboard] Exception in thread Thread-1 (_thread_spin_target):
[harness 13:08:45] [rosboard] Traceback (most recent call last):
[harness 13:08:45] [rosboard]   File "/usr/lib/python3.12/threading.py", line 1073, in _bootstrap_inner
[harness 13:08:45] [rosboard]     self.run()
[harness 13:08:45] [rosboard]   File "/usr/lib/python3.12/threading.py", line 1010, in run
[harness 13:08:45] [rosboard]     self._target(*self._args, **self._kwargs)
[harness 13:08:45] [rosboard]   File "/opt/rosboard/rosboard/rospy2/__init__.py", line 85, in _thread_spin_target
[harness 13:08:45] [rosboard]     rclpy.spin(_node)
[harness 13:08:45] [rosboard]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 247, in spin
[harness 13:08:45] [rosboard]     executor.spin_once()
[harness 13:08:45] [rosboard]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 923, in spin_once
[harness 13:08:45] [rosboard]     self._spin_once_impl(timeout_sec)
[harness 13:08:45] [rosboard]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 904, in _spin_once_impl
[harness 13:08:45] [rosboard]     handler, entity, node = self.wait_for_ready_callbacks(
[harness 13:08:45] [rosboard]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[harness 13:08:45] [rosboard]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 874, in wait_for_ready_callbacks
[harness 13:08:45] [rosboard]     return next(self._cb_iter)
[harness 13:08:45] [rosboard]            ^^^^^^^^^^^^^^^^^^^
[harness 13:08:45] [rosboard]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 782, in _wait_for_ready_callbacks
[harness 13:08:45] [rosboard]     raise ExternalShutdownException()
[harness 13:08:45] [rosboard] rclpy.executors.ExternalShutdownException
[harness 13:08:46] === Emitting report ===
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
  "errors": []
}
---E2E_REPORT_END---


```

</details>
