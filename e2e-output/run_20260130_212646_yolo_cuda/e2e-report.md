# E2E Test Report

**Model:** yolo | **Backend:** cuda | **Variant:** yolo_v8s
**ROS Distro:** jazzy | **Input:** usb_cam | **rqt:** yes
**Date:** 2026-01-30 19:28:58 UTC | **Result:** PASS

## Stages

| Stage | Status | Duration |
|-------|--------|----------|
| Package generation | PASS | 0.0s |
| Docker build | PASS | 1.5s |
| Build & Inference | PASS | 130.3s |

## Inference Summary

- **Node started:** True
- **Node startup time:** 0.8s
- **Bag played:** False
- **Messages received:** 7
- **Output topic:** /detections

## Errors

None

<details><summary>Docker build log</summary>

```
#0 building with "default" instance using docker driver

#1 [internal] load build definition from Dockerfile
#1 transferring dockerfile: 1.19kB done
#1 DONE 0.0s

#2 [internal] load metadata for docker.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
#2 DONE 1.1s

#3 [internal] load .dockerignore
#3 transferring context: 2B done
#3 DONE 0.0s

#4 [1/5] FROM docker.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04@sha256:c2621d98e7de80c2aec5eb8403b19c67454c8f5b0c929e8588fd3563c9b6558d
#4 resolve docker.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04@sha256:c2621d98e7de80c2aec5eb8403b19c67454c8f5b0c929e8588fd3563c9b6558d 0.0s done
#4 DONE 0.0s

#5 [internal] load build context
#5 transferring context: 207B done
#5 DONE 0.0s

#6 [4/5] RUN pip3 install --no-cache-dir -r /tmp/requirements.txt --break-system-packages
#6 CACHED

#7 [2/5] RUN apt-get update && apt-get install -y     software-properties-common     curl     && add-apt-repository universe     && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list     && apt-get update && apt-get install -y     ros-jazzy-ros-base     python3-colcon-common-extensions     python3-pip     python3-opencv     ros-jazzy-vision-msgs     ros-jazzy-cv-bridge     ros-jazzy-usb-cam     ros-jazzy-rqt     ros-jazzy-rqt-image-view     ros-jazzy-rqt-topic     && rm -rf /var/lib/apt/lists/*
#7 CACHED

#8 [3/5] COPY requirements.txt /tmp/requirements.txt
#8 CACHED

#9 [5/5] WORKDIR /ros_ws
#9 CACHED

#10 exporting to image
#10 exporting layers done
#10 exporting manifest sha256:03671c2fc7a3e3a0e82b4f5d931406d58737e5f6c618c05676fb98bf666ab12f done
#10 exporting config sha256:df72c731b12a5e559ef2b492c5cef4f8fdd9cb0111d9854cfe4796090bb907ee done
#10 exporting attestation manifest sha256:9002ccb5e2bccaed0088833b3e3af608934ffbdbee0b3826c033f9125710493b 0.0s done
#10 exporting manifest list sha256:0688ec8d70eb5b3e7f514ee568fa6e9aff3529be4a1b2e1416c4c6941b9bd34a
#10 exporting manifest list sha256:0688ec8d70eb5b3e7f514ee568fa6e9aff3529be4a1b2e1416c4c6941b9bd34a done
#10 naming to docker.io/library/e2e-base-cuda:jazzy done
#10 unpacking to docker.io/library/e2e-base-cuda:jazzy 0.0s done
#10 DONE 0.2s

```

</details>

<details><summary>Build & Inference log</summary>

```

[harness 19:28:43] [node] [INFO] [1769801323.863670985] [e2e_yolo_node]: Frame 2600: received image 640x480 enc=yuv422_yuy2
[harness 19:28:43] [node] [INFO] [1769801323.875588622] [e2e_yolo_node]: Published 12 detections on /detections
[harness 19:28:43] [node] [INFO] [1769801323.876065404] [e2e_yolo_node]: Frame 2600: bridge=0.001s infer=0.010s publish=0.002s
[harness 19:28:45] [viz] [INFO] [1769801325.394920049] [e2e_viz_node]: Viz frame 5300: 10 boxes drawn
[harness 19:28:47] [viz] [INFO] [1769801327.484381006] [e2e_viz_node]: Viz frame 5400: 9 boxes drawn
[harness 19:28:48] [node] [INFO] [1769801328.024154504] [e2e_yolo_node]: Frame 2700: received image 640x480 enc=yuv422_yuy2
[harness 19:28:48] [node] [INFO] [1769801328.039936417] [e2e_yolo_node]: Published 10 detections on /detections
[harness 19:28:48] [node] [INFO] [1769801328.040439338] [e2e_yolo_node]: Frame 2700: bridge=0.001s infer=0.014s publish=0.001s
[harness 19:28:49] [viz] [INFO] [1769801329.557885306] [e2e_viz_node]: Viz frame 5500: 11 boxes drawn
[harness 19:28:51] [viz] [INFO] [1769801331.645866467] [e2e_viz_node]: Viz frame 5600: 11 boxes drawn
[harness 19:28:52] [node] [INFO] [1769801332.183589107] [e2e_yolo_node]: Frame 2800: received image 640x480 enc=yuv422_yuy2
[harness 19:28:52] [node] [INFO] [1769801332.200478790] [e2e_yolo_node]: Published 11 detections on /detections
[harness 19:28:52] [node] [INFO] [1769801332.201302981] [e2e_yolo_node]: Frame 2800: bridge=0.001s infer=0.014s publish=0.002s
[harness 19:28:53] [viz] [INFO] [1769801333.721222811] [e2e_viz_node]: Viz frame 5700: 11 boxes drawn
[harness 19:28:55] [viz] [INFO] [1769801335.792139893] [e2e_viz_node]: Viz frame 5800: 10 boxes drawn
[harness 19:28:56] [node] [INFO] [1769801336.328586326] [e2e_yolo_node]: Frame 2900: received image 640x480 enc=yuv422_yuy2
[harness 19:28:56] [node] [INFO] [1769801336.343012271] [e2e_yolo_node]: Published 10 detections on /detections
[harness 19:28:56] [node] [INFO] [1769801336.343542902] [e2e_yolo_node]: Frame 2900: bridge=0.001s infer=0.012s publish=0.002s
[harness 19:28:57] [viz] [INFO] [1769801337.870719865] [e2e_viz_node]: Viz frame 5900: 10 boxes drawn
[harness 19:28:57] Messages counted: 7
[harness 19:28:57] [viz] Traceback (most recent call last):
[harness 19:28:57] [viz]   File "/ros_ws/viz_node.py", line 110, in main
[harness 19:28:57] [viz]     rclpy.spin(node)
[harness 19:28:57] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 247, in spin
[harness 19:28:57] [viz]     executor.spin_once()
[harness 19:28:57] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 923, in spin_once
[harness 19:28:57] [viz]     self._spin_once_impl(timeout_sec)
[harness 19:28:57] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 904, in _spin_once_impl
[harness 19:28:57] [viz]     handler, entity, node = self.wait_for_ready_callbacks(
[harness 19:28:57] [viz]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[harness 19:28:57] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 874, in wait_for_ready_callbacks
[harness 19:28:57] [viz]     return next(self._cb_iter)
[harness 19:28:57] [viz]            ^^^^^^^^^^^^^^^^^^^
[harness 19:28:57] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 782, in _wait_for_ready_callbacks
[harness 19:28:57] [viz]     raise ExternalShutdownException()
[harness 19:28:57] [viz] rclpy.executors.ExternalShutdownException
[harness 19:28:57] [viz] 
[harness 19:28:57] [viz] During handling of the above exception, another exception occurred:
[harness 19:28:57] [viz] 
[harness 19:28:57] [viz] Traceback (most recent call last):
[harness 19:28:57] [viz]   File "/ros_ws/viz_node.py", line 119, in <module>
[harness 19:28:57] [viz]     main()
[harness 19:28:57] [viz]   File "/ros_ws/viz_node.py", line 115, in main
[harness 19:28:57] [viz]     rclpy.shutdown()
[harness 19:28:57] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 134, in shutdown
[harness 19:28:57] [viz]     _shutdown(context=context)
[harness 19:28:57] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/utilities.py", line 82, in shutdown
[harness 19:28:57] [viz]     context.shutdown()
[harness 19:28:57] [viz]   File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/context.py", line 129, in shutdown
[harness 19:28:57] [viz]     self.__context.shutdown()
[harness 19:28:57] [viz] rclpy._rclpy_pybind11.RCLError: failed to shutdown: rcl_shutdown already called on the given context, at ./src/rcl/init.c:333
---E2E_REPORT_START---
{
  "node_started": true,
  "node_startup_time_s": 0.8,
  "input_source": "usb_cam",
  "input_active": true,
  "rqt_launched": true,
  "messages_received": 7,
  "first_message_received": true,
  "output_topic": "/detections",
  "duration_s": 120.0,
  "errors": []
}
---E2E_REPORT_END---
[harness 19:28:58] === Emitting report ===


```

</details>
