FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy package
WORKDIR /ros_ws/src
COPY . e2e_depth_anything/

# Build
WORKDIR /ros_ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select e2e_depth_anything

# Entrypoint
COPY <<'ENTRYPOINT' /entrypoint.sh
#!/bin/bash
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash
exec "$@"
ENTRYPOINT
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "e2e_depth_anything", "vision.launch.py"]
