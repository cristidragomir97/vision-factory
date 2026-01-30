FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install ROS2 jazzy on top of CUDA base
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y \
    ros-jazzy-ros-base \
    python3-colcon-common-extensions \
    python3-pip \
    python3-opencv \
    ros-jazzy-vision-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-usb-cam \
    ros-jazzy-rqt \
    ros-jazzy-rqt-image-view \
    ros-jazzy-rqt-topic \
    && rm -rf /var/lib/apt/lists/*

# Pre-install pip deps so they're cached in the image layer
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt --break-system-packages

WORKDIR /ros_ws
