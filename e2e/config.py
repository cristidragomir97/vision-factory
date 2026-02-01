"""Central configuration for e2e test infrastructure."""

CUDA_VERSION = "13.0.0"
ROS_DISTRO = "jazzy"
BASE_IMAGE_TAG = f"vision-factory-e2e-base:{ROS_DISTRO}"
COMPOSE_NETWORK = "vision-e2e"
COMPOSE_PROJECT = "vision-e2e"
