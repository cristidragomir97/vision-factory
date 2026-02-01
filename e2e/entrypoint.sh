#!/bin/bash
set -e

echo "=== E2E Entrypoint ==="
echo "ROS_DISTRO: $ROS_DISTRO"
echo "PACKAGE_NAME: $PACKAGE_NAME"
echo "NODE_NAME: $NODE_NAME"

# Activate the venv (pip deps live here, isolated from deb packages)
echo "--- Activating venv ---"
source /opt/venv/bin/activate
echo "Python: $(which python3) ($(python3 --version))"

# Source ROS
echo "--- Sourcing ROS ---"
source /opt/ros/${ROS_DISTRO}/setup.bash
echo "ROS sourced: $(which ros2)"

# Build the mounted package with colcon
# COLCON_PYTHON_EXECUTABLE ensures entry point shebangs use the venv Python.
echo "--- Building with colcon ---"
export COLCON_PYTHON_EXECUTABLE=/opt/venv/bin/python3
cd /ros_ws
colcon build --packages-select ${PACKAGE_NAME}
echo "colcon build done."
echo "Entry point shebang: $(head -1 /ros_ws/install/${PACKAGE_NAME}/lib/${PACKAGE_NAME}/${PACKAGE_NAME}_node 2>/dev/null || echo 'not found')"

# Source the workspace
echo "--- Sourcing workspace ---"
source /ros_ws/install/setup.bash
echo "Workspace sourced."

# Run the test harness (all remaining args forwarded)
echo "--- Launching test harness ---"
echo "Args: $@"
exec python3 /ros_ws/test_harness.py "$@"
