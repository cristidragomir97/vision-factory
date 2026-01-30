#!/bin/bash
set -e

echo "=== E2E Entrypoint ==="
echo "ROS_DISTRO: $ROS_DISTRO"
echo "PACKAGE_NAME: $PACKAGE_NAME"
echo "NODE_NAME: $NODE_NAME"

# Source ROS
echo "--- Sourcing ROS ---"
source /opt/ros/${ROS_DISTRO}/setup.bash
echo "ROS sourced: $(which ros2)"

# Install pip deps from the mounted package
if [ -f /ros_ws/src/${PACKAGE_NAME}/requirements.txt ]; then
    echo "--- Installing pip requirements ---"
    pip3 install --no-cache-dir -r /ros_ws/src/${PACKAGE_NAME}/requirements.txt --break-system-packages
    echo "pip install done."
else
    echo "WARNING: No requirements.txt found at /ros_ws/src/${PACKAGE_NAME}/requirements.txt"
fi

# Build the mounted package with colcon
echo "--- Building with colcon ---"
cd /ros_ws
colcon build --packages-select ${PACKAGE_NAME}
echo "colcon build done."

# Source the workspace
echo "--- Sourcing workspace ---"
source /ros_ws/install/setup.bash
echo "Workspace sourced."

# Run the test harness (all remaining args forwarded)
echo "--- Launching test harness ---"
echo "Args: $@"
exec python3 /ros_ws/test_harness.py "$@"
