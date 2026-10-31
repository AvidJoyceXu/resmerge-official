#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
: "${CATKIN_WS:?Set CATKIN_WS to your configured ROS workspace}"
: "${ROBOT_IP:?Set ROBOT_IP to the Franka robot address}"
source "${CATKIN_WS}/devel/setup.bash"
export PYTHONPATH="${REPO_ROOT}/imle_dex/serl_robot_infra:${PYTHONPATH:-}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,${ROBOT_IP}"
exec python "${REPO_ROOT}/imle_dex/serl_robot_infra/robot_servers/franka_server.py" \
  --gripper_type=None --robot_ip="${ROBOT_IP}" \
  --flask_url=127.0.0.1 --ros_port="${ROS_PORT:-11311}" "$@"
