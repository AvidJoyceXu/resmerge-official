#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3.11}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 is required"'
# Some keyboard/USB dependencies build native extensions on Linux.
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
command -v "$CC" >/dev/null || { echo "Install a C/C++ build toolchain (e.g. build-essential on Ubuntu)"; exit 1; }
"$PYTHON" -m venv .resmerge-venv
PY="$ROOT/.resmerge-venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements/requirements.txt
"$PY" -m pip install --no-deps -r requirements/requirements-git.txt
"$PY" -m pip install --no-deps -e dex-retargeting -e XRoboToolkit-Teleop-Sample-Python -e imle_dex/serl_launcher -e imle_dex/serl_robot_infra -e imle_dex/precise_dex
"$PY" -m pip check
printf 'Python runtime installed. Native PICO SDK, PC Service and ROS controller remain separate prerequisites.\n'
