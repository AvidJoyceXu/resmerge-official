# Third-party projects and dependencies

ResMerge builds on the projects below for robot control, teleoperation, hand retargeting and learning. The learning code is substantially adapted from ConRFT to support IMLE pretraining, LoRA residual RL and multi-task residual-policy merging.

## Upstream projects

| Project | Role in ResMerge | Location / integration |
|---|---|---|
| [SERL](https://github.com/rail-berkeley/serl) | Foundation inherited through ConRFT for RL utilities and Franka interfaces | Adapted code in `imle_dex/serl_launcher/` and `imle_dex/serl_robot_infra/` |
| [ConRFT](https://github.com/cccedric/conrft) | Foundation for the real-robot RL framework, actor/learner workflow and robot interfaces | Adapted code in `imle_dex/` |
| [LEAP Hand API](https://github.com/leap-hand/LEAP_Hand_API) | Python motor communication and LEAP hand control | `LEAP_Hand_API/` |
| [XRoboToolkit Teleop Sample Python](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python) | PICO tracking client and teleoperation integration | `XRoboToolkit-Teleop-Sample-Python/`, integrated by `teleop/` |
| [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) | Receives tracking data from the PICO device | Installed separately with its matching Python SDK |
| [dex-retargeting](https://github.com/dexsuite/dex-retargeting) | Maps human-hand keypoints to robot joint targets | `dex-retargeting/`, configured for the LEAP right hand |
| [dex-urdf](https://github.com/dexsuite/dex-urdf) | Robot geometry for hand retargeting | LEAP right-hand assets in `dex-urdf/` |

The bundled code includes ResMerge-specific adaptations. Use the local packages installed by `install.sh`; follow the upstream hardware setup guides for your machine.

## Main runtime libraries

| Libraries | Purpose |
|---|---|
| PyTorch, torchvision, timm, diffusers | IMLE policy and visual-model components |
| JAX, Flax, Optax, Distrax | Residual RL networks, optimization and action distributions |
| Agentlace | Communication between the robot actor and learner |
| Gymnasium | Environment and observation/action interfaces |
| NumPy, SciPy | Numerical computation and data processing |
| Zarr, h5py | Demonstration and training-data storage |
| Hydra, OmegaConf | Training configuration |
| OpenCV, pyrealsense2 | Image processing and camera access |
| Pinocchio, NLopt | Robot kinematics and retargeting optimization |
| dynamixel-sdk, pyserial | LEAP motor and serial communication |
| pyspacemouse, hidapi, pynput | Teleoperation inputs and keyboard labels |
| Flask, requests | Robot HTTP service and client communication |
| Matplotlib | Correction-field analysis plots |
| Weights & Biases | Optional experiment logging |

Installation dependencies are maintained in `requirements/`. Hardware drivers, ROS, the Franka controller and XRoboToolkit PC Service are configured separately as described in [README.md](README.md) and [HARDWARE.md](HARDWARE.md).

## Licenses and acknowledgments

Thanks to the authors and maintainers of these projects. Bundled upstream license files are retained alongside their code: ConRFT and SERL use Apache-2.0, XRoboToolkit Teleop and dex-retargeting use MIT, and LEAP Hand API uses CC BY-NC 4.0. Each component remains subject to its respective license.


The LEAP hand geometry retains its bundled `dex-urdf/robots/hands/leap_hand/LICENSE.txt` notice. The root ResMerge license does not replace any third-party license. In particular, the bundled LEAP Hand API is licensed for non-commercial use under CC BY-NC 4.0.

## Integrated source

The real-robot implementation was contributed from [vinkda/Resmerge_real_robot](https://github.com/vinkda/Resmerge_real_robot), commit `9368bdadfaeb126936d89610c51ac503cbaae453`, on 2026-10-31. That upstream repository may require access permission. The official copy adds paper-RFC analysis, sequential merge support, corrected pruning compensation, deterministic evaluation, consistent residual-scale metadata and documentation for its `real_robot/` location. Hardware calibration and controller behavior remain configured through the supplied task and hardware settings.

SERL upstream Apache-2.0 notices are included in `imle_dex/serl_launcher/LICENSE` and `imle_dex/serl_robot_infra/LICENSE`. The inherited `serl_launcher/setup.py` contains an MIT metadata label that differs from the upstream LICENSE; both the inherited metadata and the upstream notices are retained for provenance.
