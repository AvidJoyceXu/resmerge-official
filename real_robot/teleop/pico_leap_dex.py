"""PICO hand tracking and LEAP right-hand teleoperation.

Pipeline: PICO tracking -> 21 MediaPipe keypoints -> dex-retargeting joint
angles -> LEAP joint ordering and offsets -> motor position commands.
PyBullet displays the retargeted joint angles when the GUI is enabled.

Run from the repository root:
    python teleop/pico_leap_dex.py
    python teleop/pico_leap_dex.py --hardware
    python teleop/pico_leap_dex.py --hardware --no-gui
    python teleop/pico_leap_dex.py --print-joints

The script resolves dex-urdf, XRoboToolkit-Teleop-Sample-Python and
LEAP_Hand_API relative to the repository root. Configure the PICO SDK and
LEAP serial connection as described in README.md and HARDWARE.md.

Outputs: dex_qpos uses the URDF joint order; leap_q contains 16 angles in
LEAP API order (index, middle, ring, thumb), in radians. The PyBullet model
is placed at [-0.3, 0, 0] for display.
"""

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import pybullet as p

try:
    
    import sys
    import os
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.normpath(os.path.join(_HERE, '..'))
    leap_api_path = os.path.join(_ROOT, 'LEAP_Hand_API', 'python')
    if leap_api_path not in sys.path:
        sys.path.insert(0, leap_api_path)
    
    
    from main import LeapNode
    _LEAPNODE_AVAILABLE = True
except ImportError as e:
    LeapNode = None          # type: ignore[assignment,misc]
    _LEAPNODE_AVAILABLE = False
    print(f"[WARNING] Could not import LeapNode: {e}")
    print(f"          Make sure LEAP_Hand_API is at: {leap_api_path}")
    
# ---------------------------------------------------------------------------
# Root of the leaphand workspace  (parent of this file's directory)
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))   # …/PICO_Leap_Dex_Teleop
_ROOT      = os.path.normpath(os.path.join(_HERE, '..'))  # …/leaphand

# ---------------------------------------------------------------------------
# Extend sys.path so all local packages are importable without pip install
# Order matters: insert before other paths so local versions take priority.
# ---------------------------------------------------------------------------
# 1. dex-retargeting  (D:\Jia\leaphand\dex-retargeting\src\dex_retargeting)
_DEX_RETARGETING_SRC = os.path.join(_ROOT, 'dex-retargeting', 'src')
if _DEX_RETARGETING_SRC not in sys.path:
    sys.path.insert(0, _DEX_RETARGETING_SRC)

# 2. XRoboToolkit-Teleop-Sample-Python
_XROBO_PATH = os.path.join(_ROOT, 'XRoboToolkit-Teleop-Sample-Python')
if _XROBO_PATH not in sys.path:
    sys.path.insert(0, _XROBO_PATH)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.dex_hand_utils import (
    DexHandTracker,
    pico_hand_state_to_mediapipe,
)
from dex_retargeting.constants import HandType, RetargetingType, RobotName


def leap_joint_limits_from_config(config: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Read LEAP 16-DOF joint limits (radians, LEAP API order) from an experiment TrainConfig.

    New task configs should define LEAP_JOINT_LOW and LEAP_JOINT_HIGH (each shape (16,)),
    same as passed to LeapHandWrapper in conrft_dex experiments.
    """
    low = getattr(config, "LEAP_JOINT_LOW", None)
    high = getattr(config, "LEAP_JOINT_HIGH", None)
    if low is None or high is None:
        return None, None
    lo = np.asarray(low, dtype=np.float32).reshape(-1)
    hi = np.asarray(high, dtype=np.float32).reshape(-1)
    if lo.size != 16 or hi.size != 16:
        print(
            f"[LEAP] WARN: LEAP_JOINT_LOW/HIGH must have 16 elements; got {lo.size}/{hi.size}. "
            "Ignoring limits."
        )
        return None, None
    return lo, hi


# ---------------------------------------------------------------------------
# Default paths derived from _ROOT (no manual configuration needed)
# ---------------------------------------------------------------------------
# dex-urdf LEAP Hand URDF  (used by both DexHandTracker and PyBullet visualization)
_DEX_URDF_ROOT  = os.path.join(_ROOT, 'dex-urdf', 'robots', 'hands')
_DEX_URDF_RIGHT = os.path.join(_DEX_URDF_ROOT, 'leap_hand', 'leap_hand_right.urdf')

# ---------------------------------------------------------------------------
# Hardware joint mapping  –  DEX_TO_LEAP_ORDER
# ---------------------------------------------------------------------------
# DexHandTracker.retarget() returns qpos whose indices correspond to the URDF
# joint ordering of  dex-urdf/robots/hands/leap_hand/leap_hand_{side}.urdf.
#
# LeapNode.set_leap() expects a fixed 16-element ordering:
#   [0..3]   Index  (MCP_side, MCP_flex, PIP, DIP)
#   [4..7]   Middle (MCP_side, MCP_flex, PIP, DIP)
#   [8..11]  Ring   (MCP_side, MCP_flex, PIP, DIP)
#   [12..15] Thumb  (CMC_side, CMC_flex, MCP,  DIP)
#
# HOW TO FILL IN THIS MAPPING
# ----------------------------
# 1. Run on hardware machine:
#        python pico_leap_dex.py --print-joints
#    The output lists all joint names with their index in the qpos vector:
#        [ 0]  <name>
#        [ 1]  <name>  ...
# 2. Cross-reference each name with the LEAP motor manual to find which
#    LEAP API position (0-15) it corresponds to.
# 3. Set DEX_TO_LEAP_ORDER_RIGHT so that DEX_TO_LEAP_ORDER_RIGHT[k] gives
#    the dex-retargeting qpos index for LEAP API position k.
#
DEX_TO_LEAP_ORDER_RIGHT = [
    1,  0,  2,  3,    
    9,  8, 10, 11,    
   13, 12, 14, 15,    
    4,  5,  6,  7     
]


# ---------------------------------------------------------------------------
# Hardware motion safety parameters
# ---------------------------------------------------------------------------
_HW_WARMUP_FRAMES = 10    # consecutive valid tracking frames before engaging hardware
_HW_MAX_DELTA     = 0.20  # max joint angle change per ~30 ms frame (rad, ≈ 5.7°)
_HW_INTERP_STEPS  = 60    # startup interpolation steps (≈ 1 s at 60 × 1/60 s)


_LEAP_MCP_EXTENDED_RAD = np.pi       # 3.14 rad = flat out for Index/Middle/Ring MCP
_LEAP_MCP_TOLERANCE_RAD = 0.6        

_LEAP_MCP_JOINT_NAMES = (            
    "Index_MCP_side", "Index_MCP_flex", "Index_PIP", "Index_DIP",
    "Middle_MCP_side", "Middle_MCP_flex", "Middle_PIP", "Middle_DIP",
    "Ring_MCP_side", "Ring_MCP_flex", "Ring_PIP", "Ring_DIP",
    "Thumb_CMC_side", "Thumb_CMC_flex", "Thumb_MCP", "Thumb_DIP",
)
_LEAP_MCP_INDICES = (0, 1, 2 ,3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13, 14, 15)  


_COMPARE_PRINT_EVERY = 10                 
_COMPARE_FOCUS_INDICES = (0, 4, 8, 15)    
# ---------------------------------------------------------------------------
# PyBullet joint mapping  –  DEX_TO_PYBULLET_ORDER
# ---------------------------------------------------------------------------
# This mapping is built automatically at runtime by matching joint names
# between dex-urdf URDF and PyBullet URDF.  If joint names don't match exactly,
# this will be None and a warning will be printed.
DEX_TO_PYBULLET_ORDER_RIGHT = None  # Auto-filled at runtime
DEX_TO_PYBULLET_ORDER_LEFT  = None  # Auto-filled at runtime


# ===========================================================================
# PyBulletLeapViz
# ===========================================================================
class PyBulletLeapViz:
    """
    Loads the LEAP Hand robot_pybullet.urdf into PyBullet and provides:
      • Joint animation from DexHandTracker qpos  (apply_dex_qpos)
      • Four coloured marker balls (update_target_vis_from_mediapipe)

    NOTE: p.connect() must be called before creating this object.
    """

    def __init__(self, dex_joint_names: list[str] | None = None):
        """
        Args:
            dex_joint_names: List of joint names from DexHandTracker (used to
                            build mapping to PyBullet joint indices).
        """
        # URDF file from dex-urdf (same as used by DexHandTracker for consistency)
        # Position offset: [-0.3, 0.0, 0.0] places the model away from origin for better viewing.
        # This translation sets the displayed model position in the PyBullet world.
        # The offset can be changed to [0, 0, 0] if you prefer the model at the origin.
        urdf = _DEX_URDF_RIGHT
        self.leap_id = p.loadURDF(
            urdf,
            [-0.3, 0.0, 0.0],                   # X-offset: -0.3m (purely for visualization)
            p.getQuaternionFromEuler([0, 0, 0]),  # Rotation: identity
            useFixedBase=True,
        )

        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

        # Build mapping: dex qpos index → PyBullet joint index
        self._dex_to_pybullet = self._build_joint_mapping(dex_joint_names)

    def _build_joint_mapping(self, dex_joint_names: list[str] | None) -> list[int] | None:
        
        """
        Build mapping from dex qpos indices to PyBullet joint indices by
        matching joint names.

        Returns:
            List of PyBullet joint indices, or None if mapping fails.
        """
        if dex_joint_names is None:
            return None

        # Query PyBullet model joint names
        num_joints = p.getNumJoints(self.leap_id)
        pybullet_joint_names = {}
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.leap_id, i)
            joint_name = joint_info[1].decode('utf-8') if isinstance(joint_info[1], bytes) else joint_info[1]
            if joint_name:  # Skip empty names (fixed joints)
                pybullet_joint_names[joint_name] = i

        # Match dex joint names to PyBullet joint indices
        mapping = []
        for dex_idx, dex_name in enumerate(dex_joint_names):
            if dex_name in pybullet_joint_names:
                mapping.append(pybullet_joint_names[dex_name])
            else:
                print(f"[WARNING] PyBullet joint '{dex_name}' (dex index {dex_idx}) not found in model.")
                return None  # Fail if any joint doesn't match

        print(f"[PyBullet] Built joint mapping: {len(mapping)} joints matched.")
        return mapping


    def apply_dex_qpos(self, dex_qpos: np.ndarray):
        """
        Apply DexHandTracker qpos to PyBullet model joints.

        Args:
            dex_qpos: (N,) Joint angles from DexHandTracker (dex-urdf order).
        """
        if self._dex_to_pybullet is None:
            print("[WARNING] PyBullet joint mapping not available – skipping visualization update.")
            return

        p.stepSimulation()

        # Apply each dex qpos to corresponding PyBullet joint
        for dex_idx, pybullet_idx in enumerate(self._dex_to_pybullet):
            if dex_idx < len(dex_qpos):
                p.setJointMotorControl2(
                    bodyIndex=self.leap_id,
                    jointIndex=pybullet_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=float(dex_qpos[dex_idx]),
                    targetVelocity=0,
                    force=500,
                    positionGain=0.3,
                    velocityGain=1,
                )

    def get_dex_joint_positions(self) -> list[float] | None:
        """
        Read current PyBullet joint positions in dex_qpos ordering.
        """
        if self._dex_to_pybullet is None:
            return None
        sim_q = []
        for pybullet_idx in self._dex_to_pybullet:
            joint_state = p.getJointState(self.leap_id, pybullet_idx)
            sim_q.append(float(joint_state[0]))
        return sim_q


# ===========================================================================
# PicoLeapDexController
# ===========================================================================
class PicoLeapDexController:
    """
    PICO 4 Ultra → LEAP Hand teleoperation controller.

    Unified pipeline (visualization and hardware use same retargeting):
    ------------------------------------------------------------------
    PICO hand state  (27×7)
        │
        └─ pico_hand_state_to_mediapipe()  → (21, 3) wrist-centred
           → DexHandTracker.retarget()     → filtered qpos (dex ordering)
              │
              ├─ [Visualization]  (optional: use_pybullet=True)
              │    → PyBulletLeapViz.apply_dex_qpos()      (animates PyBullet model)
              │    → PyBulletLeapViz.update_target_vis()   (moves 4 coloured balls)
              │
              └─ [Hardware]
                  → _apply_dex_to_leap_mapping()  → 16-DOF LEAP API order
                  → LeapNode.set_leap()           (only when --hardware)
    """


    def __init__(
        self,
        gui: bool = True,
        use_hardware: bool = False,
        urdf_path: str | None = None,
        retargeting_type: RetargetingType = RetargetingType.vector,
        print_joints: bool = False,
        use_pybullet: bool = True,
        leap_joint_low: np.ndarray | None = None,
        leap_joint_high: np.ndarray | None = None,
    ):
        self.hand_type = "right"
        self._leap_joint_low: np.ndarray | None = None
        self._leap_joint_high: np.ndarray | None = None
        if leap_joint_low is not None and leap_joint_high is not None:
            lo = np.asarray(leap_joint_low, dtype=np.float32).reshape(16)
            hi = np.asarray(leap_joint_high, dtype=np.float32).reshape(16)
            self._leap_joint_low = lo
            self._leap_joint_high = hi
            print("[LEAP] Per-joint command limits enabled (16 DOF, from experiment config).")

        # ---- PICO client ---------------------------------------------------
        self.xr_client = XrClient()

        # ---- DexHandTracker ------------------------------------------------
        if urdf_path is None:
            urdf_path = _DEX_URDF_RIGHT
        self.dex_tracker = DexHandTracker(
            robot_name=RobotName.leap,
            urdf_path=urdf_path,
            retargeting_type=retargeting_type,
            hand_type=HandType.right,
        )

        if print_joints:
            print("[dex-retargeting] joint names in qpos order:")
            for i, name in enumerate(self.dex_tracker.retargeting.joint_names):
                print(f"  [{i:2d}]  {name}")
            print()
            print("Fill in DEX_TO_LEAP_ORDER_RIGHT at the top of this file.")

        # ---- PyBullet (optional; skip entirely for hardware-only / TF co-process) ----
        self.viz: PyBulletLeapViz | None = None
        if use_pybullet:
            if gui:
                p.connect(p.GUI)
                dex_joint_names = (
                    self.dex_tracker.retargeting.joint_names
                    if hasattr(self.dex_tracker, "retargeting")
                    and hasattr(self.dex_tracker.retargeting, "joint_names")
                    else None
                )
                self.viz = PyBulletLeapViz(dex_joint_names=dex_joint_names)
            else:
                p.connect(p.DIRECT)  # headless; PyBulletLeapViz not created
        elif gui:
            print(
                "[PICO] PicoLeapDexController: use_pybullet=False — ignoring gui=True "
                "(no PyBullet connection)."
            )

        # ---- Hardware (optional) ------------------------------------------
        self.leap_node = None
        
        # ---- Motion-safety state (used only when hardware is enabled) ----
        self._hw_prev_q: list | None = None  # last commanded position, for delta clipping
        self._stable_frames: int = 0          # consecutive valid-tracking frame count
        self._hw_ready: bool = False          # True after startup interpolation completes
        self._hw_set_leap_count: int = 0
        self._hw_extended_check_done: bool = False  
        self._compare_counter: int = 0              
        if use_hardware:
            if not _LEAPNODE_AVAILABLE:
                raise RuntimeError(
                    "--hardware requested but leaphand_rw is not installed.\n"
                    "Install it on the hardware machine:\n"
                    "  pip install -e LEAP_Hand_API/python"
                )
            self.leap_node = LeapNode()
            print("[LEAP] Hardware node initialised.")


        # Joint-order mapping for right hand
        self._dex_to_leap = DEX_TO_LEAP_ORDER_RIGHT

    def _clip_leap_q_list(self, leap_q: list) -> list:
        """Clip 16-DOF LEAP commands to experiment config limits (no-op if unset)."""
        if self._leap_joint_low is None or self._leap_joint_high is None:
            return leap_q
        arr = np.clip(
            np.asarray(leap_q, dtype=np.float32).reshape(16),
            self._leap_joint_low,
            self._leap_joint_high,
        )
        return arr.tolist()

    
    # -----------------------------------------------------------------------
    # Helper: startup interpolation  (hardware path)
    # -----------------------------------------------------------------------
    def _startup_interpolate(self, target_q: list) -> None:
        """
        Smoothly move from the hardware's current joint positions to target_q
        over _HW_INTERP_STEPS steps (≈ 1 second).

        Falls back to all-zeros as the start position if read_pos() fails.
        """
        print("[LEAP] _startup_interpolate target_q =", [round(x, 3) for x in target_q])
        try:
            current_q = list(self.leap_node.read_pos())
        except Exception:
            current_q = [0.0] * 16
            print("[LEAP] Could not read current position – interpolating from zeros.")

        current_arr = np.array(current_q, dtype=float)
        target_arr = np.array(self._clip_leap_q_list(list(target_q)), dtype=float)
        dt = 1.0 / _HW_INTERP_STEPS
        for i in range(_HW_INTERP_STEPS):
            alpha    = (i + 1) / _HW_INTERP_STEPS
            interp_q = self._clip_leap_q_list(
                list(current_arr + alpha * (target_arr - current_arr))
            )
            self.leap_node.set_leap(interp_q)
            time.sleep(dt)
        print("[LEAP] Startup interpolation complete.")

    # -----------------------------------------------------------------------
    # Helper: dex-retargeting qpos → LeapNode 16-DOF order  (hardware path)
    # -----------------------------------------------------------------------
    def _apply_dex_to_leap_mapping(self, qpos: np.ndarray) -> list:
        """
        Reorders dex-retargeting qpos to the 16-DOF order expected by
        LeapNode.set_leap().

        If DEX_TO_LEAP_ORDER is still None (not yet filled in), the raw qpos
        is forwarded with a warning.  Use --print-joints to determine the
        correct mapping before running on hardware.
        """
        if self._dex_to_leap is None:
            print(
                "[WARNING] DEX_TO_LEAP_ORDER not set – forwarding raw dex qpos to hardware.\n"
                "          Run with --print-joints and fill in DEX_TO_LEAP_ORDER_RIGHT/LEFT."
            )
            return qpos.tolist()
        leap_q = [float(qpos[i]) for i in self._dex_to_leap]
        return self._apply_leap_mcp_offset(leap_q)

    # -----------------------------------------------------------------------
    
    # -----------------------------------------------------------------------
    def _apply_leap_mcp_offset(self, leap_q: list) -> list:
        """
        Convert straight-joint URDF angles (approximately zero) to the LEAP convention with +pi
        offsets for all joints.
        """
        out = list(leap_q)
        for idx in _LEAP_MCP_INDICES:
            if idx < len(out):
                delta = _LEAP_MCP_EXTENDED_RAD
                out[idx] = out[idx] + delta
        return out

    # -----------------------------------------------------------------------
    
    # -----------------------------------------------------------------------
    def _check_leap_extended_pose(self, leap_q: list) -> None:
        """
        Check the mapped pose before sending commands. Straight MCP joints use approximately pi
        radians in the LEAP convention; missing offsets can cause extreme motor motion.
        """
        if self._hw_extended_check_done or len(leap_q) < 16:
            return
        self._hw_extended_check_done = True

        tol = _LEAP_MCP_TOLERANCE_RAD
        low_pi = _LEAP_MCP_EXTENDED_RAD - tol
        high_pi = _LEAP_MCP_EXTENDED_RAD + tol
        
        print("[LEAP] -------- Extended-pose check (all ~3.14 rad) --------")
        all_ok = True
        for idx in _LEAP_MCP_INDICES:
            val = leap_q[idx]
            exp = _LEAP_MCP_EXTENDED_RAD
            ok = low_pi <= val <= high_pi
            if not ok:
                all_ok = False
            status = "OK" if ok else "DEVIATION"
            name = _LEAP_MCP_JOINT_NAMES[idx] if idx < len(_LEAP_MCP_JOINT_NAMES) else f"joint_{idx}"
            print(f"  LEAP[{idx:2d}] {name:18s}  expected ~{exp:.2f}  actual={val:.3f}  [{status}]")
        print("[LEAP] -------------------------------------------------------------------")
        if not all_ok:
            print(
                "[LEAP] WARNING: Some MCP values are far from 3.14 rad. If the real hand looks\n"
                "       twisted/locked, add an offset (e.g. +π) for these joints in the\n"
                "       dex→LEAP mapping or use LEAP_Hand_API allegro_to_LEAPhand()."
            )
        print()

    def _print_joint_comparison(
        self,
        cmd_leap_q: list,
        sim_leap_q: list | None,
        hw_leap_q: list | None,
    ) -> None:
        """
        Print command/simulation/hardware comparison for each LEAP joint.
        """
        print(f"[CMP] frame={self._compare_counter}  (cmd/sim/hw in rad)")
        print("      idx   cmd      sim      hw     sim-cmd   hw-cmd")
        for i in range(min(16, len(cmd_leap_q))):
            cmd = float(cmd_leap_q[i])
            sim = float(sim_leap_q[i]) if sim_leap_q is not None and i < len(sim_leap_q) else float("nan")
            hw = float(hw_leap_q[i]) if hw_leap_q is not None and i < len(hw_leap_q) else float("nan")
            sim_err = sim - cmd if not np.isnan(sim) else float("nan")
            hw_err = hw - cmd if not np.isnan(hw) else float("nan")
            print(f"      {i:2d}  {cmd:7.3f}  {sim:7.3f}  {hw:7.3f}   {sim_err:7.3f}  {hw_err:7.3f}")

        focus_parts = []
        for idx in _COMPARE_FOCUS_INDICES:
            if idx >= len(cmd_leap_q):
                continue
            cmd = float(cmd_leap_q[idx])
            hw = float(hw_leap_q[idx]) if hw_leap_q is not None and idx < len(hw_leap_q) else float("nan")
            hw_err = hw - cmd if not np.isnan(hw) else float("nan")
            focus_parts.append(f"J{idx}: cmd={cmd:.3f}, hw={hw:.3f}, err={hw_err:.3f}")
        if focus_parts:
            print("[CMP] Focus:", " | ".join(focus_parts))
        print()

    # -----------------------------------------------------------------------
    # Main per-frame step
    # -----------------------------------------------------------------------
    def step(self, verbose: bool = False) -> dict | None:
        """
        Execute one teleoperation frame.

        Returns a dict with keys (when available):
          'dex_qpos'    – np.ndarray from DexHandTracker (dex-urdf order, used for both viz and hw)
          'leap_q'      – 16-DOF list sent to LeapNode   (if hardware enabled)
        Returns None if tracking data is unavailable or retargeting fails.
        """
        hand_state = self.xr_client.get_hand_tracking_state(self.hand_type)
        if hand_state is None:
            if verbose:
                print("[PICO] No tracking data this frame.")
            self._stable_frames = 0
            return None

        hand_state = np.asarray(hand_state, dtype=float)
        if np.all(hand_state == 0):
            if verbose:
                print("[PICO] All-zero state – skipping frame.")
            self._stable_frames = 0
            return None

        result: dict = {}

        # ---- Step 1: Convert PICO → MediaPipe format ----------------------
        # pico_hand_state_to_mediapipe centres positions at the wrist and
        # maps PICO indices → MediaPipe indices.
        
        mediapipe_pos = pico_hand_state_to_mediapipe(hand_state)   # (21, 3)

        # ---- Step 2: DexHandTracker retargeting (used for both viz and hw) --
        
        dex_qpos = self.dex_tracker.retarget(mediapipe_pos)   # (N,) or None

        if dex_qpos is None:
            if verbose:
                print("[DexHandTracker] retarget() returned None – skipping.")
            self._stable_frames = 0
            return None

        result['dex_qpos'] = dex_qpos
        if verbose:
            print("[DexHandTracker] qpos =", np.round(dex_qpos, 3))

        # ---- Step 3: Apply to PyBullet visualization -----------------------
        
        if self.viz is not None:
            self.viz.apply_dex_qpos(dex_qpos)
            # Update marker balls from MediaPipe positions - DISABLED
            # Base offset matches the URDF loading position (purely cosmetic)

        # ---- Step 4: Send to hardware --------------------------------------
        if self.leap_node is not None:
            leap_q = self._clip_leap_q_list(self._apply_dex_to_leap_mapping(dex_qpos))

            
            # 4a. Wait for N consecutive stable tracking frames before engaging.
            #     Resets to 0 whenever tracking is lost (see None-return paths above).
            self._stable_frames += 1
            if self._stable_frames < _HW_WARMUP_FRAMES:
                if verbose or self._stable_frames == 1:
                    print(f"[LEAP] Waiting for stable tracking "
                          f"({self._stable_frames}/{_HW_WARMUP_FRAMES}) ...")
                result['leap_q'] = leap_q
                return result

            # 4b. First engagement after stable tracking: check extended-pose convention,
            #     then smoothly interpolate from hardware's current position to first target.
            if not self._hw_ready:
                self._check_leap_extended_pose(leap_q)
                print("[LEAP] Engaging hardware – interpolating to first pose ...")
                self._startup_interpolate(leap_q)
                self._hw_prev_q = leap_q[:]
                self._hw_ready = True

            # 4c. Normal operation: clip per-frame delta to prevent sudden jumps.
            #     Also handles re-engagement after momentary tracking loss.
            if self._hw_prev_q is not None:
                prev  = np.array(self._hw_prev_q)
                curr  = np.array(leap_q)
                delta = np.clip(curr - prev, -_HW_MAX_DELTA, _HW_MAX_DELTA)
                leap_q = self._clip_leap_q_list(list(prev + delta))

            if self._hw_set_leap_count < 5:
                print(f"[LEAP] set_leap (frame {self._hw_set_leap_count}) leap_q =", [round(v, 3) for v in leap_q])
            self._hw_set_leap_count += 1

            self.leap_node.set_leap(leap_q)
            self._hw_prev_q = leap_q[:]
            result['leap_q'] = leap_q
            if verbose:
                print("[LeapNode]       q   =", [round(v, 3) for v in leap_q])
            

        return result


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PICO 4 Ultra → LEAP Hand teleop (DexHandTracker + PyBullet viz)"
    )
    parser.add_argument("--no-gui",        action="store_true",
                        help="Disable PyBullet GUI (headless mode)")
    parser.add_argument(
        "--no-pybullet",
        action="store_true",
        help="Do not connect PyBullet at all (PICO + Dex retargeting + hardware only).",
    )
    parser.add_argument("--hardware",      action="store_true",
                        help="Enable LeapNode to control real LEAP Hand hardware")
    parser.add_argument("--urdf-path",     type=str, default=None,
                        help=f"Path to leap_hand_right.urdf from dex-urdf. Default: {_DEX_URDF_RIGHT}")
    parser.add_argument("--retargeting",   type=str, default="vector",
                        choices=["vector"],
                        help="dex-retargeting algorithm type (default: vector)")
    parser.add_argument("--print-joints",  action="store_true",
                        help="Print dex-retargeting joint names at startup, "
                             "then continue running. Use this to determine the "
                             "DEX_TO_LEAP_ORDER mapping before hardware deployment.")
    parser.add_argument("--verbose",       action="store_true",
                        help="Print per-frame debug output")
    args = parser.parse_args()

    retargeting_type = (
        RetargetingType.dexpilot if args.retargeting == "dexpilot"
        else RetargetingType.vector
    )

    print("=" * 60)
    print("PICO → LEAP Hand teleoperation  (Right hand only)")
    use_pybullet = not args.no_pybullet
    print(f"  GUI         : {not args.no_gui}")
    print(f"  PyBullet    : {use_pybullet}")
    print(f"  Hardware    : {args.hardware}")
    print(f"  Retargeting : {args.retargeting}")
    print("=" * 60)

    controller = PicoLeapDexController(
        gui=not args.no_gui,
        use_hardware=args.hardware,
        urdf_path=args.urdf_path,
        retargeting_type=retargeting_type,
        print_joints=args.print_joints,
        use_pybullet=use_pybullet,
    )

    no_data_count = 0
    while True:
        result = controller.step(verbose=args.verbose)

        if result is None:
            no_data_count += 1
            if no_data_count % 50 == 1:
                print("[PICO] Waiting for hand tracking …")
        else:
            no_data_count = 0
            if not args.verbose:
                dex_q = result.get('dex_qpos')

        time.sleep(0.03)