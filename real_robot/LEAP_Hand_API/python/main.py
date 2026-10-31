import os

import numpy as np
import argparse

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time
#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second.  Used the combined commands if you can to save time.  Also don't forget about the USB latency settings in the readme.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more in radians.

#LEAP hand conventions:
#3.14 rad is flat out home pose for the index, middle, ring, finger MCPs.
#Applying a positive angle closes the joints more and more to curl closed in radians.
#The MCP is centered at 3.14 and can move positive or negative to that in radians.

#The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
#For instance, the MCP Side of Index is ID 0, the MCP Forward of Ring is 9, the DIP of Ring is 11

"""
########################################################
class LeapNode:
    def __init__(self, serial_port=None, baudrate=4000000):
        ####Some parameters
        # I recommend you keep the current limit from 350 for the lite, and 550 for the full hand
        # Increase KP if the hand is too weak, decrease if it's jittery.
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350  ##set this to 550 if you are using full motors!!!!
        home = np.array(lhu.allegro_to_LEAPhand(np.zeros(16)), dtype=float)
        self.prev_pos = self.pos = self.curr_pos = home
        # Serial port: pass serial_port=..., or export LEAP_SERIAL_PORT=/dev/ttyUSB0
        # (ls /dev/serial/by-id/* for stable FTDI paths). Default /dev/ttyUSB0.
        if serial_port is None or str(serial_port).strip() == "":
            serial_port = os.environ.get("LEAP_SERIAL_PORT", "/dev/ttyUSB0")
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        print(f"[LeapNode] Dynamixel port={serial_port} baud={baudrate}")
        self.dxl_client = DynamixelClient(motors, serial_port, baudrate)
        self.dxl_client.connect()

        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility joint angles.  It adds 180 to make the fully open position at 0 instead of 180
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position of the robot
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()
    #These combined commands are faster FYI and return a list of data
    def pos_vel(self):
        return self.dxl_client.read_pos_vel()
    #These combined commands are faster FYI and return a list of data
    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()
        
def main(**kwargs):
    parser = argparse.ArgumentParser(
        description="LEAP Hand API quick test / joint calibration helper"
    )
    parser.add_argument(
        "--mode",
        choices=["hold_open", "hold_home", "joint_test", "joint8_cycle", "open_then_j8"],
        default="hold_open",
        help="hold_open: default open; hold_home: keep __init__ home pose; "
             "joint_test: sweep; joint8_cycle: cycle; "
             "open_then_j8: API open hold then override J8 only (hardware zero test)"
    )
    parser.add_argument(
        "--joints",
        type=str,
        default="0,4,8,15",
        help="Comma-separated joint ids for joint_test mode (default: 0,4,8,15)"
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=0.20,
        help="Sweep amplitude around center in rad (default: 0.20)"
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=0.6,
        help="Hold time per target in seconds (default: 0.6)"
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=30.0,
        help="Sampling rate for prints during hold in Hz (default: 30)"
    )
    parser.add_argument(
        "--cycle-interval",
        type=float,
        default=1.0,
        help="Seconds per step for joint8_cycle mode (default: 1.0)"
    )
    parser.add_argument(
        "--cycle-joint",
        type=int,
        default=8,
        help="Joint id for joint8_cycle mode, e.g. 15 for Thumb_DIP (default: 8)"
    )
    parser.add_argument(
        "--cycle-amp-left",
        type=float,
        default=0.3,
        help="joint8_cycle: amplitude left of center in rad (default: 0.3)"
    )
    parser.add_argument(
        "--cycle-amp-right",
        type=float,
        default=0.2,
        help="joint8_cycle: amplitude right of center in rad (default: 0.2)"
    )
    parser.add_argument(
        "--phase1-sec",
        type=float,
        default=3.0,
        help="open_then_j8: seconds to hold standard API open (cmd[J8]≈π) before override; use 0 to skip straight to --j8-target (default: 3)"
    )
    parser.add_argument(
        "--j8-target",
        type=float,
        default=1.9,
        help="open_then_j8: LEAP rad for joint 8 after phase 1 (default: 1.9)"
    )
    parser.add_argument(
        "--override-joint",
        type=int,
        default=8,
        help="open_then_j8: which joint index to override (default: 8)"
    )
    args = parser.parse_args()

    leap_hand = LeapNode()

    if args.mode == "hold_open":
        while True:
            # Set to an open pose and read the joint angles.
            leap_hand.set_allegro(np.zeros(16))
            print("Position: " + str(leap_hand.read_pos()))
            time.sleep(0.03)
        return
    
    if args.mode == "hold_home":
        home = np.array(leap_hand.curr_pos, dtype=float)
        print("=" * 72)
        print("hold_home: keep LeapNode.__init__ home pose")
        print(f"  home[8] = {home[8]:.4f} rad")
        print("=" * 72)
        while True:
            leap_hand.set_leap(home.tolist())
            fb = np.array(leap_hand.read_pos(), dtype=float)
            err = fb - home
            print(
                f"[hold_home] J08 cmd={home[8]:.4f} fb={fb[8]:.4f} err={err[8]:+.4f}"
            )
            print("  cmd: " + str(np.round(home, 4)))
            print("  fb : " + str(np.round(fb, 4)))
            print("  err: " + str(np.round(err, 4)))
            time.sleep(0.03)
        return

    if args.mode == "open_then_j8":
        if args.override_joint < 0 or args.override_joint > 15:
            raise ValueError(f"--override-joint out of range [0, 15]: {args.override_joint}")
        jid = int(args.override_joint)
        
        open_pose = np.array(lhu.allegro_to_LEAPhand(np.zeros(16)), dtype=float)
        print("=" * 72)
        print("open_then_j8: compare nominal API open vs your mechanical zero on J8")
        print(f"  Phase 1: hold standard open ({args.phase1_sec:.1f} s)")
        print(f"    cmd[{jid}] from allegro_to_LEAPhand(0): {open_pose[jid]:.4f} rad")
        print(f"  Phase 2: same pose but cmd[{jid}] := {args.j8_target:.4f} rad")
        print("=" * 72)

        leap_hand.set_leap(open_pose.tolist())
        # phase1_sec may be 0 → skip straight to Phase 2 (only J8 overridden)
        phase1 = max(float(args.phase1_sec), 0.0)
        if phase1 <= 0:
            print("[P1] skipped (--phase1-sec 0); going straight to Phase 2")
        t_end = time.time() + phase1
        while time.time() < t_end:
            pos = np.array(leap_hand.read_pos(), dtype=float)
            rem = t_end - time.time()
            print(
                f"[P1 open] J{jid:02d} cmd={open_pose[jid]:.3f} pos={pos[jid]:.3f} "
                f"err={pos[jid]-open_pose[jid]:+.3f}  | hold {rem:.1f}s"
            )
            time.sleep(0.2)

        modified = open_pose.copy()
        modified[jid] = float(args.j8_target)
        leap_hand.set_leap(modified.tolist())
        print(f"[P2] switched to override: cmd[{jid}]={modified[jid]:.4f} (printing @ 5 Hz, Ctrl+C stop)")
        while True:
            pos = np.array(leap_hand.read_pos(), dtype=float)
            err = float(pos[jid] - modified[jid])
            print(
                f"[P2 hold] J{jid:02d} cmd={modified[jid]:.3f} pos={pos[jid]:.3f} err={err:+.3f}"
            )
            time.sleep(0.2)

    if args.mode == "joint8_cycle":
        if args.cycle_joint < 0 or args.cycle_joint > 15:
            raise ValueError(f"--cycle-joint out of range [0, 15]: {args.cycle_joint}")
        baseline = np.array(leap_hand.read_pos(), dtype=float)
        jid = int(args.cycle_joint)
        center = float(baseline[jid])
        al = max(0.0, float(args.cycle_amp_left))
        ar = max(0.0, float(args.cycle_amp_right))
        sequence = [center, center - al, center, center + ar, center]

        print("=" * 72)
        print(f"Joint {jid} cycle mode (center → left → center → right → center)")
        print(f"  center(J{jid:02d})  : {center:.3f} rad")
        print(f"  amp left/right     : {al:.3f} / {ar:.3f} rad")
        print(f"  sequence           : {[round(v, 3) for v in sequence]}")
        print(f"  interval    : {args.cycle_interval:.3f} s per step")
        print("=" * 72)

        cmd = baseline.copy()
        step = 0
        while True:
            target = sequence[step % len(sequence)]
            cmd[jid] = target
            leap_hand.set_leap(cmd)

            t_end = time.time() + max(args.cycle_interval, 0.05)
            while time.time() < t_end:
                pos = np.array(leap_hand.read_pos(), dtype=float)
                err = float(pos[jid] - target)
                print(f"[J{jid:02d}-cycle] step={step:04d}  cmd={target:.3f}  pos={pos[jid]:.3f}  err={err:+.3f}")
                time.sleep(0.1)
            step += 1
        return

    # joint_test mode
    try:
        joints = [int(x.strip()) for x in args.joints.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"Invalid --joints value: {args.joints}")
    for jid in joints:
        if jid < 0 or jid > 15:
            raise ValueError(f"Joint id out of range [0, 15]: {jid}")

    # Start from current measured pose so the test is gentle.
    baseline = np.array(leap_hand.read_pos(), dtype=float)
    targets_by_joint = {}
    for jid in joints:
        c = float(baseline[jid])
        targets_by_joint[jid] = [c - args.amp, c, c + args.amp, c]

    dt = 1.0 / max(args.print_rate, 1.0)
    print("=" * 72)
    print("Joint test mode")
    print(f"  joints     : {joints}")
    print(f"  baseline   : {[round(float(baseline[j]), 3) for j in joints]}")
    print(f"  amplitude  : ±{args.amp:.3f} rad")
    print(f"  dwell      : {args.dwell:.3f} s each target")
    print(f"  print_rate : {args.print_rate:.1f} Hz")
    print("=" * 72)

    for jid in joints:
        print()
        print(f"[TEST] Joint {jid} sweep start")
        cmd = baseline.copy()
        for target in targets_by_joint[jid]:
            cmd[jid] = target
            leap_hand.set_leap(cmd)
            t_end = time.time() + args.dwell
            while time.time() < t_end:
                pos = np.array(leap_hand.read_pos(), dtype=float)
                cur = np.array(leap_hand.read_cur(), dtype=float)
                err = float(pos[jid] - cmd[jid])
                print(
                    f"[J{jid:02d}] cmd={cmd[jid]:.3f}  "
                    f"pos={pos[jid]:.3f}  err={err:+.3f}  "
                    f"cur={cur[jid]:+.3f}"
                )
                time.sleep(dt)
        # Return tested joint to its original baseline value.
        cmd[jid] = baseline[jid]
        leap_hand.set_leap(cmd)
        time.sleep(0.3)
        print(f"[TEST] Joint {jid} sweep done")

    print()
    print("[TEST] All requested joints complete. Holding open pose (Ctrl+C stop)...")
    while True:
        leap_hand.set_allegro(np.zeros(16))
        print("Position: " + str(leap_hand.read_pos()))
        time.sleep(0.1)

if __name__ == "__main__":
    main()
