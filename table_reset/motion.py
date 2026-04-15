"""
Robot motion primitives for the Franka (DROID) reset policy.

Provides low-level motion execution and high-level pick-and-place sequences.
Extracted and extended from anygrasp_utils/main_eval.py.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from table_reset.config import (
    TOP_POSE, Z_SAFETY_FLOOR, MOTION_HZ,
    GRASP_CLOSE_DURATION, GRASP_OPEN_DURATION,
    LIFT_DURATION, MOVE_DURATION,
)
from table_reset.utils import (
    normalize_rpy_to_reference, canonicalize_yaw,
    rpy_to_matrix, interpolate_poses as _interpolate,
)


# =============================================================================
# Low-level motion
# =============================================================================

def run(env, pose6d, duration=1.0, grip_close=False, hz=MOTION_HZ):
    """Send a constant 7D action to the robot for a given duration.

    Args:
        env: DROID RobotEnv instance
        pose6d: [x, y, z, roll, pitch, yaw]
        duration: seconds to hold this command
        grip_close: True=close gripper, False=open
        hz: control loop frequency
    """
    pose = np.array(pose6d, dtype=np.float32)
    grip = np.array([1.0 if grip_close else 0.0], dtype=np.float32)
    action = np.concatenate([pose, grip])
    for _ in range(int(duration * hz)):
        env.step(action)
        time.sleep(1.0 / hz)


def interpolate_poses(pose1, pose2, resolution, euler_axes='xyz'):
    """Interpolate between two 6D poses with LERP (position) + SLERP (rotation).

    Args:
        pose1: start pose [x, y, z, roll, pitch, yaw]
        pose2: end pose [x, y, z, roll, pitch, yaw]
        resolution: number of intermediate poses (including start and end)

    Returns:
        np.ndarray of shape (resolution, 6)
    """
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)

    pos1, rpy1 = pose1[:3], pose1[3:]
    pos2, rpy2 = pose2[:3], pose2[3:]

    times = np.linspace(0, 1, resolution)
    positions = (1 - times[:, None]) * pos1 + times[:, None] * pos2

    key_rots = Rot.from_euler(euler_axes, [rpy1, rpy2])
    slerp = Slerp([0, 1], key_rots)
    rotations = slerp(times).as_euler(euler_axes)

    return np.hstack((positions, rotations))


def waypoint_move(env, start_pose, target_pose, steps=30,
                  grip_close=False, step_duration=0.15):
    """Smooth waypoint motion from start to target with SLERP rotation.

    Args:
        env: DROID RobotEnv
        start_pose: [x,y,z,r,p,y] current pose
        target_pose: [x,y,z,r,p,y] goal pose
        steps: number of interpolation steps
        grip_close: gripper state during motion
        step_duration: time per waypoint
    """
    inter = interpolate_poses(start_pose, target_pose, steps)
    for pose in inter:
        run(env, pose.tolist(), duration=step_duration,
            grip_close=grip_close, hz=MOTION_HZ)


# =============================================================================
# High-level motion sequences
# =============================================================================

def move_to_top(env, grip_close=False):
    """Return to the observation / safe pose."""
    run(env, TOP_POSE, duration=MOVE_DURATION, grip_close=grip_close)


def pick(env, grasp_pose6d, approach_height=0.10):
    """Execute a pick sequence: approach -> descend -> grasp -> lift.

    Args:
        env: DROID RobotEnv
        grasp_pose6d: [x, y, z, roll, pitch, yaw] in base frame
        approach_height: height above grasp to start descent (m)

    Returns:
        True if sequence completed, False if aborted due to safety
    """
    grasp = np.array(grasp_pose6d, dtype=float)

    # Normalize RPY relative to top_pose for shortest path
    rpy = normalize_rpy_to_reference(TOP_POSE[3:6], grasp[3:6])
    rpy[2] = canonicalize_yaw(rpy[2])
    grasp[3:6] = rpy

    # Safety: enforce minimum z
    grasp[2] = max(grasp[2], Z_SAFETY_FLOOR)

    # 1. Approach pose: above the grasp, same xy and orientation
    approach = grasp.copy()
    approach[2] = grasp[2] + approach_height

    # Move from top_pose to approach (lateral + rotation)
    top_to_approach = TOP_POSE.copy()
    top_to_approach[0:2] = approach[0:2]
    top_to_approach[3:6] = rpy
    waypoint_move(env, TOP_POSE, top_to_approach, steps=30,
                  grip_close=False, step_duration=0.15)

    # 2. Descend to grasp
    descent = interpolate_poses(top_to_approach, grasp.tolist(), 20)
    for i, pose in enumerate(descent):
        pose = np.array([float(v) for v in pose])
        if pose[2] < Z_SAFETY_FLOOR:
            print(f"[SAFETY] z={pose[2]:.3f} < {Z_SAFETY_FLOOR}. Aborting pick.")
            move_to_top(env, grip_close=False)
            return False
        dur = 0.5 if i == 0 else 0.25
        run(env, pose, duration=dur, grip_close=False, hz=MOTION_HZ)

    # 3. Close gripper
    run(env, grasp.tolist(), duration=GRASP_CLOSE_DURATION, grip_close=True)

    # 4. Lift back to top
    run(env, TOP_POSE, duration=LIFT_DURATION, grip_close=True)

    return True


def place(env, target_xyz, place_height=0.05):
    """Execute a place sequence: move above target -> descend -> release -> lift.

    Assumes the robot is currently at TOP_POSE holding an object.

    Args:
        env: DROID RobotEnv
        target_xyz: [x, y, z] target placement position in base frame
        place_height: extra height above target z for release
    """
    target = np.array(target_xyz, dtype=float)

    # Use top_pose orientation (straight down)
    place_pose = list(TOP_POSE)
    place_pose[0] = target[0]
    place_pose[1] = target[1]
    place_pose[2] = target[2] + place_height

    # Move above target
    above_target = list(TOP_POSE)
    above_target[0] = target[0]
    above_target[1] = target[1]
    waypoint_move(env, TOP_POSE, above_target, steps=20,
                  grip_close=True, step_duration=0.15)

    # Descend
    descent = interpolate_poses(above_target, place_pose, 15)
    for pose in descent:
        run(env, pose.tolist(), duration=0.25, grip_close=True, hz=MOTION_HZ)

    # Release
    run(env, place_pose, duration=GRASP_OPEN_DURATION, grip_close=False)

    # Lift back to top
    move_to_top(env, grip_close=False)


def pick_and_place(env, grasp_pose6d, target_xyz):
    """Full pick-and-place cycle.

    Args:
        env: DROID RobotEnv
        grasp_pose6d: [x,y,z,r,p,y] grasp in base frame
        target_xyz: [x,y,z] placement target in base frame

    Returns:
        True if completed, False if pick failed
    """
    success = pick(env, grasp_pose6d)
    if not success:
        return False
    place(env, target_xyz)
    return True
