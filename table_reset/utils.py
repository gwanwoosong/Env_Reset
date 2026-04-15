"""
Transform utilities and coordinate conversion helpers.

Extracted and cleaned from anygrasp_utils/main_eval.py.
All transforms follow the convention: T_A_B means "B expressed in A's frame".
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def make_T(R, t):
    """Build a 4x4 homogeneous transform from rotation matrix and translation.

    Args:
        R: (3,3) rotation matrix
        t: (3,) or (3,1) translation vector

    Returns:
        (4,4) homogeneous transformation matrix
    """
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def pose6d_to_matrix(pose6d):
    """Convert [x, y, z, roll, pitch, yaw] to 4x4 homogeneous matrix.

    Euler convention: extrinsic 'xyz' (roll about X, pitch about Y, yaw about Z).
    """
    x, y, z, roll, pitch, yaw = pose6d
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose6d(T):
    """Convert 4x4 homogeneous matrix to [x, y, z, roll, pitch, yaw].

    Returns:
        np.ndarray of shape (6,)
    """
    xyz = T[:3, 3]
    rpy = Rot.from_matrix(T[:3, :3]).as_euler('xyz')
    return np.concatenate([xyz, rpy])


def rpy_to_matrix(roll, pitch, yaw):
    """Convert roll, pitch, yaw to 3x3 rotation matrix (extrinsic xyz)."""
    return Rot.from_euler('xyz', [roll, pitch, yaw]).as_matrix()


def matrix_to_rpy(R):
    """Convert 3x3 rotation matrix to (roll, pitch, yaw) tuple."""
    rpy = Rot.from_matrix(R).as_euler('xyz')
    return rpy[0], rpy[1], rpy[2]


def base_to_gripper(eef_pose):
    """Convert 6D EEF pose to 4x4 base-to-gripper transform.

    Alias for pose6d_to_matrix, kept for compatibility with existing code.
    """
    return pose6d_to_matrix(eef_pose)


def normalize_rpy_to_reference(ref_rpy, target_rpy):
    """Adjust target Euler angles to be within [-pi, pi] of the reference.

    Ensures the robot takes the shortest rotation path.
    """
    ref = np.array(ref_rpy, dtype=float)
    result = np.array(target_rpy, dtype=float)
    for i in range(3):
        diff = result[i] - ref[i]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        result[i] = ref[i] + diff
    return result


def canonicalize_yaw(yaw):
    """Exploit parallel gripper symmetry: wrap yaw into [-pi/2, pi/2].

    A parallel gripper is symmetric about its axis, so yaw and yaw +/- pi
    produce equivalent grasps.
    """
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw < -np.pi / 2:
        yaw += np.pi
    return yaw


def unproject_depth(depth_img, fx, fy, cx, cy, scale=1000.0,
                    z_min=0.02, z_max=1.5):
    """Convert a depth image to a 3D point cloud in camera frame.

    Args:
        depth_img: (H, W) depth image (uint16, in mm if scale=1000)
        fx, fy, cx, cy: camera intrinsics
        scale: depth unit conversion (default 1000 = mm to m)
        z_min, z_max: depth range filter (meters)

    Returns:
        points: (N, 3) float32 array of 3D points
        mask: (H, W) boolean mask of valid points
    """
    H, W = depth_img.shape[:2]
    xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))

    points_z = depth_img.astype(np.float32) / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > z_min) & (points_z < z_max) & np.isfinite(points_z)

    points = np.stack([points_x, points_y, points_z], axis=-1)
    return points, mask


def compute_grasp_in_base(R_grasp, t_grasp, e_T_c, eef_pose,
                          R_z_180_4x4, obj_T_pose, pose_T_dummy):
    """Full transform chain: camera frame grasp -> base frame target pose.

    Pipeline:
        1. c_T_obj = make_T(R_grasp, t_grasp)
        2. Apply upside-down camera correction (R_z_180)
        3. Apply object orientation adjustment (obj_T_pose)
        4. Apply gripper length offset (pose_T_dummy)
        5. Transform to EEF frame (e_T_c)
        6. Transform to base frame (b_T_e from current eef_pose)

    Args:
        R_grasp: (3,3) grasp rotation in camera frame
        t_grasp: (3,) grasp translation in camera frame
        e_T_c: (4,4) EEF-to-camera transform
        eef_pose: (6,) current EEF pose [x,y,z,r,p,y]
        R_z_180_4x4: (4,4) upside-down camera correction
        obj_T_pose: (4,4) object orientation adjustment
        pose_T_dummy: (4,4) gripper length offset

    Returns:
        pose6d: (6,) target pose in base frame [x,y,z,roll,pitch,yaw]
        b_T_target: (4,4) full transform in base frame
    """
    c_T_obj = make_T(R_grasp, t_grasp)
    c_T_obj = R_z_180_4x4 @ c_T_obj
    c_T_dummy = c_T_obj @ obj_T_pose
    e_T_dummy = e_T_c @ c_T_dummy
    b_T_e = pose6d_to_matrix(eef_pose)
    b_T_dummy = b_T_e @ e_T_dummy
    b_T_target = b_T_dummy @ pose_T_dummy

    return matrix_to_pose6d(b_T_target), b_T_target
