"""
Grasp planning module with zone-based point cloud processing.

Branching strategy:
  - OUTSIDE zone objects: individual bbox crop -> AnyGrasp
  - YELLOW_PLATE / BLUE_TRAY objects: zone-wide crop with z-threshold -> AnyGrasp picks best
"""

import numpy as np

from table_reset.config import (
    CAMERA_ID, CAMERA_INTRINSICS, DEPTH_SCALE,
    E_T_C, R_Z_180_4x4,
    ANYGRASP_CHECKPOINT, ANYGRASP_MAX_GRIPPER_WIDTH,
    ANYGRASP_GRIPPER_HEIGHT, ANYGRASP_TOP_DOWN, ANYGRASP_LIMS,
    GRIPPER_LENGTH,
    YELLOW_PLATE_ZONE, BLUE_TRAY_ZONE,
    YELLOW_PLATE_SURFACE_Z, BLUE_TRAY_SURFACE_Z,
)
from table_reset.utils import (
    unproject_depth, compute_grasp_in_base, pose6d_to_matrix,
)
from table_reset.perception import Detection


# =============================================================================
# AnyGrasp initialization
# =============================================================================

def create_anygrasp():
    """Initialize and load AnyGrasp model.

    Returns:
        anygrasp: loaded AnyGrasp instance
    """
    from gsnet import AnyGrasp

    class Cfg:
        pass

    cfgs = Cfg()
    cfgs.checkpoint_path = ANYGRASP_CHECKPOINT
    cfgs.max_gripper_width = ANYGRASP_MAX_GRIPPER_WIDTH
    cfgs.gripper_height = ANYGRASP_GRIPPER_HEIGHT
    cfgs.top_down_grasp = ANYGRASP_TOP_DOWN
    cfgs.debug = False

    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()
    print("[AnyGrasp] Model loaded.")
    return anygrasp


# =============================================================================
# Pre-built transform matrices
# =============================================================================

# Object orientation adjustment (Y-axis 90 deg rotation)
_R_y_90 = np.array([
    [0., 0., 1.],
    [0., 1., 0.],
    [-1., 0., 0.]
])
OBJ_T_POSE = np.eye(4)
OBJ_T_POSE[:3, :3] = _R_y_90

# Gripper length offset
POSE_T_DUMMY = np.eye(4)
POSE_T_DUMMY[2, 3] = -GRIPPER_LENGTH


# =============================================================================
# Point cloud extraction helpers
# =============================================================================

def _get_rgbd(env):
    """Get RGB (float [0,1]) and depth images from the camera."""
    obs = env.get_observation()
    image_bgra = obs['image'][CAMERA_ID]
    depth_img = obs['depth'][CAMERA_ID]
    eef_pose = obs['robot_state']['cartesian_position']

    bgr = image_bgra[..., :3]
    rgb = bgr[..., ::-1].copy().astype(np.float32) / 255.0
    return rgb, depth_img, eef_pose


def _depth_to_pointcloud(depth_img, rgb, mask=None):
    """Convert depth + RGB to point cloud arrays.

    Args:
        depth_img: (H, W) depth image
        rgb: (H, W, 3) float32 RGB in [0, 1]
        mask: optional (H, W) boolean mask to restrict region

    Returns:
        points: (N, 3) float32
        colors: (N, 3) float32
    """
    fx = CAMERA_INTRINSICS['fx']
    fy = CAMERA_INTRINSICS['fy']
    cx = CAMERA_INTRINSICS['cx']
    cy = CAMERA_INTRINSICS['cy']

    points, valid_mask = unproject_depth(depth_img, fx, fy, cx, cy,
                                         scale=DEPTH_SCALE)

    if mask is not None:
        valid_mask = valid_mask & mask

    pts = points[valid_mask].astype(np.float32)
    cols = rgb[valid_mask].astype(np.float32)
    return pts, cols


def _bbox_to_pixel_mask(bbox, shape):
    """Convert [x1, y1, x2, y2] bbox to a boolean pixel mask.

    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates
        shape: (H, W) of the image

    Returns:
        (H, W) boolean mask
    """
    H, W = shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _zone_to_pixel_mask(zone_bounds, depth_img, eef_pose, e_T_c):
    """Project a base-frame zone rectangle onto the camera image plane.

    Converts the 4 corners of the zone (at table height) to pixel coordinates,
    then creates a rectangular mask enclosing them.

    Args:
        zone_bounds: [x_min, x_max, y_min, y_max] in base frame
        depth_img: (H, W) for image dimensions
        eef_pose: current 6D EEF pose
        e_T_c: (4,4) EEF-to-camera transform

    Returns:
        (H, W) boolean mask covering the zone's projection
    """
    xmin, xmax, ymin, ymax = zone_bounds
    table_z = 0.18  # approximate table height in base frame

    # 4 corners of the zone in base frame
    corners_base = np.array([
        [xmin, ymin, table_z, 1],
        [xmin, ymax, table_z, 1],
        [xmax, ymin, table_z, 1],
        [xmax, ymax, table_z, 1],
    ])

    # Transform to camera frame
    b_T_e = pose6d_to_matrix(eef_pose)
    b_T_c = b_T_e @ e_T_c
    c_T_b = np.linalg.inv(b_T_c)

    corners_cam = (c_T_b @ corners_base.T).T[:, :3]

    # Project to pixel coords
    fx = CAMERA_INTRINSICS['fx']
    fy = CAMERA_INTRINSICS['fy']
    cx = CAMERA_INTRINSICS['cx']
    cy = CAMERA_INTRINSICS['cy']

    us = (fx * corners_cam[:, 0] / corners_cam[:, 2] + cx).astype(int)
    vs = (fy * corners_cam[:, 1] / corners_cam[:, 2] + cy).astype(int)

    H, W = depth_img.shape[:2]
    u_min, u_max = np.clip(us.min() - 20, 0, W), np.clip(us.max() + 20, 0, W)
    v_min, v_max = np.clip(vs.min() - 20, 0, H), np.clip(vs.max() + 20, 0, H)

    mask = np.zeros((H, W), dtype=bool)
    mask[v_min:v_max, u_min:u_max] = True
    return mask


def _filter_surface_points(points, colors, surface_z, margin=0.01):
    """Remove points at or below the container surface height.

    This prevents AnyGrasp from trying to grasp the plate/tray itself.

    Args:
        points: (N, 3) in camera frame (z = depth)
        colors: (N, 3)
        surface_z: z height of container surface in camera frame
        margin: extra margin below surface to also remove (m)

    Returns:
        filtered_points: (M, 3)
        filtered_colors: (M, 3)
    """
    # In camera frame, z is depth (distance from camera).
    # We need to filter by the camera-frame z that corresponds to the
    # base-frame surface height. Since this depends on camera pose,
    # we use a simpler approach: filter by relative depth.
    # Points on the container surface will have larger z (farther from camera)
    # than objects sitting on top of the container.
    #
    # Actually, the surface_z threshold is in base frame. We'll handle this
    # differently: the caller will convert to base frame and filter there.
    # For now, return as-is and let the grasp planner handle filtering.
    return points, colors


# =============================================================================
# Grasp planning
# =============================================================================

def plan_grasp_outside(env, anygrasp, detection):
    """Plan a grasp for an object in the Outside zone (individual bbox crop).

    Args:
        env: DROID RobotEnv
        anygrasp: loaded AnyGrasp instance
        detection: Detection with bbox

    Returns:
        grasp_pose6d: (6,) in base frame, or None if no grasp found
    """
    rgb, depth_img, eef_pose = _get_rgbd(env)

    # Crop point cloud by bbox
    mask = _bbox_to_pixel_mask(detection.bbox, depth_img.shape[:2])
    points, colors = _depth_to_pointcloud(depth_img, rgb, mask=mask)

    if len(points) < 50:
        print(f"[GraspPlanner] Too few points for {detection.name}: {len(points)}")
        return None

    return _run_anygrasp(anygrasp, points, colors, eef_pose)


def plan_grasp_zone(env, anygrasp, zone_name):
    """Plan a grasp for the top object in a plate/tray zone (zone-wide crop).

    Crops the entire zone region, removes points at/below the container surface,
    and lets AnyGrasp choose the best grasp (naturally the topmost object).

    Args:
        env: DROID RobotEnv
        anygrasp: loaded AnyGrasp instance
        zone_name: "YELLOW_PLATE" or "BLUE_TRAY"

    Returns:
        grasp_pose6d: (6,) in base frame, or None if no grasp found
    """
    if zone_name == "YELLOW_PLATE":
        zone_bounds = YELLOW_PLATE_ZONE
        surface_z = YELLOW_PLATE_SURFACE_Z
    elif zone_name == "BLUE_TRAY":
        zone_bounds = BLUE_TRAY_ZONE
        surface_z = BLUE_TRAY_SURFACE_Z
    else:
        raise ValueError(f"Unknown zone: {zone_name}")

    rgb, depth_img, eef_pose = _get_rgbd(env)

    # Get pixel mask for the zone
    zone_mask = _zone_to_pixel_mask(zone_bounds, depth_img, eef_pose, E_T_C)
    points, colors = _depth_to_pointcloud(depth_img, rgb, mask=zone_mask)

    if len(points) < 50:
        print(f"[GraspPlanner] Too few points in {zone_name}: {len(points)}")
        return None

    # Filter out container surface points using base-frame z threshold.
    # Convert camera-frame points to base frame, filter, convert back.
    b_T_e = pose6d_to_matrix(eef_pose)
    b_T_c = b_T_e @ E_T_C

    # Homogeneous coordinates
    ones = np.ones((len(points), 1), dtype=np.float32)
    pts_cam_h = np.hstack([points, ones])
    pts_base = (b_T_c @ pts_cam_h.T).T[:, :3]

    # Keep only points above the container surface (with margin)
    above_surface = pts_base[:, 2] > (surface_z + 0.01)
    points = points[above_surface]
    colors = colors[above_surface]

    if len(points) < 50:
        print(f"[GraspPlanner] No object points above surface in {zone_name}")
        return None

    return _run_anygrasp(anygrasp, points, colors, eef_pose)


def _run_anygrasp(anygrasp, points, colors, eef_pose):
    """Run AnyGrasp and convert the best grasp to base frame.

    Args:
        anygrasp: loaded AnyGrasp instance
        points: (N, 3) camera-frame points
        colors: (N, 3) RGB colors
        eef_pose: current 6D EEF pose

    Returns:
        grasp_pose6d: (6,) in base frame, or None
    """
    gg, cloud = anygrasp.get_grasp(
        points, colors,
        lims=ANYGRASP_LIMS,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,
    )

    if len(gg) == 0:
        print("[AnyGrasp] No grasp detected.")
        return None

    gg = gg.nms().sort_by_score()
    print(f"[AnyGrasp] Top grasp score: {gg[0].score:.3f}")

    R = gg.rotation_matrices[0]
    t = gg.translations[0]

    grasp_pose6d, _ = compute_grasp_in_base(
        R, t,
        e_T_c=E_T_C,
        eef_pose=eef_pose,
        R_z_180_4x4=R_Z_180_4x4,
        obj_T_pose=OBJ_T_POSE,
        pose_T_dummy=POSE_T_DUMMY,
    )

    return grasp_pose6d
