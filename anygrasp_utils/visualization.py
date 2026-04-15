import numpy as np
import cv2
from cam_config import intrinsics

def project_points_cam_to_px(P_cam, fx, fy, cx, cy):
    """
    P_cam: (N, 3) points in camera frame
    returns: (N, 2) pixel coords, and a mask for valid points
    """
    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
    valid = Z > 1e-6
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    pts = np.stack([u, v], axis=1)
    return pts, valid

def draw_grasp_axes(img_bgr, R, t, K=intrinsics, axis_len=0.05):
    """
    img_bgr: (720,1280,3)
    R: (3,3) rotation grasp->cam
    t: (3,) translation grasp origin in cam
    K: dict with fx, fy, cx, cy
    """
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]

    # Gripper-frame axes endpoints
    p0_g = np.array([0.0, 0.0, 0.0])
    px_g = np.array([axis_len, 0.0, 0.0])
    py_g = np.array([0.0, axis_len, 0.0])
    pz_g = np.array([0.0, 0.0, axis_len])

    P_g = np.stack([p0_g, px_g, py_g, pz_g], axis=0)  # (4,3)

    for rot, trans in zip(R, t):
        # Transform to camera frame: P_cam = R*P_g + t
        P_cam = (rot @ P_g.T).T + trans[None, :]

        pts_px, valid = project_points_cam_to_px(P_cam, fx, fy, cx, cy)
        if not valid.all():
            # If any are behind camera, you can skip or handle partially
            pass

        # Convert to int pixel coords
        pts = pts_px.astype(np.int32)
        p0, px, py, pz = pts[0], pts[1], pts[2], pts[3]

        H, W = img_bgr.shape[:2]
        def in_bounds(p):
            return 0 <= p[0] < W and 0 <= p[1] < H

        # Draw only if endpoints are within image bounds
        if in_bounds(p0) and in_bounds(px):
            cv2.line(img_bgr, tuple(p0), tuple(px), (0,0,255), 2)   # x in red
        if in_bounds(p0) and in_bounds(py):
            cv2.line(img_bgr, tuple(p0), tuple(py), (0,255,0), 2)   # y in green
        if in_bounds(p0) and in_bounds(pz):
            cv2.line(img_bgr, tuple(p0), tuple(pz), (255,0,0), 2)   # z in blue

        cv2.circle(img_bgr, tuple(p0), 4, (255,255,255), -1)
    return img_bgr

