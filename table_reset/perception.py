"""
Perception module: SAM3 multi-class detection, zone classification, move queue.

Handles Phase 1 (detection), Phase 2 (zone classification), and Phase 3 (queue building).
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from table_reset.config import (
    CAMERA_ID, CAMERA_INTRINSICS, DEPTH_SCALE,
    YELLOW_PLATE_ZONE, BLUE_TRAY_ZONE, ALLOWED_ZONE,
    YELLOW_PLATE_SURFACE_Z, BLUE_TRAY_SURFACE_Z,
    SAM3_ALL_PROMPT, SAM3_SCORE_THRESHOLD,
    CONTAINER_NAMES,
)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Detection:
    """A single detected object."""
    name: str
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixel coords
    score: float
    centroid_px: np.ndarray   # [cx, cy] pixel
    centroid_3d: Optional[np.ndarray] = None  # [x, y, z] in base frame
    zone: Optional[str] = None  # YELLOW_PLATE, BLUE_TRAY, ALLOWED, OUTSIDE
    is_container: bool = False


@dataclass
class SceneState:
    """Full scene perception result."""
    detections: List[Detection] = field(default_factory=list)
    move_queue: List[Detection] = field(default_factory=list)
    objects_in_allowed: List[Detection] = field(default_factory=list)


# =============================================================================
# Zone classification
# =============================================================================

def _is_inside_zone(x, y, zone_bounds):
    """Check if point (x, y) is inside a rectangular zone [xmin, xmax, ymin, ymax]."""
    xmin, xmax, ymin, ymax = zone_bounds
    return xmin <= x <= xmax and ymin <= y <= ymax


def classify_zone(x, y):
    """Classify a base-frame (x, y) position into one of the four zones.

    Returns:
        One of: "YELLOW_PLATE", "BLUE_TRAY", "ALLOWED", "OUTSIDE"
    """
    if _is_inside_zone(x, y, YELLOW_PLATE_ZONE):
        return "YELLOW_PLATE"
    elif _is_inside_zone(x, y, BLUE_TRAY_ZONE):
        return "BLUE_TRAY"
    elif _is_inside_zone(x, y, ALLOWED_ZONE):
        return "ALLOWED"
    else:
        return "OUTSIDE"


# =============================================================================
# SAM3 detection
# =============================================================================

def _to_numpy(x):
    """Convert torch tensor or similar to numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.array(x)


def detect_all(env, processor, prompt=SAM3_ALL_PROMPT):
    """Run SAM3 multi-class detection on the current camera view.

    Args:
        env: DROID RobotEnv (must be at observation pose)
        processor: Sam3Processor instance
        prompt: comma-separated text prompt for all classes

    Returns:
        List[Detection] — one per detected instance
    """
    obs = env.get_observation()
    image_bgra = obs['image'][CAMERA_ID]
    depth_img = obs['depth'][CAMERA_ID]

    # BGR(A) -> RGB
    bgr = image_bgra[..., :3]
    rgb = bgr[..., ::-1].copy()
    image = Image.fromarray(rgb.astype(np.uint8))

    # SAM3 inference
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks_np = _to_numpy(output["masks"])
    boxes_np = _to_numpy(output["boxes"])
    scores_np = _to_numpy(output["scores"])

    if scores_np is None or len(scores_np) == 0:
        print("[SAM3] No objects detected.")
        return []

    # Parse prompt to map indices to names
    class_names = [n.strip() for n in prompt.split(",")]

    detections = []
    for i in range(len(scores_np)):
        score = float(scores_np[i])
        if score < SAM3_SCORE_THRESHOLD:
            continue

        bbox = boxes_np[i][:4].astype(float)
        x1, y1, x2, y2 = bbox
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0

        # Map detection index to class name
        # SAM3 returns detections in the order of the prompt classes.
        # If more detections than classes, use modulo (multiple instances).
        name_idx = i % len(class_names) if len(class_names) > 0 else 0
        name = class_names[name_idx] if name_idx < len(class_names) else f"object_{i}"

        det = Detection(
            name=name,
            bbox=bbox,
            score=score,
            centroid_px=np.array([cx_px, cy_px]),
            is_container=(name in CONTAINER_NAMES),
        )
        detections.append(det)

    print(f"[SAM3] Detected {len(detections)} objects: "
          f"{[d.name for d in detections]}")
    return detections


def compute_3d_centroids(detections, env, e_T_c):
    """Compute base-frame 3D centroid for each detection using depth.

    Updates each Detection's centroid_3d field in-place.

    Args:
        detections: list of Detection objects
        env: DROID RobotEnv
        e_T_c: (4,4) EEF-to-camera transform
    """
    from table_reset.utils import pose6d_to_matrix

    obs = env.get_observation()
    depth_img = obs['depth'][CAMERA_ID]
    eef_pose = obs['robot_state']['cartesian_position']

    fx = CAMERA_INTRINSICS['fx']
    fy = CAMERA_INTRINSICS['fy']
    cx = CAMERA_INTRINSICS['cx']
    cy = CAMERA_INTRINSICS['cy']

    b_T_e = pose6d_to_matrix(eef_pose)
    b_T_c = b_T_e @ e_T_c

    for det in detections:
        px, py = int(det.centroid_px[0]), int(det.centroid_px[1])

        # Clamp to image bounds
        H, W = depth_img.shape[:2]
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)

        # Get depth at centroid (average small region for robustness)
        r = 3  # radius
        y_lo, y_hi = max(0, py - r), min(H, py + r + 1)
        x_lo, x_hi = max(0, px - r), min(W, px + r + 1)
        depth_patch = depth_img[y_lo:y_hi, x_lo:x_hi].astype(float)
        depth_patch = depth_patch[depth_patch > 0]  # filter invalid

        if len(depth_patch) == 0:
            print(f"[WARN] No valid depth for {det.name} at ({px}, {py})")
            continue

        z_cam = np.median(depth_patch) / DEPTH_SCALE

        # Unproject to camera frame
        x_cam = (px - cx) / fx * z_cam
        y_cam = (py - cy) / fy * z_cam
        p_cam = np.array([x_cam, y_cam, z_cam, 1.0])

        # Transform to base frame
        p_base = b_T_c @ p_cam
        det.centroid_3d = p_base[:3]


def classify_all_zones(detections):
    """Classify each detection's zone based on its 3D centroid.

    Updates each Detection's zone field in-place.
    """
    for det in detections:
        if det.centroid_3d is None:
            det.zone = "UNKNOWN"
            continue
        det.zone = classify_zone(det.centroid_3d[0], det.centroid_3d[1])


# =============================================================================
# Move queue
# =============================================================================

def build_move_queue(detections):
    """Build prioritized move queue from classified detections.

    Priority:
        1. Objects in YELLOW_PLATE / BLUE_TRAY (higher z first — top of stack)
        2. Objects in OUTSIDE zone
        3. Skip: ALLOWED zone objects and containers

    Returns:
        SceneState with move_queue and objects_in_allowed populated
    """
    state = SceneState(detections=detections)

    plate_tray = []
    outside = []

    for det in detections:
        if det.is_container:
            continue  # never move containers
        if det.zone == "ALLOWED":
            state.objects_in_allowed.append(det)
        elif det.zone in ("YELLOW_PLATE", "BLUE_TRAY"):
            plate_tray.append(det)
        elif det.zone == "OUTSIDE":
            outside.append(det)
        # UNKNOWN: skip (no valid depth)

    # Sort plate/tray objects by z descending (top object first)
    plate_tray.sort(key=lambda d: d.centroid_3d[2] if d.centroid_3d is not None else 0,
                    reverse=True)

    state.move_queue = plate_tray + outside

    print(f"[Queue] {len(state.move_queue)} objects to move: "
          f"{[d.name for d in state.move_queue]}")
    print(f"[Queue] {len(state.objects_in_allowed)} objects already in Allowed Zone")

    return state


# =============================================================================
# Convenience: full perception pipeline
# =============================================================================

def perceive_scene(env, processor, e_T_c):
    """Run the full perception pipeline (Phase 1 + 2 + 3).

    Args:
        env: DROID RobotEnv (at observation pose)
        processor: Sam3Processor
        e_T_c: (4,4) EEF-to-camera transform

    Returns:
        SceneState with all fields populated
    """
    detections = detect_all(env, processor)
    if not detections:
        return SceneState()

    compute_3d_centroids(detections, env, e_T_c)
    classify_all_zones(detections)
    return build_move_queue(detections)
