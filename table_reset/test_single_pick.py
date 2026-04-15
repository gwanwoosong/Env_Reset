"""
Single object pick-and-place test for debugging.

Usage:
    python -m table_reset.test_single_pick [--mode teleop|auto] [--place x,y,z]

Modes:
    teleop (default): manually teleop to grasp pose
    auto: SAM3 detect + AnyGrasp plan, pick the highest-scored grasp

Examples:
    python -m table_reset.test_single_pick
    python -m table_reset.test_single_pick --mode auto
    python -m table_reset.test_single_pick --mode auto --place 0.45,0.0,0.25
"""

import argparse
import time
import numpy as np

from table_reset.config import (
    TOP_POSE, HOME_POSE, ALLOWED_ZONE, CAMERA_ID,
    E_T_C, MOTION_HZ, GRID_SLOTS,
)
from table_reset.motion import move_to_top, pick, place, run


# =========================================================================
# Env
# =========================================================================

def create_env():
    from droid.robot_env import RobotEnv
    return RobotEnv(
        action_space="cartesian_position",
        gripper_action_space="position",
        camera_kwargs=dict(
            hand_camera=dict(
                image=True, depth=True,
                resolution=(1280, 720), resize_func='cv2',
            )
        ),
    )


# =========================================================================
# Teleop pick
# =========================================================================

def teleop_pick(env):
    """Teleop to grasp pose, pick, and return to top."""
    from table_reset.calibrate import teleop_touch

    move_to_top(env)
    print("\n[teleop] Move to the object you want to pick.")
    grasp_pose = teleop_touch(env, HOME_POSE, label="grasp target")
    if grasp_pose is None:
        print("[teleop] Aborted.")
        return None

    move_to_top(env)
    print(f"\n[teleop] Picking at: {np.round(grasp_pose[:3], 4).tolist()}")
    success = pick(env, grasp_pose)
    if success:
        print("[teleop] Pick succeeded. Holding object at TOP_POSE.")
    else:
        print("[teleop] Pick failed (safety abort).")
    return grasp_pose if success else None


# =========================================================================
# Auto pick (SAM3 + AnyGrasp)
# =========================================================================

def auto_detect(env, prompt=None):
    """Run SAM3 detection and print results."""
    from table_reset.perception import detect_all, compute_3d_centroids, classify_all_zones
    from table_reset.config import SAM3_ALL_PROMPT

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    prompt = prompt or SAM3_ALL_PROMPT

    move_to_top(env)
    print(f"\n[auto] Running SAM3 detection with prompt: '{prompt}'")
    detections = detect_all(env, processor, prompt=prompt)

    if not detections:
        print("[auto] No objects detected.")
        return None, None

    compute_3d_centroids(detections, env, E_T_C)
    classify_all_zones(detections)

    print(f"\n[auto] Detected {len(detections)} objects:")
    for i, d in enumerate(detections):
        pos = np.round(d.centroid_3d, 4).tolist() if d.centroid_3d is not None else "N/A"
        print(f"  [{i}] {d.name:15s}  score={d.score:.2f}  zone={d.zone:15s}  pos={pos}")

    return detections, processor


def auto_pick(env, detections):
    """Pick the user-selected object using AnyGrasp."""
    from table_reset.grasp_planner import (
        create_anygrasp, plan_grasp_outside, plan_grasp_zone,
    )
    from table_reset.perception import Detection

    idx = input("\n[auto] Pick which object? (index, or Enter for 0): ").strip()
    idx = int(idx) if idx else 0

    if idx < 0 or idx >= len(detections):
        print(f"[auto] Invalid index: {idx}")
        return None

    det = detections[idx]
    print(f"[auto] Selected: {det.name} (zone={det.zone})")

    print("[auto] Loading AnyGrasp...")
    anygrasp = create_anygrasp()

    move_to_top(env)

    if det.zone in ("YELLOW_PLATE", "BLUE_TRAY"):
        grasp_pose = plan_grasp_zone(env, anygrasp, det.zone)
    else:
        grasp_pose = plan_grasp_outside(env, anygrasp, det)

    if grasp_pose is None:
        print("[auto] No grasp found.")
        return None

    print(f"[auto] Grasp pose: {np.round(grasp_pose[:3], 4).tolist()}")
    confirm = input("[auto] Execute pick? (y/n): ").strip().lower()
    if confirm != 'y':
        print("[auto] Cancelled.")
        return None

    success = pick(env, grasp_pose)
    if success:
        print("[auto] Pick succeeded.")
    else:
        print("[auto] Pick failed.")
    return grasp_pose if success else None


# =========================================================================
# Place
# =========================================================================

def do_place(env, target_xyz=None):
    """Place the held object."""
    if target_xyz is None:
        target_xyz = GRID_SLOTS[4]  # center slot
        print(f"[place] Using center grid slot: {target_xyz}")

    print(f"[place] Placing at: {np.round(target_xyz, 4).tolist()}")
    place(env, target_xyz)
    print("[place] Done.")


# =========================================================================
# Standalone sub-tests
# =========================================================================

def test_detect_only(env, prompt=None):
    """Just run detection and print results, no picking."""
    detections, _ = auto_detect(env, prompt=prompt)
    return detections


def test_move_to_top(env):
    """Just move to TOP_POSE and read current pose."""
    move_to_top(env)
    obs = env.get_observation()
    pose = obs['robot_state']['cartesian_position']
    print(f"[top] Current pose: {np.round(pose, 4).tolist()}")


def test_camera(env):
    """Grab one frame and print stats."""
    obs = env.get_observation()
    img = obs['image'][CAMERA_ID]
    depth = obs['depth'][CAMERA_ID]
    print(f"[camera] Image: {img.shape}, Depth: {depth.shape}")
    print(f"[camera] Depth range: [{depth.min()}, {depth.max()}]")

    try:
        from PIL import Image
        rgb = img[..., :3][..., ::-1].copy()
        Image.fromarray(rgb).save("/tmp/test_capture.png")
        print("[camera] Saved /tmp/test_capture.png")
    except Exception as e:
        print(f"[camera] Could not save image: {e}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Single pick-and-place test")
    parser.add_argument("--mode", choices=["teleop", "auto", "detect", "top", "camera"],
                        default="teleop",
                        help="teleop: manual grasp | auto: SAM3+AnyGrasp | "
                             "detect: detection only | top: move to top | camera: capture")
    parser.add_argument("--prompt", type=str, default=None,
                        help="SAM3 text prompt (e.g. 'orange' or 'orange, apple')")
    parser.add_argument("--place", type=str, default=None,
                        help="Place target as x,y,z (e.g. 0.45,0.0,0.25)")
    parser.add_argument("--skip-place", action="store_true",
                        help="Pick only, don't place")
    args = parser.parse_args()

    target_xyz = None
    if args.place:
        target_xyz = [float(v) for v in args.place.split(",")]

    print("\n" + "=" * 50)
    print(f"  Single Pick Test  (mode={args.mode})")
    print("=" * 50)

    env = create_env()

    # Standalone tests
    if args.mode == "detect":
        test_detect_only(env, prompt=args.prompt)
        return
    if args.mode == "top":
        test_move_to_top(env)
        return
    if args.mode == "camera":
        test_camera(env)
        return

    # Pick
    if args.mode == "teleop":
        result = teleop_pick(env)
    else:
        detections, _ = auto_detect(env, prompt=args.prompt)
        if detections is None:
            return
        result = auto_pick(env, detections)

    if result is None:
        return

    # Place
    if args.skip_place:
        print("\n[skip-place] Skipping place. Opening gripper at top.")
        run(env, TOP_POSE, duration=2.0, grip_close=False)
    else:
        do_place(env, target_xyz)

    print("\n[done]")


if __name__ == "__main__":
    main()
