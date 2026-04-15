"""
Pre-experiment calibration & checklist script.

Run on the Franka workstation before experiments:
    python -m table_reset.calibrate

Steps:
    1. Tray zone calibration (teleop -> touch corners)
    2. Allowed zone calibration
    3. TOP_POSE / Z_SAFETY_FLOOR verification
    4. Camera check (intrinsics, image capture)
    5. SAM3 + AnyGrasp model load test
    6. Single pick-and-place dry run
"""

import sys
import time
import numpy as np

from table_reset.config import (
    TOP_POSE, HOME_POSE, Z_SAFETY_FLOOR,
    CAMERA_ID, CAMERA_INTRINSICS,
    E_T_C, MOTION_HZ,
)


# =========================================================================
# Robot env setup
# =========================================================================

def create_env():
    from droid.robot_env import RobotEnv
    env = RobotEnv(
        action_space="cartesian_position",
        gripper_action_space="position",
        camera_kwargs=dict(
            hand_camera=dict(
                image=True, depth=True,
                resolution=(1280, 720), resize_func='cv2',
            )
        ),
    )
    return env


# =========================================================================
# Teleop (inline, same as anygrasp_utils/teleop.py)
# =========================================================================

def _getch():
    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _run(env, pose6, duration=1.0, grip_close=False):
    pose = np.array(pose6, dtype=np.float32)
    grip = np.array([1.0 if grip_close else 0.0], dtype=np.float32)
    action = np.concatenate([pose, grip])
    for _ in range(int(duration * MOTION_HZ)):
        env.step(action)
        time.sleep(1.0 / MOTION_HZ)


def teleop_touch(env, start_pose, label="point", step=0.01, rot_step=0.05):
    """Teleop to touch a point. Returns [x, y, z, r, p, y]."""
    pose = list(start_pose)
    grip_close = False

    print(f"\n{'='*50}")
    print(f"  TELEOP: touch {label}")
    print(f"{'='*50}")
    print("  w/s=X  a/d=Y  q/e=Z  |  g=gripper  r=reset  Enter=confirm")

    key_map = {
        'w': (0, +step), 's': (0, -step),
        'a': (1, +step), 'd': (1, -step),
        'q': (2, +step), 'e': (2, -step),
        'i': (3, +rot_step), 'k': (3, -rot_step),
        'j': (4, +rot_step), 'l': (4, -rot_step),
        'u': (5, +rot_step), 'o': (5, -rot_step),
    }

    while True:
        grip_str = "CLOSED" if grip_close else "OPEN"
        print(f"\r  xyz=({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f})  "
              f"grip={grip_str}    ", end='', flush=True)

        key = _getch()

        if key in ('\r', '\n'):
            print(f"\n  Confirmed: xyz=({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f})")
            break
        elif key == '\x03':
            print("\n  Aborted.")
            return None
        elif key == 'r':
            pose = list(start_pose)
            grip_close = False
            _run(env, pose, duration=1.0, grip_close=False)
            print("\n  Reset.\n")
        elif key == 'g':
            grip_close = not grip_close
            _run(env, pose, duration=0.5, grip_close=grip_close)
        elif key in key_map:
            idx, delta = key_map[key]
            pose[idx] += delta
            _run(env, pose, duration=step / 0.05, grip_close=grip_close)

    return pose


def read_current_pose(env):
    obs = env.get_observation()
    return list(obs['robot_state']['cartesian_position'])


# =========================================================================
# Step 1: Tray zone calibration
# =========================================================================

def calibrate_zone(env, zone_name):
    """Touch two diagonal corners of a tray/zone. Returns [xmin,xmax,ymin,ymax], surface_z."""
    print(f"\n{'#'*60}")
    print(f"  Calibrating: {zone_name}")
    print(f"  Touch two DIAGONAL corners of the {zone_name}.")
    print(f"{'#'*60}")

    _run(env, TOP_POSE, duration=2.0)

    p1 = teleop_touch(env, HOME_POSE, label=f"{zone_name} corner 1")
    if p1 is None:
        return None, None

    _run(env, TOP_POSE, duration=1.0)

    p2 = teleop_touch(env, HOME_POSE, label=f"{zone_name} corner 2")
    if p2 is None:
        return None, None

    _run(env, TOP_POSE, duration=1.0)

    x_min = min(p1[0], p2[0])
    x_max = max(p1[0], p2[0])
    y_min = min(p1[1], p2[1])
    y_max = max(p1[1], p2[1])
    surface_z = (p1[2] + p2[2]) / 2.0

    zone_bounds = [round(x_min, 4), round(x_max, 4),
                   round(y_min, 4), round(y_max, 4)]
    surface_z = round(surface_z, 4)

    print(f"\n  {zone_name}_ZONE = {zone_bounds}")
    print(f"  {zone_name}_SURFACE_Z = {surface_z}")

    return zone_bounds, surface_z


# =========================================================================
# Step 2: Camera check
# =========================================================================

def check_camera(env):
    print(f"\n{'#'*60}")
    print("  Camera Check")
    print(f"{'#'*60}")

    obs = env.get_observation()

    if CAMERA_ID not in obs['image']:
        print(f"  [FAIL] Camera '{CAMERA_ID}' not found!")
        print(f"  Available cameras: {list(obs['image'].keys())}")
        return False

    img = obs['image'][CAMERA_ID]
    depth = obs['depth'][CAMERA_ID]
    print(f"  [OK] Camera '{CAMERA_ID}' found")
    print(f"  Image shape: {img.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth range: [{depth.min()}, {depth.max()}]")
    print(f"  Intrinsics: fx={CAMERA_INTRINSICS['fx']}, fy={CAMERA_INTRINSICS['fy']}")
    return True


# =========================================================================
# Step 3: Model load test
# =========================================================================

def check_models():
    print(f"\n{'#'*60}")
    print("  Model Load Test")
    print(f"{'#'*60}")

    ok = True

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        print("  [OK] SAM3 loaded")
    except Exception as e:
        print(f"  [FAIL] SAM3: {e}")
        ok = False

    try:
        from table_reset.grasp_planner import create_anygrasp
        anygrasp = create_anygrasp()
        print("  [OK] AnyGrasp loaded")
    except Exception as e:
        print(f"  [FAIL] AnyGrasp: {e}")
        ok = False

    return ok


# =========================================================================
# Step 4: Single pick-and-place test
# =========================================================================

def test_pick_and_place(env):
    print(f"\n{'#'*60}")
    print("  Single Pick-and-Place Test")
    print(f"{'#'*60}")
    print("  Teleop to an object, confirm grasp pose, then it will")
    print("  pick and place to the center of the Allowed Zone.")

    _run(env, TOP_POSE, duration=2.0)

    grasp_pose = teleop_touch(env, HOME_POSE, label="grasp target")
    if grasp_pose is None:
        return False

    _run(env, TOP_POSE, duration=1.0)

    from table_reset.config import ALLOWED_ZONE
    target_x = (ALLOWED_ZONE[0] + ALLOWED_ZONE[1]) / 2
    target_y = (ALLOWED_ZONE[2] + ALLOWED_ZONE[3]) / 2
    target_z = 0.25
    target = [target_x, target_y, target_z]

    print(f"  Place target: {target}")

    from table_reset.motion import pick_and_place
    success = pick_and_place(env, grasp_pose, target)

    if success:
        print("  [OK] Pick-and-place succeeded")
    else:
        print("  [FAIL] Pick-and-place failed")
    return success


# =========================================================================
# Main
# =========================================================================

STEPS = {
    "1": ("Calibrate tray zones", "zone"),
    "2": ("Camera check", "camera"),
    "3": ("SAM3 + AnyGrasp load test", "models"),
    "4": ("Single pick-and-place test", "pnp"),
    "all": ("Run all steps", "all"),
}


def main():
    print("\n" + "=" * 60)
    print("  TABLE RESET - Pre-Experiment Calibration")
    print("=" * 60)
    print("\nSteps:")
    for k, (desc, _) in STEPS.items():
        print(f"  {k}. {desc}")
    print()

    choice = input("Select step (1/2/3/4/all): ").strip()
    if choice not in STEPS:
        print("Invalid choice.")
        return

    run_zone = choice in ("1", "all")
    run_camera = choice in ("2", "all")
    run_models = choice in ("3", "all")
    run_pnp = choice in ("4", "all")

    env = None
    if run_zone or run_camera or run_pnp:
        print("\nInitializing robot env...")
        env = create_env()

    results = {}

    # --- Zone calibration ---
    if run_zone:
        _run(env, TOP_POSE, duration=2.0)

        y_bounds, y_sz = calibrate_zone(env, "YELLOW_PLATE")
        b_bounds, b_sz = calibrate_zone(env, "BLUE_TRAY")

        print(f"\n{'='*60}")
        print("  Calibration Results - paste into config.py:")
        print(f"{'='*60}")
        if y_bounds:
            print(f"  YELLOW_PLATE_ZONE = {y_bounds}")
            print(f"  YELLOW_PLATE_SURFACE_Z = {y_sz}")
        if b_bounds:
            print(f"  BLUE_TRAY_ZONE = {b_bounds}")
            print(f"  BLUE_TRAY_SURFACE_Z = {b_sz}")

        print("\n  Also calibrate ALLOWED_ZONE (center area between trays).")
        ans = input("  Calibrate Allowed Zone too? (y/n): ").strip().lower()
        if ans == 'y':
            a_bounds, _ = calibrate_zone(env, "ALLOWED")
            if a_bounds:
                print(f"  ALLOWED_ZONE = {a_bounds}")

        results['zone'] = True

    # --- Camera ---
    if run_camera:
        _run(env, TOP_POSE, duration=2.0)
        results['camera'] = check_camera(env)

    # --- Models ---
    if run_models:
        results['models'] = check_models()

    # --- Pick and place ---
    if run_pnp:
        results['pnp'] = test_pick_and_place(env)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for step, passed in results.items():
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {step}")
    print()


if __name__ == "__main__":
    main()
