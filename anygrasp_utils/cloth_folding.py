import torch
import numpy as np
import json
import os
from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime

# --- SAM3 Imports ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Robot Env Imports ---
from droid.robot_env import RobotEnv
import random

# ================= Configuration =================
action_space = "cartesian_position"
gripper_action_space = "position"

W, H = 1280, 720

camera_kwargs = dict(
    hand_camera=dict(image=True, concatenate_images=False, resolution=(W, H), resize_func=None),
    varied_camera=dict(image=True, concatenate_images=False, resolution=(W, H), resize_func=None),
)

# ================= Poses =================
top_pose = [
    0.4350, 0.0, 0.75,
    -3.135387735082638, 0.0117385168564208, -0.008736370437155985
]

default_orientation = [
    -3.135387735082638, 0.0117385168564208, -0.008736370437155985
]

hold_pose = [
    0.4028504033168524, -0.13965526467332448, 0.55,
    -3.135387735082638, 0.0117385168564208, -0.008736370437155985
]

basket_pose = [
    0.47285040331685246, 0.23034473532667558, 0.800,
    -3.135387735082638, 0.0117385168564208, -0.008736370437155985
]

# ================= Motion =================

def run(env, pose, duration=1.0, grp=False):
    pose = np.array(pose)
    pose = np.concatenate([pose, np.array([1.0]) if grp else np.array([0.0])])
    for _ in range(int(duration * 15)):
        env.step(pose)
        time.sleep(1 / 15)


def dry_run(env, pose, grp=False):
    pose = np.array(pose)
    pose = np.concatenate([pose, np.array([1.0]) if grp else np.array([0.0])])
    env.step(pose)
    time.sleep(1 / 30)


# ================= Perception =================

def parse_intrinsics(m):
    return {'fx': m[0, 0], 'fy': m[1, 1], 'cx': m[0, 2], 'cy': m[1, 2]}


def get_observation_data(env, cam_key="14013996_left"):
    obs = env.get_observation()

    if cam_key not in obs["image"]:
        cam_key = list(obs["image"].keys())[0]
        print(f"Warning: Using fallback camera '{cam_key}'")

    img = obs["image"][cam_key][..., :3]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    image_pil = Image.fromarray(img)

    intrinsics = parse_intrinsics(obs["camera_intrinsics"][cam_key])
    extrinsics = obs.get("camera_extrinsic", {}).get(cam_key, np.eye(4))

    depth_map = None
    if "depth" in obs and cam_key in obs["depth"]:
        depth_map = obs["depth"][cam_key]
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)
    else:
        depth_map = np.ones((H, W)) * 0.75

    return image_pil, depth_map, intrinsics, extrinsics


def get_3d_coords(box, depth_map, intrinsics):
    x1, y1, x2, y2 = box
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = depth_map.shape
    Z = depth_map[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))]
    if np.isnan(Z) or np.isinf(Z) or Z <= 0:
        return None
    X = (u - intrinsics['cx']) * Z / intrinsics['fx']
    Y = (v - intrinsics['cy']) * Z / intrinsics['fy']
    return (X, Y, Z)


def to_base_frame(point_cam, T):
    p = np.array([*point_cam, 1.0])
    return (T @ p)[:3]


def draw_boxes_pil(image, boxes, scores=None, color="red", width=3):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    W_img, H_img = img.size
    boxes = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    if scores is not None:
        scores = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
    for i, b in enumerate(boxes):
        if np.max(b) <= 1.5:
            b = b * np.array([W_img, H_img, W_img, H_img])
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if scores is not None:
            text = f"{scores[i]:.2f}"
            bbox = draw.textbbox((x1, y1), text)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), text, fill="white")
    return img


# ================= Detection =================

def detect_object(env, processor, object_name, cam_key="14013996_left",
                  folder_name=None, save_idx=0):
    """Detect and return grasp pose, or None."""
    image_pil, depth_map, intrinsics, extrinsics = get_observation_data(env, cam_key)

    b_ch, g_ch, r_ch = image_pil.split()
    image_rgb = Image.merge("RGB", (r_ch, g_ch, b_ch))

    state = processor.set_image(image_rgb)
    output = processor.set_text_prompt(state=state, prompt=object_name)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    if len(boxes) == 0:
        return None

    best_idx = torch.argmax(scores) if hasattr(scores, 'argmax') else np.argmax(scores)
    box = boxes[best_idx]
    box_np = box.detach().cpu().numpy() if hasattr(box, "detach") else np.asarray(box)
    if np.max(box_np) <= 1.5:
        box_np = box_np * np.array([W, H, W, H])

    coords_cam = get_3d_coords(box_np, depth_map, intrinsics)
    if coords_cam is None:
        return None

    coords_base = to_base_frame(coords_cam, extrinsics)
    Xb, Yb, Zb = coords_base
    print(f"[Detect] Base frame: X={Xb:.4f}, Y={Yb:.4f}, Z={Zb:.4f}")

    if folder_name:
        vis = draw_boxes_pil(image_rgb, boxes, scores)
        vis.save(f"{folder_name}/detect_{save_idx}.png")

    noise_x = random.uniform(-0.03, 0.03)
    noise_y = random.uniform(-0.03, 0.03)
    calib_offset_x = 0.22

    grasp_pose = [
        0.4350 + Yb + calib_offset_x + noise_x,
        Xb + noise_y,
        0.185,
        *default_orientation
    ]
    return grasp_pose


# ================= Fast Unfold Primitives =================

def whip(env, pose, amplitude=0.28):
    """
    Single fast vertical whip.
    ~2 seconds total.
    """
    up = pose.copy()
    up[2] += amplitude
    run(env, up, duration=0.3, grp=True)   # fast up
    time.sleep(0.2)                          # sudden stop - cloth swings

    down = pose.copy()
    down[2] -= amplitude * 0.4
    run(env, down, duration=0.2, grp=True)  # snap down
    time.sleep(0.15)

    run(env, pose, duration=0.3, grp=True)  # return


def shake(env, pose, cycles=3, amp=0.09):
    """
    Quick lateral shake.
    ~1.5 seconds for 3 cycles.
    """
    for _ in range(cycles):
        l = pose.copy(); l[1] -= amp
        run(env, l, duration=0.1, grp=True)
        r = pose.copy(); r[1] += amp
        run(env, r, duration=0.1, grp=True)
    run(env, pose, duration=0.2, grp=True)


# ================= Main =================

def main():
    env = RobotEnv(
        action_space=action_space,
        gripper_action_space=gripper_action_space,
        camera_kwargs=camera_kwargs
    )
    run(env, top_pose, duration=1.5, grp=False)

    print("Loading SAM3 Model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    object_name = input("Object to reset: ")
    iteration = int(input("How many iterations? [1]: ") or "1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"cloth_folding/{object_name}_{timestamp}"
    os.makedirs(folder, exist_ok=True)

    for i in range(iteration):
        print(f"\n{'='*40}")
        print(f"  Iteration {i+1}/{iteration}")
        print(f"{'='*40}")

        # --- 1. Move to observation pose ---
        run(env, top_pose, duration=1.0, grp=False)

        # --- 2. Detect & compute grasp ---
        grasp = detect_object(env, processor, object_name,
                              folder_name=folder, save_idx=i)
        if grasp is None:
            print(f"[Main] '{object_name}' not found.")
            object_name = input("New object name: ")
            continue

        # --- 3. Grasp (with enough time to fully close) ---
        run(env, grasp, duration=1.5, grp=False)    # approach
        run(env, grasp, duration=2.0, grp=True)     # close gripper - wait for full grasp
        time.sleep(0.5)                              # extra settle time

        # --- 4. Lift ---
        run(env, hold_pose, duration=1.0, grp=True)

        # --- 5. Unfold: whip × 2 ---
        whip(env, hold_pose, amplitude=0.28)
        whip(env, hold_pose, amplitude=0.22)

        # --- 6. Place: move high above basket, drop ---
        high_basket = basket_pose.copy()
        high_basket[2] += 0.10                        # extra height for air drop
        run(env, high_basket, duration=1.0, grp=True)
        run(env, high_basket, duration=0.3, grp=False) # release

        # --- 7. Return home ---
        run(env, top_pose, duration=1.0, grp=False)

        print(f"[Main] Iteration {i+1} done!")
        if i < iteration - 1:
            input("Enter to continue...")

    print("\n[Main] All done.")


if __name__ == "__main__":
    main()