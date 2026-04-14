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

# Native Resolution of ZED Mini (HD720)
W, H = 1280, 720

# ================= Setup Environment =================
camera_kwargs = dict(
    hand_camera=dict(image=True, concatenate_images=False, resolution=(W, H), resize_func=None),
    varied_camera=dict(image=True, concatenate_images=False, resolution=(W, H), resize_func=None),
)
def run(env, pose, duration=1.0, grp=False):
    pose = np.array(pose)
    pose = np.concatenate([pose, np.array([1.0]) if grp else np.array([0.0])])
    for i in range(int(duration * 15)):
        env.step(pose)
        time.sleep(1/15)

top_pose = [0.4350,
0.0,
0.75,
-3.135387735082638,
0.0117385168564208,
-0.008736370437155985]

home_pose = [0.4350,
0.0,
0.5,
-3.135387735082638,
0.0117385168564208,
-0.008736370437155985]

bottom_pose = [0.4321105480194092,
 0,
 0.1790000000000000,
 -3.141084945536129,
 0.022017310521937228,
 -0.01628570321540242]

env = RobotEnv(
    action_space=action_space,
    gripper_action_space=gripper_action_space,
    camera_kwargs=camera_kwargs
)

run(env, top_pose, duration=2.0, grp=False)

# ================= Helper Functions =================


def visualize_bbox_and_mask(rgb, box, mask, title="Detection"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)

    # bbox format is usually [x1, y1, x2, y2]
    x1, y1, x2, y2 = box[:4]
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    # overlay mask
    mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    mask_overlay[..., 0] = 1.0   # red
    mask_overlay[..., 3] = mask.astype(np.float32) * 0.35
    ax.imshow(mask_overlay)

    ax.set_title(title)
    ax.axis("off")

def parse_intrinsics(intrinsics_matrix):
    """Directly extracts fx, fy, cx, cy from the raw 3x3 intrinsic matrix."""
    return {
        'fx': intrinsics_matrix[0, 0],
        'fy': intrinsics_matrix[1, 1],
        'cx': intrinsics_matrix[0, 2],
        'cy': intrinsics_matrix[1, 2]
    }

def get_observation_data(env, cam_key="14013996_left"):
    """
    Captures observation, handling intrinsics, extrinsics, and image parsing.
    """
    obs_dict = env.get_observation()
    print(obs_dict.keys())
    # 1. Handle Key fallback
    if cam_key not in obs_dict["image"]:
        available_keys = list(obs_dict["image"].keys())
        print(f"Warning: Camera '{cam_key}' not found. Using '{available_keys[0]}'")
        cam_key = available_keys[0]

    # 2. Process Image
    # img_array = obs_dict["image"][cam_key][..., :3] # Get RGB/
    img_array = obs_dict["image"][cam_key][..., :3]
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    image_pil = Image.fromarray(img_array)

    # 3. Process Intrinsics (3x3 Matrix)
    raw_intrinsics = obs_dict["camera_intrinsics"][cam_key] 
    intrinsics = parse_intrinsics(raw_intrinsics)

    # 4. Process Extrinsics (4x4 Matrix)
    # Assumes matrix is T_base_camera (Camera Pose in Base Frame)
    if "camera_extrinsic" in obs_dict:
        extrinsics = obs_dict["camera_extrinsic"][cam_key]
    else:
        print("Warning: 'camera_extrinsic' key not found in obs_dict. Using Identity.")
        extrinsics = np.eye(4)

    # 5. Process Depth
    depth_map = None
    if "depth" in obs_dict and cam_key in obs_dict["depth"]:
         depth_map = obs_dict["depth"][cam_key]
         if depth_map.ndim == 3:
             depth_map = depth_map.squeeze(-1)
    else:
        print("Warning: Real depth not found. Using dummy depth map.")
        depth_map = np.ones((H, W)) * 0.75 

    return image_pil, depth_map, intrinsics, extrinsics

def get_zed_3d_coordinates(box, depth_map, intrinsics):
    """Calculates X, Y, Z in CAMERA FRAME."""
    x1, y1, x2, y2 = box
    u_center = (x1 + x2) / 2
    v_center = (y1 + y2) / 2
    
    h, w = depth_map.shape
    u_idx = int(np.clip(u_center, 0, w - 1))
    v_idx = int(np.clip(v_center, 0, h - 1))
    
    Z = depth_map[v_idx, u_idx]
    
    if np.isnan(Z) or np.isinf(Z) or Z <= 0:
        return None

    X = (u_center - intrinsics['cx']) * Z / intrinsics['fx']
    Y = (v_center - intrinsics['cy']) * Z / intrinsics['fy']
    
    return (X, Y, Z)

def transform_to_base_frame(point_cam, extrinsic_matrix):
    """
    Transforms a point from Camera Frame to Robot Base Frame.
    point_cam: (x, y, z) tuple
    extrinsic_matrix: 4x4 numpy array (T_base_camera)
    """
    # Create homogeneous point [x, y, z, 1]
    p_cam_homog = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
    
    # Apply transformation: P_base = T * P_cam
    p_base_homog = extrinsic_matrix @ p_cam_homog
    
    # Extract x, y, z
    return p_base_homog[:3]

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

def dry_run(env, pose, grp=False):
    pose = np.array(pose)
    pose = np.concatenate([pose, np.array([1.0]) if grp else np.array([0.0])])
    env.step(pose)
    time.sleep(1/30)

def fling(env, target_fling_pose = [0.395, 0.16, 0.45, -3.135387735082638, 0.0117385168564208, -0.008736370437155985]):
    resolution = 40
    radius = 0.10

    circle_start_pose = target_fling_pose.copy()
    circle_start_pose[0] = circle_start_pose[0] - radius
    circle_start_pose[4] = circle_start_pose[4] + np.pi/6

    run(env, circle_start_pose, duration=2.0, grp=True)

    for i in range(resolution):
        if i < resolution / 2:
            circle_current_pose = target_fling_pose.copy()
            circle_current_pose[0] = circle_current_pose[0] - radius * np.cos(2*np.pi / resolution * i)
            circle_current_pose[2] = circle_current_pose[2] - radius * np.sin(2*np.pi / resolution * i)
            circle_current_pose[4] = circle_current_pose[4] + np.pi/6 - (np.pi/2.5+np.pi/6) * i / (resolution / 2)
            
            dry_run(env, circle_current_pose, grp=True)
        elif i < resolution * 3 / 4 :
            circle_current_pose = target_fling_pose.copy()
            circle_current_pose[0] = circle_current_pose[0] - radius * np.cos(2*np.pi / resolution * i)
            circle_current_pose[2] = circle_current_pose[2] - radius * np.sin(2*np.pi / resolution * i)
            circle_current_pose[4] = circle_current_pose[4] + 0.0117385168564208 - (np.pi/2.5)
            dry_run(env, circle_current_pose, grp=True)
        else:
            circle_current_pose = target_fling_pose.copy()
            circle_current_pose[0] = circle_current_pose[0] - radius * np.cos(2*np.pi / resolution * i)
            circle_current_pose[2] = circle_current_pose[2] - radius * np.sin(2*np.pi / resolution * i)
            circle_current_pose[4] = circle_current_pose[4] - (np.pi/2.5) * (resolution - i) / (resolution / 4)

            dry_run(env, circle_current_pose, grp=False)
    circle_end_pose = target_fling_pose.copy()
    circle_end_pose[0] = circle_end_pose[0] - radius

    run(env, circle_end_pose, grp=False)


def teleop(env, start_pose, step=0.01):
    """
    Keyboard teleop to manually move EEF to a target position.
    Controls:
      w/s  : +/- X
      a/d  : +/- Y
      q/e  : +/- Z
      Enter: confirm current position and return pose
      r    : reset to start_pose

    Returns the final pose [x,y,z,roll,pitch,yaw].
    """
    import sys
    import tty
    import termios

    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    pose = list(start_pose)
    print("\n[TELEOP] Use keys to move EEF:")
    print("  w/s : X +/-    a/d : Y +/-    q/e : Z +/-")
    print("  r   : reset to start    Enter : confirm & exit\n")

    key_map = {
        'w': ( step, 0,    0),
        's': (-step, 0,    0),
        'a': (0,  step,    0),
        'd': (0, -step,    0),
        'q': (0,    0,  step),
        'e': (0,    0, -step),
    }

    while True:
        print(f"\r  pos: x={pose[0]:.4f}  y={pose[1]:.4f}  z={pose[2]:.4f}    ", end='', flush=True)
        key = getch()

        if key == '\r' or key == '\n':
            print(f"\n[TELEOP] Confirmed: {pose}")
            break
        elif key == 'r':
            pose = list(start_pose)
            run(env, pose, duration=2)
            print("\n[TELEOP] Reset to start pose")
        elif key in key_map:
            dx, dy, dz = key_map[key]
            pose[0] += dx
            pose[1] += dy
            pose[2] += dz
            run(env, pose, duration=step/0.05)
        elif key == '\x03':  # Ctrl+C
            print("\n[TELEOP] Interrupted.")
            break

    return pose

def waypoint_pose_xyz_rpy(env, start_pose, trans_xyz, target_rot_mat,
                          steps=30, grip_close=False, hz=15, step_duration=2.0):
    """
    start_pose: [x,y,z,roll,pitch,yaw]  (radians)
    trans_xyz:  (3,) delta translation in same frame as pose
    target_rot_mat: (3,3) desired orientation (absolute) in same frame as pose
    """

    start_pose = np.array(start_pose, dtype=np.float32)

    # rotations: start (from current rpy) -> goal (from matrix)
    r0 = Rot.from_euler('xyz', start_pose[3:6], degrees=False)  # roll,pitch,yaw about x,y,z
    r1 = Rot.from_matrix(target_rot_mat)

    slerp = Slerp([0.0, 1.0], Rot.concatenate([r0, r1]))

    for i in range(1, steps + 1):
        a = i / steps

        # position waypoint
        pose = start_pose.copy()
        pose[0:3] = start_pose[0:3] + a * np.asarray(trans_xyz).reshape(3,)

        # rotation waypoint (slerp)
        ri = slerp([a])[0]
        roll_i, pitch_i, yaw_i = ri.as_euler('xyz', degrees=False)  # -> roll,pitch,yaw
        pose[3:6] = [roll_i, pitch_i, yaw_i]

        run(env, pose.tolist(), duration=step_duration, grip_close=grip_close, hz=hz)

def place_to_basket(
    env,
    target_basket_place_pose=[0.405, 0.300, 0.85
                              -3.135387735082638, 0.0117385168564208, -0.008736370437155985]
):
    """
    Place the cloth into the basket with a snapping motion in the yz plane.
    x is fixed, and y/z follow a half-circle trajectory.
    """

    resolution = 30
    radius = 0.10
    release_ratio = 0.75

    # Start pose: slightly offset in y, with a small tilt
    start_pose = target_basket_place_pose.copy()
    start_pose[1] -= radius
    start_pose[3] -= np.pi / 8

    run(env, start_pose, duration=1.5, grp=True)

    for i in range(resolution):
        theta = np.pi * i / (resolution - 1)   # 0 -> pi

        current_pose = target_basket_place_pose.copy()

        # Half-circle in yz plane: x fixed
        current_pose[0] = target_basket_place_pose[0]
        current_pose[1] = target_basket_place_pose[1] - radius * np.cos(theta)
        current_pose[2] = target_basket_place_pose[2] - radius * np.sin(theta)

        # Optional wrist tilt during motion
        current_pose[3] = target_basket_place_pose[3] - np.pi / 8 + (np.pi / 4) * (i / (resolution - 1))

        # Release near the end
        if i < int(resolution * release_ratio):
            dry_run(env, current_pose, grp=True)
        else:
            dry_run(env, current_pose, grp=False)

    # Move slightly upward after release
    end_pose = target_basket_place_pose.copy()
    end_pose[2] += 0.05
    run(env, end_pose, duration=1.0, grp=False)

# ================= Main Execution =================

# 1. Load SAM3 Model

print("Loading SAM3 Model...")
model = build_sam3_image_model()
processor = Sam3Processor(model)

object = "pink cloth"
iteration = 100

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"cloth_folding/{object}_{timestamp}"

fling_counter = 0
grad_z=0.015
z_offset=0
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
else:
    pass
i = 0

real_trial_count=0
success_count=0
while i  < iteration:
    run(env, top_pose, duration=2.0, grp=False)
    
    # 2. Get Data from Robot
    print("Capturing data from RobotEnv...")
    image_pil, depth_map, zed_intrinsics, zed_extrinsics = get_observation_data(env, cam_key="14013996_left")

    print(f"Captured Image Size: {image_pil.size}")
    print(f"Intrinsics: fx={zed_intrinsics['fx']:.2f}")
    print("Extrinsics (Base <- Camera):\n", zed_extrinsics)

    # 3. Run Inference
    print("Running Inference...")
    b, g, r = image_pil.split()
    image_pil = Image.merge("RGB", (r, g, b))
    inference_state = processor.set_image(image_pil)
    output = processor.set_text_prompt(state=inference_state, prompt=object)

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    # 4. Process Results
    if len(boxes) > 0:
        best_idx = torch.argmax(scores) if hasattr(scores, 'argmax') else np.argmax(scores)
        best_box = boxes[best_idx]
        best_score = scores[best_idx]
        
        # Convert to numpy
        b = best_box.detach().cpu().numpy() if hasattr(best_box, "detach") else np.asarray(best_box)
        if np.max(b) <= 1.5: b = b * np.array([W, H, W, H])

        # A. Calculate Camera Frame Position
        coords_cam = get_zed_3d_coordinates(b, depth_map, zed_intrinsics)
        

        
        mid_x = (b[0] + b[2]) / 2
        mid_y = (b[1] + b[3]) / 2
        
        print(f"\n[Result] Found '{object}' (Score: {best_score:.3f})")
        print(f"[Result] 2D Center: ({mid_x:.1f}, {mid_y:.1f})")
        
        
        
        
        if coords_cam:
            Xc, Yc, Zc = coords_cam
            print(f"[Result] Camera Frame: X={Xc:.4f}m, Y={Yc:.4f}m, Z={Zc:.4f}m")
            
            # B. Calculate Robot Base Frame Position
            coords_base = transform_to_base_frame(coords_cam, zed_extrinsics)
            Xb, Yb, Zb = coords_base
            print(f"[Result] Robot Base Frame: X={Xb:.4f}m, Y={Yb:.4f}m, Z={Zb:.4f}m")
            
        else:
            print("[Result] Invalid depth at center pixel.")

        # 5. Save Visualization
        vis_img = draw_boxes_pil(image_pil, boxes, scores)
        vis_img.save(f"{folder_name}/result_{object}_detection_{i}.png")
        print("Visualization saved")


        # Target pose:  [0.515, 0.23000000000000007, 0.75]
        
        # Xb, Yb -> Give random noise to this 
        
        x_noise = random.uniform(-0.07, 0.07)
        y_noise = random.uniform(-0.07, 0.07)
        # x_noise = 0
        # y_noise = 0
        
        if fling_counter == 0:
            input("press to enter")
            z_offset = 0
        elif fling_counter == 1 or fling_counter == 2:
            z_offset += 0.01
        else:
            z_offset = 0

        new_calib_offset_x = 0.22
        
        target_pose = [0.4350 + Yb + new_calib_offset_x + x_noise , # holding a tip
            Xb + y_noise, # tip
            0.179 + z_offset,
            -3.135387735082638,
            0.0117385168564208,
            -0.008736370437155985]
        print(target_pose)
        run(env, target_pose, duration=2.0, grp=False) # go to grasp pose
        run(env, target_pose, duration=2.0, grp=True) # grasp

        # Go to safe position to fling
        
        target_fling_pose = [0.396, 0.16, 0.44, -3.135387735082638, 0.0117385168564208, -0.008736370437155985]
        run(env, target_fling_pose, duration=2.0, grp=True) # grasp
        
        target_basket_place_pose= [0.415, -0.24, 0.86, -3.135387735082638, 0.0117385168564208, -0.008736370437155985]
        
        # print("teleop")
        # teleop(env, target_fling_pose)
        
        
        if fling_counter < 3:
            fling(env, target_fling_pose=target_fling_pose)
            fling_counter += 1
            continue
        else:
            fling_counter = 0

        print("Place cloth to the basket", target_basket_place_pose)
        run(env, top_pose, duration=2.0, grp=True) 
        
        # place_to_basket(env, target_basket_place_pose)
        run(env, target_basket_place_pose, duration=2.0, grp=True) # go to place pose
        run(env, target_basket_place_pose, duration=1.0, grp=False) # Relase
        run(env, top_pose, duration=2.0, grp=False) # Relase
        print("Finished") 
        success = int(input("1: success, 0: fail -1: Not valid evaluation"))
        if success == 0 or success == 1:
            print("i: ", i)
            i += 1  
        if success:
            success_count += 1
            print (f"Success rate: {success_count} / {i}")
        if i == 11:
            print("Experiments are done!")
            run(env, home_pose, duration=2.0, grp=False)
            import sys
            sys.exit()
    else:
        print(f"[Result] No {object} detected.")
        object = "pink cloth"