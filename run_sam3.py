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
    hand_camera=dict(image=False, concatenate_images=False, resolution=(W, H), resize_func=None),
    varied_camera=dict(image=False, concatenate_images=False, resolution=(W, H), resize_func=None),
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

def fling(env):
    resolution = 40
    radius = 0.10

    circle_start_pose = [0.4350 - radius,
    0.0,
    0.4,
    -3.135387735082638,
    0.0117385168564208 + np.pi/6,
    -0.008736370437155985]

    run(env, circle_start_pose, duration=2.0, grp=True)

    for i in range(resolution):
        if i < resolution / 2:
            circle_current_pose = [0.4350 - radius * np.cos(2*np.pi / resolution * i),
                0.0,
                0.4 - radius * np.sin(2*np.pi / resolution * i),
                -3.135387735082638,
                0.0117385168564208 + np.pi/6 - (np.pi/2.5+np.pi/6) * i / (resolution / 2),
                -0.008736370437155985
            ]
            dry_run(env, circle_current_pose, grp=True)
        elif i < resolution * 3 / 4 :
            circle_current_pose = [0.4350 - radius * np.cos(2*np.pi / resolution * i),
                0.0,
                0.4 - radius * np.sin(2*np.pi / resolution * i),
                -3.135387735082638,
                0.0117385168564208 - (np.pi/2.5),
                -0.008736370437155985
            ]
            dry_run(env, circle_current_pose, grp=True)
        else:
            circle_current_pose = [0.4350 - radius * np.cos(2*np.pi / resolution * i),
                0.0,
                0.4 - radius * np.sin(2*np.pi / resolution * i),
                -3.135387735082638,
                0.0117385168564208 - (np.pi/2.5) * (resolution - i) / (resolution / 4),
                -0.008736370437155985
            ]
            dry_run(env, circle_current_pose, grp=False)

    circle_end_pose = [0.4350 - radius,
    0.0,
    0.4,
    -3.135387735082638,
    0.0117385168564208,
    -0.008736370437155985]

    run(env, circle_end_pose, grp=False)


# ================= Main Execution =================

# 1. Load SAM3 Model
print("Loading SAM3 Model...")
model = build_sam3_image_model()
processor = Sam3Processor(model)

object = str(input("Which object you want to reset?:"))
iteration = int(input("How many times you want to try?:"))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{object}_{timestamp}"

fling_counter = 0

if not os.path.exists(folder_name):
    os.mkdir(folder_name)
else:
    pass
input("Enter to continue...")
i = 0
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
        print("Saved visualization to result_carrot_detection.png")


        target_pose = [0.4350 + Yb + 0.15, # holding a tip
            Xb-0.05, # tip
            0.165,
            -3.135387735082638,
            0.0117385168564208,
            -0.008736370437155985]

        run(env, target_pose, duration=2.0, grp=False) # go to grasp pose
        run(env, target_pose, duration=2.0, grp=True) # grasp

        if fling_counter < 2:
            fling(env)
            fling_counter += 1
            continue
        else:
            fling_counter = 0

        run(env, home_pose, duration=1.5, grp=True) # go to home

        x = random.uniform(-0.0, 0.3)
        y = random.uniform(-0.2, 0.0)

        target_place_pose = [0.4350 + x,
            y,
            0.16,
            -3.135387735082638,
            0.0117385168564208,
            -0.008736370437155985
        ]
        print("Random Place pose", target_place_pose)

        run(env, target_place_pose, duration=2.0, grp=True) # go to place pose
        run(env, target_place_pose, duration=1.0, grp=False) # Relase
        run(env, home_pose, duration=2.0, grp=False) # Relase
        print("Finished") 
        input("Enter to continue...") 
        i += 1  
    else:
        print(f"[Result] No {object} detected.")
        object = str(input("Which object you want to reset?:"))