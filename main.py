import torch
import numpy as np
import yaml
import os
from PIL import Image
import time
from datetime import datetime

# --- SAM3 & Robot Imports ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from droid.robot_env import RobotEnv


with open("config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

g_cfg = cfg['global_params']
W, H = cfg['camera']['resolution']


def run(env, pose, duration=1.0, grp=False):
    pose = np.array(pose)
    pose = np.concatenate([pose, np.array([1.0]) if grp else np.array([0.0])])
    for _ in range(int(duration * 15)):
        env.step(pose)
        time.sleep(1/15)

def execute_unfold(env, target_pose):
    print("Action: Executing Unfold Arc...")
    res = 20
    for i in range(res):
        p = np.array(target_pose).copy()
        p[1] += 0.15 - np.cos(np.pi * i / res) * 0.15
        p[2] += np.sin(np.pi * i / res) * 0.1
        p[3] += -np.pi / 6 + np.pi / 3 * np.sin(np.pi / 2 * i / res)
        run(env, p, duration=0.1 if i + 1 < res else 1, grp=True)

def execute_fling(env):
    print("Action: Executing 1-time Fling and Release...")
    res = 40
    rad = 0.10
    top_rot = cfg['poses']['top_view'][3:]
    for i in range(res):
        if i < 20: rot = top_rot[1] + np.pi/6 - (np.pi/2.5+np.pi/6) * i / 20
        elif i < 30: rot = top_rot[1] - (np.pi/2.5)
        else: rot = top_rot[1] - (np.pi/2.5) * (40 - i) / 10
        
        p = [0.435 - rad * np.cos(2*np.pi/res*i), 0.0, 0.4 - rad * np.sin(2*np.pi/res*i), top_rot[0], rot, top_rot[2]]

        env.step(np.concatenate([p, [1.0 if i < 30 else 0.0]]))
        time.sleep(1/30)

# ================= Main Execution =================
def main():
    print("Loading SAM3 Model...")
    processor = Sam3Processor(build_sam3_image_model())


    task_input = str(input("Which task do you want to reset? [teddy-bear, towel-folding, towel-flattening]: "))
    if task_input not in cfg['tasks']:
        print("Invalid task name."); return
    
    t_cfg = cfg['tasks'][task_input]
    iteration = int(input("How many times you want to try?: "))

    env = RobotEnv(
        action_space="cartesian_position", 
        gripper_action_space="position", 
        camera_kwargs=dict(hand_camera=dict(image=True, resolution=(W, H)))
    )

    for i in range(iteration):
        run(env, cfg['poses']['top_view'], duration=2.0, grp=False)
        print(f"\n--- Cycle {i+1} | Task: {task_input} ---")
        input(">>> [SAFETY CHECK] Press ENTER to start reset...")


        obs = env.get_observation()
        cam = cfg['camera']['primary_id']
        img_array = obs["image"][cam][..., :3]
        if img_array.max() <= 1.0: img_array = (img_array * 255).astype(np.uint8)
        
        image_pil = Image.fromarray(img_array)
        b, g, r = image_pil.split()
        image_pil = Image.merge("RGB", (r, g, b))

        out = processor.set_text_prompt(state=processor.set_image(image_pil), prompt=t_cfg['prompt'])

        if len(out["boxes"]) > 0:

            best_idx = torch.argmax(out["scores"])
            box = out["boxes"][best_idx].cpu().numpy()
            if np.max(box) <= 1.5: box *= [W, H, W, H]
            
            u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            intr = obs["camera_intrinsics"][cam]
            ext = obs["camera_extrinsic"][cam] if "camera_extrinsic" in obs else np.eye(4)
            
            Xc = (u - intr[0, 2]) * g_cfg['grasp_z'] / intr[0, 0]
            Yc = (v - intr[1, 2]) * g_cfg['grasp_z'] / intr[1, 1]
            p_base = (ext @ np.array([Xc, Yc, g_cfg['grasp_z'], 1.0]))[:3]


            grasp_pose = [
                g_cfg['base_x'] + p_base[1] + t_cfg['edge_offset_y'], 
                p_base[0] + t_cfg['edge_offset_x'], 
                g_cfg['grasp_z'], 
                *cfg['poses']['top_view'][3:]
            ]

            run(env, grasp_pose, duration=2.0, grp=False)
            run(env, grasp_pose, duration=2.0, grp=True)


            if t_cfg['action_type'] == "unfold":
                execute_unfold(env, grasp_pose)
            
            elif t_cfg['action_type'] == "fling":
                execute_fling(env)
            
            elif t_cfg['action_type'] == "place":

                place_pose = [grasp_pose[0], grasp_pose[1] + g_cfg['place_x_shift'], g_cfg['place_z'], *cfg['poses']['top_view'][3:]]
                run(env, place_pose, duration=2.0, grp=True)
                run(env, place_pose, duration=1.0, grp=False)

            run(env, cfg['poses']['top_view'], duration=2.0, grp=False)
            print(f"Cycle {i+1} Finished.")
        else:
            print(f"Object '{t_cfg['prompt']}' not found.")

if __name__ == "__main__":
    main()