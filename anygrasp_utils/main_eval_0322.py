
from droid.robot_env import RobotEnv
from visualization import draw_grasp_axes
import numpy as np
import open3d as o3d
from PIL import Image
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import time
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
class Cfg: pass
cfgs = Cfg()
cfgs.checkpoint_path = "../log/checkpoint_detection.tar"
cfgs.max_gripper_width = 0.1
cfgs.gripper_height = 0.03
cfgs.top_down_grasp = True
cfgs.debug = True
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))


home_pose = [   0.4350, # x
                0.0,    # y
                0.4,    # z
                -3.135387735082638, # roll - rotation around x-axis
                # -3.135387735082638, # roll - rotation around x-axis
                0.0117385168564208, # pitch - rotation around y-axis
                -0.008736370437155985] # yaw - rotation around z-axis

top_pose = [0.4350,
    0.0,
    0.75,
    -3.135387735082638,
    0.0117385168564208,
    -0.008736370437155985]

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def waypoint_pose_xyz_rpy(env, start_pose, trans_xyz, target_rot_mat,
                          steps=1, grip_close=False, hz=15, step_duration=4):
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

def normalize_rpy_to_reference(ref_rpy, target_rpy):
    """
    Adjust each target Euler angle so it is within [-pi, pi] of the reference.
    Ensures the robot takes the shortest rotation path from ref to target.
    """
    ref = np.array(ref_rpy, dtype=float)
    result = np.array(target_rpy, dtype=float)
    for i in range(3):
        diff = result[i] - ref[i]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        result[i] = ref[i] + diff
    return result

def rotation_matrix_to_rpy_zyx(R_mat):
    rot = Rot.from_matrix(R_mat)
    roll, pitch, yaw = rot.as_euler('xyz', degrees=False)  # returns [yaw,pitch,roll]
    return roll, pitch, yaw

def rpy_zyx_to_rotation_matrix(roll, pitch, yaw):
    """
    Converts roll, pitch, yaw back into a 3x3 rotation matrix using the 'xyz' convention.
    """
    # Scipy expects the angles in the same order as the axes string ('xyz')
    rot = Rot.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return rot.as_matrix()

def base_to_gripper(eef_pose):
    """
    Converts a 6D end-effector pose [x, y, z, roll, pitch, yaw] 
    into a 4x4 transformation matrix using the 'xyz' convention.
    """
    x, y, z, roll, pitch, yaw = eef_pose
    
    # Build the 4x4 homogeneous transformation matrix
    T_base_to_gripper = np.eye(4)
    T_base_to_gripper[:3, :3] = rpy_zyx_to_rotation_matrix(roll, pitch, yaw)
    T_base_to_gripper[:3, 3] = [x, y, z]
    
    return T_base_to_gripper
# def rotation_matrix_to_rpy_zyx(R):
#     """
#     Convert 3x3 rotation matrix to roll, pitch, yaw (ZYX convention).
#     Returns angles in radians.
#     """

#     # Check if we are near singularity (cos(pitch) ≈ 0)
#     if abs(R[2, 0]) < 1.0 - 1e-8:
#         pitch = -np.arcsin(R[2, 0])
#         roll  = np.arctan2(R[2, 1], R[2, 2])
#         yaw   = np.arctan2(R[1, 0], R[0, 0])
#     else:
#         # Gimbal lock case
#         pitch = np.pi/2 if R[2, 0] <= -1 else -np.pi/2
#         roll  = np.arctan2(-R[0, 1], -R[0, 2])
#         yaw   = 0.0

#     return roll, pitch, yaw


def run(env, pose6, duration=1.0, grip_close=False, hz=15):
    """
        pose6: [x,y,z,rx,ry,rz]
        
        grip_close: True==Close / False==Open
    """
    pose= np.array(pose6, dtype=np.float32)
    grip= np.array([1.0 if grip_close else 0.0], dtype=np.float32)
    action = np.concatenate([pose, grip], axis=0)

    for _ in range(int(duration * hz)):
        env.step(action)
        time.sleep(1.0 / hz)

import numpy as np
def interpolate_poses(pose1, pose2, resolution, euler_axes='xyz'):
    """
    Interpolates between two 6D poses [x, y, z, roll, pitch, yaw].
    
    Args:
        pose1: List or array of the start pose [x, y, z, r, p, y]
        pose2: List or array of the end pose [x, y, z, r, p, y]
        resolution: Integer number of total poses in the output (including start and end)
        euler_axes: String defining the Euler angle convention (default is extrinsic 'xyz')
        
    Returns:
        A numpy array of shape (resolution, 6) containing the interpolated poses.
    """
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)
    
    # Extract translation and rotation
    pos1, rpy1 = pose1[:3], pose1[3:]
    pos2, rpy2 = pose2[:3], pose2[3:]
    
    # 1. Interpolate Translation (LERP)
    times = np.linspace(0, 1, resolution)
    interpolated_positions = (1 - times[:, None]) * pos1 + times[:, None] * pos2
    interpolated_positions = np.array(interpolated_positions)
    
    # 2. Interpolate Rotation (SLERP)
    # Define the key rotations and the "times" at which they occur
    key_rots = Rot.from_euler(euler_axes, [rpy1, rpy2])
    key_times = [0, 1]
    
    # Create the SLERP function and apply it to our target times
    slerp = Slerp(key_times, key_rots)
    interpolated_rotations = slerp(times)
    
    # Convert back to Euler angles
    interpolated_rpy = interpolated_rotations.as_euler(euler_axes)
    
    # Combine positions and rotations back into [x, y, z, r, p, y]
    interpolated_poses = np.hstack((interpolated_positions, interpolated_rpy))
    
    return interpolated_poses

def demo(env, p_depth: np.ndarray):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    # depth_img = Image.open(os.path.join(data_dir, 'depth.png'))
    # depths = np.array(depth_img)

    obs = env.get_observation()
    colors = obs['image']['11049903_left'][:, :, :3] / 255.0
    
    # Visualization Purpose
    print("Visualization, No post processing depth")
    depths = obs['depth']['11049903_left']
    
    depths = p_depth
    # colors, depths should be processed!
    # print(colors.shape, depths.shape)

    # print("depth mode:", depth_img.mode, "dtype:", depths.dtype, "min/max:", depths.min(), depths.max())

    # TODO: replace with your camera intrinsics
    fx = 732.277771
    fy = 732.277771
    cx = 614.49353027
    cy = 352.48422241
    scale = 1000.0  # mm->m

    lims = [-0.5, 0.5, -0.5, 0.5, 0.02, 1.5]

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0.02) & (points_z < 1.5) & np.isfinite(points_z)

    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    print(points.min(), points.max())
    colors = colors[mask].astype(np.float32)

    # print("points range:", points.min(axis=0), points.max(axis=0), "N:", len(points))

    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims,
                                   apply_object_mask=True, dense_grasp=False, collision_detection=True)
    print("gg, cloud succedded")
    if len(gg) == 0:
        print("No grasp detected.")
        return None

    gg = gg.nms().sort_by_score()
    print("sorted")

    print("top score:", gg[0].score)

    # if cfgs.debug:
    trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    cloud.transform(trans_mat)
    grippers = gg.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(trans_mat)
    o3d.visualization.draw_geometries([grippers[0], cloud])

    return gg




if __name__ == "__main__":
    import imageio
    import argparse
    from cam_config import e_T_c, gprime_T_g

    parser = argparse.ArgumentParser()
    parser.add_argument('--failed', action='store_true',
                        help='Use center-depth-reduce for flat/reflective objects (e.g. pot lid)')
    args = parser.parse_args()
    

        # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 Loaded")

    def depth_process(curr_frame: np.ndarray, curr_depth: np.ndarray, text_prompt: str = "gold lid",
    background_depth_value=None, score_threshold=None,
    center_reduce_amount: int = 0, center_reduce_radius: int = 50) -> np.ndarray:


        print("Target object: ", text_prompt)
        
        if not isinstance(curr_frame, np.ndarray):
            raise TypeError("curr_frame must be a numpy.ndarray")

        if curr_frame.ndim != 3 or curr_frame.shape[-1] != 4:
            raise ValueError(f"Expected curr_frame shape (H, W, 4), got {curr_frame.shape}")

        # Split BGRD
        bgr = curr_frame[..., :3]
        depth = curr_depth

        # BGR -> RGB
        rgb = bgr[..., ::-1].copy()

        # SAM3 expects image input
        image = Image.fromarray(rgb.astype(np.uint8))
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        def to_numpy(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            return np.array(x)

        masks_np = to_numpy(masks)
        boxes_np = to_numpy(boxes)
        scores_np = to_numpy(scores)

        # No detection -> signal failure
        if scores_np is None or len(scores_np) == 0:
            print("[SAM3] No object detected.")
            return None

        best_idx = int(np.argmax(scores_np))
        best_score = scores_np[best_idx]

        if score_threshold is not None and best_score < score_threshold:
            print(f"[SAM3] Best score {best_score:.3f} below threshold {score_threshold}. No detection.")
            return None

        best_mask = masks_np[best_idx]
        best_box = boxes_np[best_idx]  # not used for processing, but useful for visualization


        x1,y1,x2,y2 = best_box[:4].astype(int)
        box_mask = np.zeros(curr_depth.shape[:2], dtype=bool)
        box_mask [y1:y2, x1:x2] = True

        large_depth_value = curr_depth.max() + 1000  # adjust depending on your depth scale
        p_depth = depth.copy()
        p_depth[~box_mask] = large_depth_value

        # Optional: lower depth at center of detected object (helps for flat/reflective surfaces)
        if center_reduce_amount > 0:
            cx = (x1 + x2) // 2 
            cy = (y1 + y2) // 2 
            yy, xx = np.mgrid[0:curr_depth.shape[0], 0:curr_depth.shape[1]]
            circle_mask = ((xx - cx)**2 + (yy - cy)**2) <= center_reduce_radius**2
            region = circle_mask & box_mask
            p_depth[region] = np.maximum(p_depth[region].astype(np.int32) - center_reduce_amount, 1).astype(p_depth.dtype)
            print(f"[DBG] center_reduce applied: cx={cx}, cy={cy}, r={center_reduce_radius}, amount={center_reduce_amount}")

        return p_depth

    gripper_length = 0.190  # 0.05 is compensation
    pose_T_dummy = np.eye(4)
    pose_T_dummy[2, 3] = -gripper_length

    def make_T(R, t):
        """
        R: (3,3)
        t: (3,) or (3,1)
        returns: (4,4)
        """
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3]  = np.asarray(t).reshape(3,)
        return T

    # Camera, Env
    action_space = "cartesian_position"
    gripper_action_space = "position"

    camera_kwargs = dict(
        hand_camera=dict(image=True, depth=True, resolution=(1280, 720), resize_func='cv2')
    )

    env = RobotEnv(
        action_space=action_space,
        gripper_action_space=gripper_action_space,
        camera_kwargs=camera_kwargs
    )

    run(env, top_pose, duration=4, grip_close=0)
    R_y_minus_90 = np.array([
        [0., 0., -1.],
        [0., 1., 0.],
        [1., 0., 0.]
    ])
    R_y_90 = np.array([
        [0., 0., 1.],
        [0., 1., 0.],
        [-1., 0., 0.]
    ])
    R_z_minus_90 = np.array([
        [ 0.0,  1.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ])
    R_x_180 = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    R_z_180 = np.array([
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., 1.]
    ])
    # P3 = R_y_minus_90
    # P3 = R_z_minus_90 @ R_x_180
    P3 = R_y_90 # @ R_z_180
    obj_T_pose = np.eye(4)
    obj_T_pose[:3, :3] = P3
    P4 = np.eye(4)
    P4[:3, :3] = R_x_180
        
    def get_grasp_pose():
        curr_frame = env.get_observation()['image']['11049903_left']
        curr_depth = env.get_observation()['depth']['11049903_left']

        curr_pos = env.get_observation()['robot_state']['cartesian_position']

        while True:
            target = "gold lid"
            if args.failed:
                p_depth = depth_process(curr_frame, curr_depth, text_prompt=target,
                                        center_reduce_amount=12, center_reduce_radius=20)
            else:
                p_depth = depth_process(curr_frame, curr_depth, text_prompt=target)
            if p_depth is None:
                print("[SAM3] Object not detected. Please try a different description.")
                continue
            gg = demo(env, p_depth)
            if gg is None:
                print("[AnyGrasp] No grasp found. Please try again.")
                continue
            break

        R, t = gg.rotation_matrices, gg.translations
        R = R[:1]
        t = t[:1]
        overlay_frame = draw_grasp_axes(curr_frame, R, t)[:, :, :3][:, :, ::-1]
        # make 4x4 matrix 
        c_T_obj = make_T(R, t)
        
        imageio.imwrite("overlay_frame.png", overlay_frame)

        # Camera is mounted upside-down: apply 180 deg rotation around optical (Z) axis
        # This flips X and Y in camera frame to correct for physical mounting
        # R_z_180_4x4 = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
        # c_T_obj = R_z_180_4x4 @ c_T_obj

        # b_T_dummy
    # def get_bTd(top_pose, c_T_obj):
        c_T_dummy = c_T_obj @ obj_T_pose #@ pose_T_dummy # @ P4

        # ── DEBUG: step-by-step transform values ──────────────────────────
        print(f"[DBG] c_T_obj  t (cam frame)   x={c_T_obj[0,3]:.4f}  y={c_T_obj[1,3]:.4f}  z={c_T_obj[2,3]:.4f}  <- measure cam->obj depth with ruler")
        e_T_dummy = e_T_c @ c_T_dummy        
        
        print(f"[DBG] e_T_dummy t (EEF frame)  x={e_T_dummy[0,3]:.4f}  y={e_T_dummy[1,3]:.4f}  z={e_T_dummy[2,3]:.4f}  <- measure EEF->obj with ruler")
        b_T_e = base_to_gripper(np.array(curr_pos))
        
        # b_T_e = base_to_gripper(top_pose.copy())
        b_T_dummy = b_T_e @ e_T_dummy
        
        print(f"[DBG] b_T_dummy t (base frame)  x={b_T_dummy[0,3]:.4f}  y={b_T_dummy[1,3]:.4f}  z={b_T_dummy[2,3]:.4f}  <- measure base->obj with ruler")
        b_T_test = b_T_dummy @ pose_T_dummy
        
        print(f"[DBG] b_T_test  t (final cmd)   x={b_T_test[0,3]:.4f}  y={b_T_test[1,3]:.4f}  z={b_T_test[2,3]:.4f}")
        # ──────────────────────────────────────────────────────────────────

        trans_test = b_T_test[:3, -1:]
        test_rot = b_T_test[:3, :3]
        rt, rp, ry = rotation_matrix_to_rpy_zyx(test_rot)
        print('RPY', rt, rp, ry)
        if ry < -np.pi / 2 or ry > np.pi / 2:
            print('wrist limit hit!!#@#!#@!')
            R = np.eye(4)
            R[:3, :3] = R_z_180
            b_T_test = b_T_test @ R

        trans_test = b_T_test[:3, -1:]
        test_rot = b_T_test[:3, :3]
        rt, rp, ry = rotation_matrix_to_rpy_zyx(test_rot)
        # e_T_pose = e_T_c @ c_T_obj @ obj_T_pose
        # e_T_dummy = e_T_pose @ pose_T_dummy
        # b_T_e = base_to_gripper(top_pose.copy())
        # b_T_dummy = b_T_e @ e_T_dummy
        # b_T_pose = b_T_e @ e_T_pose
        # print('dummy', b_T_dummy)
        # return b_T_dummy

        # b_T_dummy = get_bTd(top_pose, c_T_obj)
        # print('pose', b_T_pose)

        trans = b_T_dummy[:3, -1:]
        rot = b_T_dummy[:3, :3]
        roll, pitch, yaw = rotation_matrix_to_rpy_zyx(rot)
        
        #     yaw += np.pi
        # elif yaw > np.pi / 2:
        #     yaw -= np.pi
        # print("X Y Z, R P Y", trans, roll, pitch, yaw)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import os
        
        t_list = [b_T_e, e_T_c, c_T_obj, obj_T_pose, pose_T_dummy]
        t_names = ['EEF', 'CAM', 'AG', 'P4', 'Offset']
                
        # from cam_config import e_T_c
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        # 1. 로봇 베이스 좌표계 (Base Frame at 0,0,0)
        ax.quiver(0,0,0, 0.2, 0, 0, color='r', linewidth=5) # X
        ax.quiver(0,0,0, 0, 0.2, 0, color='g', linewidth=5) # Y
        ax.quiver(0,0,0, 0, 0, 0.2, color='b', linewidth=5) # Z
        ax.text(0, 0, 0, " ROBOT BASE", color='black', fontweight='bold')
        # 2. 카메라 좌표계 위치 계산 및 표시
        idx = 0
        b_T_b = np.eye(4)
        for mat, name in zip(t_list, t_names):
            # b_T_e = base_to_gripper(top_pose.copy())

            # 카메라 위치에 축 표시
            b_T_b = b_T_b @ mat
            c_pos = b_T_b[:, 3]
            c_rot = b_T_b[:3, :3]
            ax.quiver(c_pos[0], c_pos[1], c_pos[2], c_rot[0,0]*0.1, c_rot[1,0]*0.1, c_rot[2,0]*0.1, color='r', alpha=0.5)
            ax.quiver(c_pos[0], c_pos[1], c_pos[2], c_rot[0,1]*0.1, c_rot[1,1]*0.1, c_rot[2,1]*0.1, color='g', alpha=0.5)
            ax.quiver(c_pos[0], c_pos[1], c_pos[2], c_rot[0,2]*0.1, c_rot[1,2]*0.1, c_rot[2,2]*0.1, color='b', alpha=0.5)
            ax.text(c_pos[0], c_pos[1], c_pos[2], name, color='darkgray')
            
            if idx == 0:
                ax.plot([0, c_pos[0]], [0, c_pos[1]], [0, c_pos[2]], 'k--', alpha=0.2)
            else:
                ax.plot([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]], [p_pos[2], c_pos[2]], 'k--', alpha=0.2)
            p_pos = c_pos
            idx += 1

        # 그래프 환경 설정
        ax.set_xlabel('Robot X (m)'); ax.set_ylabel('Robot Y (m)'); ax.set_zlabel('Robot Z (m)')
        # 로봇 작업 반경에 맞춰 범위 자동 조정 (예: 0~1m 범위)
        ax.set_xlim([0, 0.8]); ax.set_ylim([-0.4, 0.4]); ax.set_zlim([0, 1.0])
        ax.set_title(f'Robot Base World View)')
        ax.grid(True)
        ax.view_init(elev=30, azim=225) # 입체적으로 보이게 뷰 설정

        plt.show()
        # return trans, roll, pitch, yaw
        return trans_test, rt, rp, ry
    
    trans, roll, pitch, yaw = get_grasp_pose()
    trial_count=0
    for _ in range(100):
        is_ok = input("Do you want to continue? (enter y)").lower() in ['y', 'yes'] 
        if is_ok:
            # grasp

            target_pose = top_pose.copy()
            target_pose[0:3] = trans

            x_offset = 0.02
            target_pose[0] -= x_offset

            target_pose[2] = max(target_pose[2], 0.184)  # z safety floor
            target_pose[3] = roll
            target_pose[4] = pitch

            target_pose[5] = yaw

            target_top_pose = top_pose.copy()
            target_top_pose[0:2] = trans[0:2]
            target_top_pose[3] = roll
            target_top_pose[4] = pitch
            target_top_pose[5] = yaw

            trans_delta = np.array(target_top_pose[:3]) - np.array(top_pose[:3])
            target_rot_mat = rpy_zyx_to_rotation_matrix(roll, pitch, yaw)
            
            run(env, target_pose, duration=5, grip_close=0)

            run(env, target_pose, duration=2, grip_close=0, hz=15)
            run(env, target_pose, duration=2, grip_close=1, hz=15)
            run(env, top_pose, duration=2, grip_close=1)

            # Go through top_pose, then drop position, then back to top
            drop_pose = [0.5350, 0.2100, 0.4000] + top_pose[3:]
            drop_pose_v2 = [0.7550, 0.0, 0.4000] + top_pose[3:]
            run(env, top_pose, duration=3, grip_close=1)
            run(env, drop_pose, duration=3, grip_close=1)
            # Open gripper and wait 2 sec
            run(env, drop_pose, duration=2, grip_close=0)
            drop_high_z_pose = drop_pose.copy()
            drop_high_z_pose[2] = top_pose[2]
            run(env, drop_high_z_pose, duration=2, grip_close=0)
            # Return to top_pose (imaging position)
            run(env, top_pose, duration=3, grip_close=0)

        else: 
            trans, roll, pitch, yaw = get_grasp_pose()
            