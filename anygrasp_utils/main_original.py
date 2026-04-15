import numpy as np
from gsnet import AnyGrasp
from droid.robot_env import RobotEnv
from visualization import draw_grasp_axes
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import time
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from scipy.spatial.transform import Rotation as Rot
class Cfg: pass
cfgs = Cfg()
cfgs.checkpoint_path = "../log/checkpoint_detection.tar"
cfgs.max_gripper_width = 0.1
cfgs.gripper_height = 0.03
cfgs.top_down_grasp = True
cfgs.debug = False
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

def demo(env):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    # depth_img = Image.open(os.path.join(data_dir, 'depth.png'))
    # depths = np.array(depth_img)

    obs = env.get_observation()
    colors = obs['image']['14013996_left']
    depths = obs['depth']['14013996_left']
    print(colors.shape, depths.shape)

    # print("depth mode:", depth_img.mode, "dtype:", depths.dtype, "min/max:", depths.min(), depths.max())

    # TODO: replace with your camera intrinsics
    fx = 732.277771
    fy = 732.277771
    cx = 614.49353027
    cy = 352.48422241
    scale = 1000.0  # mm->m

    lims = [-0.5, 0.5, -0.5, 0.5, 0.02, 1.5]

    xmap, ymap = np.meshgrid(np.arange(depths.shape[1]), np.arange(depths.shape[0]))
    points_z = depths.astype(np.float32) / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0.02) & (points_z < 1.5) & np.isfinite(points_z)
    points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    cols = colors[mask].astype(np.float32)

    print("points range:", points.min(axis=0), points.max(axis=0), "N:", len(points))

    gg, cloud = anygrasp.get_grasp(points, cols, lims=lims,
                                   apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print("No grasp detected.")
        return None

    gg = gg.nms().sort_by_score()
    print("top score:", gg[0].score)
    return gg




if __name__ == "__main__": 
    import imageio 
    from cam_config import e_T_c, gprime_T_g


        # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 Loaded")

    def depth_process(curr_frame: np.ndarray, processor, text_prompt: str = "gold lid",
    background_depth_value=None, score_threshold=None) -> np.ndarray:


        print("Target object: ", text_prompt)
        
        if not isinstance(curr_frame, np.ndarray):
            raise TypeError("curr_frame must be a numpy.ndarray")

        if curr_frame.ndim != 3 or curr_frame.shape[-1] != 4:
            raise ValueError(f"Expected curr_frame shape (H, W, 4), got {curr_frame.shape}")

        # Split BGRD
        bgr = curr_frame[..., :3]
        depth = curr_frame[..., 3].copy()

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

        # No detection -> just return RGBD with RGB converted
        if scores_np is None or len(scores_np) == 0:
            processed_curr_frame = np.dstack([rgb, depth])
            return processed_curr_frame

        best_idx = int(np.argmax(scores_np))
        best_score = scores_np[best_idx]

        if score_threshold is not None and best_score < score_threshold:
            processed_curr_frame = np.dstack([rgb, depth])
            return processed_curr_frame

        best_mask = masks_np[best_idx]
        best_box = boxes_np[best_idx]  # not used for processing, but useful for visualization

        # Squeeze if mask shape is (1, H, W)
        if best_mask.ndim == 3 and best_mask.shape[0] == 1:
            best_mask = best_mask[0]

        # Convert logits/probabilities to boolean
        if best_mask.dtype != np.bool_:
            best_mask = best_mask > 0.5

        # Choose large background depth
        if background_depth_value is None:
            finite_depth = depth[np.isfinite(depth)]
            if finite_depth.size == 0:
                large_depth = 1e6
            else:
                large_depth = finite_depth.max() + 1000
        else:
            large_depth = background_depth_value

        processed_depth = depth.copy()
        processed_depth[~best_mask] = large_depth

        # Return RGBD
        processed_curr_frame = np.dstack([rgb, processed_depth])

        return processed_curr_frame



    gripper_length = 0.17
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

    P3 = np.array([
        [0., 0., 1.],
        [0., 1., 0.],
        [-1., 0., 0.]
    ])
    obj_T_pose = np.eye(4)
    obj_T_pose[:3, :3] = P3
        
    curr_frame = env.get_observation()['image']['14013996_left']
    
   
    # Visualization
        
    gg = demo(env)
    R, t = gg.rotation_matrices, gg.translations
    R = R[:1]
    t = t[:1]
    overlay_frame = draw_grasp_axes(curr_frame, R, t)[:, :, :3][:, :, ::-1]
    print('R', R)
    print('t', t)

    # make 4x4 matrix 
    c_T_obj = make_T(R, t)
    imageio.imwrite("overlay_frame.png", overlay_frame)

    # b_T_dummy
    e_T_pose = e_T_c @ c_T_obj @ obj_T_pose
    e_T_dummy = e_T_pose @ pose_T_dummy
    b_T_e = base_to_gripper(top_pose.copy())
    b_T_dummy = b_T_e @ e_T_dummy
    b_T_pose = b_T_e @ e_T_pose
    print('dummy', b_T_dummy)
    print('pose', b_T_pose)

    trans = b_T_dummy[:3, -1:]
    rot = b_T_dummy[:3, :3]
    roll, pitch, yaw = rotation_matrix_to_rpy_zyx(rot)
    if yaw < -np.pi / 2:
        yaw += np.pi
    elif yaw > np.pi / 2:
        yaw -= np.pi
    print("X Y Z, R P Y", trans, roll, pitch, yaw)

    for _ in range(100):
        is_ok = input("Do you want to continue? (enter y)").lower() in ['y', 'yes'] 
        if is_ok:
            should_grasp = input("Should I grasp it? (enter y)").lower() in ['y', 'yes']
            if should_grasp: 
                # grasp 
                target_pose = top_pose.copy()

                target_pose[0:3] = trans
                # target_pose[2] += 0.2
                target_pose[3] = roll 
                target_pose[4] = pitch
                target_pose[5] = yaw                

                target_top_pose = top_pose.copy()
                target_top_pose[0:2] = trans[0:2]

                run(env, target_top_pose, duration=2, grip_close=0, hz=15)

                inter_poses = interpolate_poses(target_top_pose, target_pose, 10)
                input('how do you think?')
                for i, pose in enumerate(inter_poses):
                    pose = np.array([float(val) for val in pose])
                    if i == 0:
                        run(env, pose, duration=0.5, grip_close=0, hz=15)
                    else:
                        run(env, pose, duration=0.25, grip_close=0, hz=15)

                run(env, target_pose, duration=2, grip_close=0, hz=15)
                run(env, target_pose, duration=2, grip_close=1, hz=15)
                run(env, home_pose, duration=2, grip_close=1)

                input("reset?")

                run(env, top_pose, duration=3, grip_close=0)
            else: 
                # detect again 
                curr_frame = env.get_observation()['image']['14013996_left']
                gg = demo(env)
                R, t = gg.rotation_matrices, gg.translations
                R = R[:1]
                t = t[:1]
                print(t[:1])
                overlay_frame = draw_grasp_axes(curr_frame, R, t)[:, :, :3][:, :, ::-1]

                # make 4x4 matrix 
                c_T_obj = make_T(R, t)
                imageio.imwrite("overlay_frame.png", overlay_frame)

                e_T_pose = e_T_c @ c_T_obj @ obj_T_pose
                e_T_dummy = e_T_pose @ pose_T_dummy
                b_T_e = base_to_gripper(top_pose.copy())
                b_T_dummy = b_T_e @ e_T_dummy
                b_T_pose = b_T_e @ e_T_pose
                print('dummy', b_T_dummy)
                print('pose', b_T_pose)

                trans = b_T_dummy[:3, -1:]
                rot = b_T_dummy[:3, :3]
                roll, pitch, yaw = rotation_matrix_to_rpy_zyx(rot)
                if yaw < -np.pi / 2:
                    yaw += np.pi
                elif yaw > np.pi / 2:
                    yaw -= np.pi
                print("X Y Z, R P Y", trans, roll, pitch, yaw)
        else: 
            # detect again 
            curr_frame = env.get_observation()['image']['14013996_left']
            gg = demo(env)
            R, t = gg.rotation_matrices, gg.translations
            R = R[:1]
            t = t[:1]
            print(t[:1])
            overlay_frame = draw_grasp_axes(curr_frame, R, t)[:, :, :3][:, :, ::-1]

            # make 4x4 matrix 
            c_T_obj = make_T(R, t)
            imageio.imwrite("overlay_frame.png", overlay_frame)

            # b_T_dummy
            e_T_pose = e_T_c @ c_T_obj @ obj_T_pose
            e_T_dummy = e_T_pose @ pose_T_dummy
            b_T_e = base_to_gripper(top_pose.copy())
            b_T_dummy = b_T_e @ e_T_dummy
            b_T_pose = b_T_e @ e_T_pose
            print('dummy', b_T_dummy)
            print('pose', b_T_pose)

            trans = b_T_dummy[:3, -1:]
            rot = b_T_dummy[:3, :3]
            roll, pitch, yaw = rotation_matrix_to_rpy_zyx(rot)
            if yaw < -np.pi / 2:
                yaw += np.pi
            elif yaw > np.pi / 2:
                yaw -= np.pi
            print("X Y Z, R P Y", trans, roll, pitch, yaw)

