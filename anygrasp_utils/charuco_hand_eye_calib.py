"""
charuco_hand_eye_calib.py
─────────────────────────
Drop-in replacement for the ArUco-based hand-eye calibration in test.ipynb.

Why ChArUco instead of plain ArUco?
  • Checkerboard corners are located with much higher sub-pixel accuracy than
    ArUco marker corners, giving more precise poses.
  • Many corners per frame → robust even when the board is partially occluded.
  • The board ID is redundant; any subset of visible squares can be used.

Usage in the notebook
─────────────────────
    from charuco_hand_eye_calib import perform_hand_eye_calibration, generate_charuco_board
    
    # (optional) print/save the board so you can print it at the right scale
    generate_charuco_board('charuco_board.png')
    
    T_eef_to_cam = perform_hand_eye_calibration(env, num_poses=30)
    print("T_eef_to_cam:\n", T_eef_to_cam)

Board printing guide
──────────────────────
    Default board:  5 × 7 squares, square = 40 mm, marker = 30 mm  (DICT_4X4_50)
    Total size:     5×40 = 200 mm  ×  7×40 = 280 mm  → fits on A4 in landscape
    Print at 100 % scale (no "fit to page" scaling in the printer dialog).
    Measure the squares with a ruler after printing and update square_length /
    marker_length if they differ from the nominal values.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ──────────────────────────────────────────────────────────────────────────────
# Utility: 6-D pose → 4×4 homogeneous matrix
# ──────────────────────────────────────────────────────────────────────────────

def xyzrpy_to_matrix(pose_6d):
    """
    Convert [x, y, z, roll, pitch, yaw] → 4×4 transformation matrix.
    Angles are in radians, extrinsic 'xyz' convention.
    """
    x, y, z, rx, ry, rz = pose_6d
    R_mat = Rotation.from_euler('xyz', [rx, ry, rz], degrees=False).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [x, y, z]
    return T


# ──────────────────────────────────────────────────────────────────────────────
# Board factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_charuco_board(squares_x=9, squares_y=14,
                         square_length=0.02, marker_length=0.015,
                         aruco_dict_id=cv2.aruco.DICT_5X5_50):
    """
    Create a cv2.aruco.CharucoBoard for the user's 9x14 board.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    cv_ver = tuple(int(x) for x in cv2.__version__.split('.')[:2])
    
    if cv_ver >= (4, 7):
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, aruco_dict
        )
    else:
        # Legacy: cv2.aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary)
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_length, marker_length, aruco_dict
        )
    return board, aruco_dict


def generate_charuco_board(save_path='charuco_board_9x14.png',
                            squares_x=9, squares_y=14,
                            square_length=0.02, marker_length=0.015,
                            aruco_dict_id=cv2.aruco.DICT_5X5_50,
                            dpi=300):
    """
    Save the user's 9x14 ChArUco board image.
    """
    board, _ = _make_charuco_board(squares_x, squares_y,
                                    square_length, marker_length,
                                    aruco_dict_id)
    board_w_m = squares_x * square_length
    board_h_m = squares_y * square_length
    px_per_m = dpi / 0.0254
    img_w = int(round(board_w_m * px_per_m))
    img_h = int(round(board_h_m * px_per_m))

    if hasattr(board, 'generateImage'):
        # 4.7+
        board_img = board.generateImage((img_w, img_h), marginSize=20, borderBits=1)
    else:
        # 4.6
        board_img = board.draw((img_w, img_h), marginSize=20, borderBits=1)
    
    cv2.imwrite(save_path, board_img)
    print(f"ChArUco board (9x14) saved to '{save_path}'.")
    return board_img


# ──────────────────────────────────────────────────────────────────────────────
# Core calibration function
# ──────────────────────────────────────────────────────────────────────────────

def perform_hand_eye_calibration(
        env,
        num_poses=30,
        squares_x=14,
        squares_y=9,
        square_length=0.02,     # metres — measure from your printed board!
        marker_length=0.015,    # metres — must be < square_length
        aruco_dict_id=cv2.aruco.DICT_5X5_50,
        min_corners=30,         # Reject frames with fewer than 10 detected corners
        method=cv2.CALIB_HAND_EYE_TSAI,
):
    """
    Hand-eye calibration using the user's 9x14 board specs.

    Parameters
    ----------
    env           : robot environment (must expose get_observation() and step())
    num_poses     : how many valid robot poses to collect
    squares_x/y   : ChArUco board dimensions (must match the printed board)
    square_length : physical checker square size [m]
    marker_length : physical ArUco marker size embedded within each square [m]
    aruco_dict_id : ID of the ArUco dictionary used (e.g., cv2.aruco.DICT_5X5_50)
    min_corners   : minimum number of ChArUco corners required to accept a pose
    method        : cv2.calibrateHandEye method flag

    Returns
    -------
    T_eef_to_cam : 4×4 numpy float64 array
        Homogeneous transform from end-effector frame to camera frame.
    """

    # ── 1. Camera intrinsics ──────────────────────────────────────────────────
    camera_matrix = np.array([
        [732.277771, 0.0,        614.49353027],
        [0.0,        732.277771, 352.48422241],
        [0.0,        0.0,        1.0         ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # ── 2. Build ChArUco board & detector ────────────────────────────────────
    cv_ver = tuple(int(x) for x in cv2.__version__.split('.')[:2])
    board, aruco_dict = _make_charuco_board(squares_x, squares_y,
                                             square_length, marker_length)

    if cv_ver >= (4, 7):
        # Modern API
        aruco_params   = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params,
                                                      aruco_params)
    else:
        # Legacy API (OpenCV < 4.7)
        try:
            aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            aruco_params = cv2.aruco.DetectorParameters()
        charuco_detector = None   # we use legacy functions in the loop

    # ── 3. Storage for calibration pairs ─────────────────────────────────────
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam   = []
    t_target2cam   = []

    poses_collected = 0
    print(f"[ChArUco] Starting data collection. Need {num_poses} valid poses...")

    # Robot base pose + narrower randomization (better for small markers/close-up)
    top_pose = [
        0.4350, 0.0, 0.75,
        -3.135387735082638, 0.0117385168564208, -0.008736370437155985,
    ]
    # Narrowed XYZ translation (±10cm) and rotation (±14 deg) for better FoV coverage
    low  = np.array([-0.1, -0.1, -0.1, -0.25, -0.25, -0.25])
    high = np.array([ 0.1,  0.1,  0.0,  0.25,  0.25,  0.25])

    # Import run lazily so this module can be imported stand-alone too
    try:
        from __main__ import run as _run
    except ImportError:
        _run = None

    while poses_collected < num_poses:

        # ── 3a. Sample pose & move robot ─────────────────────────────────────
        sample  = np.random.uniform(low=low, high=high)
        pose_6d = np.array(top_pose) + sample

        if _run is not None:
            _run(env, pose_6d, 5, False, 15)
        else:
            print("WARNING: 'run' not found in __main__; skipping robot motion.")

        T_base_to_eef = xyzrpy_to_matrix(pose_6d)

        # ── 3b. Capture image ─────────────────────────────────────────────────
        obs  = env.get_observation()
        img  = obs['image']['11049903_left'][..., :3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── 3c. Detect ChArUco corners ────────────────────────────────────────
        if cv_ver >= (4, 7):
            # Returns (charucoCorners, charucoIds, markerCorners, markerIds)
            charuco_corners, charuco_ids, _mcorners, _mids = \
                charuco_detector.detectBoard(gray)
        else:
            # Two-step legacy detection
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params)
            if marker_ids is None or len(marker_ids) == 0:
                charuco_corners, charuco_ids = None, None
            else:
                _, charuco_corners, charuco_ids = \
                    cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, board,
                        cameraMatrix=camera_matrix,
                        distCoeffs=dist_coeffs,
                    )

        # ── 3d. Quality gate ──────────────────────────────────────────────────
        n_detected = 0 if charuco_ids is None else len(charuco_ids)
        if charuco_corners is None or charuco_ids is None or n_detected < min_corners:
            print(f"[ChArUco] Only {n_detected} corners detected "
                  f"(need >= {min_corners}). Resampling pose...")
            continue

        print(f"[ChArUco] Detected {n_detected} corners out of "
              f"{(squares_x - 1) * (squares_y - 1)} possible.")

        # ── 3e. Estimate board pose with PnP ─────────────────────────────────
        if hasattr(board, 'getChessboardCorners'):
            all_corners = board.getChessboardCorners()
        else:
            all_corners = board.chessboardCorners
            
        obj_pts = all_corners[charuco_ids.flatten()].astype(np.float32)
        img_pts = charuco_corners.astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            print("[ChArUco] solvePnP failed. Resampling pose...")
            continue
            
        # ── [DEBUG] Visualize detected axes ──────────────────────────────────
        # Draw 10cm axes: Red=X, Green=Y, Blue=Z
        vis_img = img.copy()
        if hasattr(cv2, 'drawFrameAxes'):
            cv2.drawFrameAxes(vis_img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        else:
            # Legacy OpenCV < 4.5.1
            cv2.aruco.drawAxis(vis_img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            
        debug_filename = f'debug_pose_{poses_collected}.png'
        cv2.imwrite(debug_filename, vis_img)
        print(f"[ChArUco] Detected {len(charuco_ids)} corners. Saved visualization to {debug_filename}")

        # ── 3f. Store the pose pair ───────────────────────────────────────────
        R_cam, _ = cv2.Rodrigues(rvec)
        t_cam    = tvec.reshape(3, 1)

        R_target2cam.append(R_cam)
        t_target2cam.append(t_cam)

        R_eef = T_base_to_eef[:3, :3]
        t_eef = T_base_to_eef[:3, 3].reshape(3, 1)
        R_gripper2base.append(R_eef)
        t_gripper2base.append(t_eef)

        poses_collected += 1
        print(f"[ChArUco] Collected pose {poses_collected}/{num_poses}")

    print("[ChArUco] Data collection complete. Running hand-eye calibration...")

    # ── 4. Hand-eye calibration ───────────────────────────────────────────────
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,   t_target2cam,
        method=method,
    )

    # ── 5. Build 4×4 result matrix ────────────────────────────────────────────
    T_eef_to_cam = np.eye(4)
    T_eef_to_cam[:3, :3] = R_cam2gripper
    T_eef_to_cam[:3, 3]  = t_cam2gripper.flatten()

    print("\n" + "="*50)
    print("Hand-Eye Calibration Complete")
    print("="*50)
    print("\nCopy-paste friendly NumPy matrix (T_eef_to_cam):")
    print("np.array([")
    for row in T_eef_to_cam:
        print(f"    [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f}],")
    print("])")
    print("="*50 + "\n")

    return T_eef_to_cam


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    # Try to import RobotEnv (adjust path if droid is not in PYTHONPATH)
    try:
        from droid.robot_env import RobotEnv
    except ImportError:
        print("ERROR: Could not import RobotEnv. "
              "Please make sure the 'droid' folder is in your PYTHONPATH.")
        sys.exit(1)

    # 1. Initialize environment
    camera_kwargs = dict(
        hand_camera=dict(image=True, depth=True, resolution=(1280, 720), resize_func='cv2')
    )
    env = RobotEnv(
        action_space="cartesian_position",
        gripper_action_space="position",
        camera_kwargs=camera_kwargs
    )
    
    # Define a simple move function for the calibration loop to use
    def run(env, pose6, duration=1.0, grip_close=False, hz=15):
        import time
        pose = np.array(pose6, dtype=np.float32)
        grip = np.array([1.0 if grip_close else 0.0], dtype=np.float32)
        action = np.concatenate([pose, grip], axis=0)
        for _ in range(int(duration * hz)):
            env.step(action)
            time.sleep(1.0 / hz)

    # Attach 'run' to __main__ so perform_hand_eye_calibration can find it
    import __main__
    __main__.run = run

    # 2. Run Calibration
    print("\n--- Starting ChArUco Calibration ---")
    T_result = perform_hand_eye_calibration(env, num_poses=30)
    
    # 3. Save result
    np.save('T_eef_to_cam.npy', T_result)
    print("\nCalibration result saved to 'T_eef_to_cam.npy'")
