import numpy as np
import sys
import tty
import termios
import time


def run(env, pose6, duration=1.0, grip_close=False, hz=15):
    """
    pose6: [x,y,z,rx,ry,rz]
    grip_close: True==Close / False==Open
    """
    pose = np.array(pose6, dtype=np.float32)
    grip = np.array([1.0 if grip_close else 0.0], dtype=np.float32)
    action = np.concatenate([pose, grip], axis=0)
    for _ in range(int(duration * hz)):
        env.step(action)
        time.sleep(1 / hz)


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def teleop(env, start_pose, step=0.01, rot_step=0.05):
    """
    Keyboard teleop for 6-DOF EEF control + gripper.

    Controls:
      Translation:
        w/s : X +/-
        a/d : Y +/-
        q/e : Z +/-

      Rotation:
        i/k : RX +/-
        j/l : RY +/-
        u/o : RZ +/-

      Gripper:
        g   : toggle gripper (open/close)

      Other:
        r     : reset to start_pose
        Enter : confirm & return current pose
        Ctrl+C: abort

    Returns [x, y, z, rx, ry, rz]
    """
    pose = list(start_pose)
    grip_close = False

    print("\n" + "=" * 50)
    print("  TELEOP MODE")
    print("=" * 50)
    print("  Translation:  w/s=X  a/d=Y  q/e=Z")
    print("  Rotation:     i/k=RX j/l=RY u/o=RZ")
    print("  Gripper:      g=toggle")
    print("  Reset:        r")
    print("  Confirm:      Enter")
    print("=" * 50 + "\n")

    key_map = {
        # translation
        'w': (0, +step), 's': (0, -step),
        'a': (1, +step), 'd': (1, -step),
        'q': (2, +step), 'e': (2, -step),
        # rotation
        'i': (3, +rot_step), 'k': (3, -rot_step),
        'j': (4, +rot_step), 'l': (4, -rot_step),
        'u': (5, +rot_step), 'o': (5, -rot_step),
    }

    while True:
        grip_str = "CLOSED" if grip_close else "OPEN"
        print(f"\r  xyz=({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f})  "
              f"rpy=({pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f})  "
              f"grip={grip_str}    ", end='', flush=True)

        key = getch()

        if key in ('\r', '\n'):
            print(f"\n\n[TELEOP] Confirmed pose: {pose}")
            print(f"[TELEOP] Gripper: {grip_str}")
            break

        elif key == '\x03':  # Ctrl+C
            print("\n[TELEOP] Aborted.")
            break

        elif key == 'r':
            pose = list(start_pose)
            grip_close = False
            run(env, pose, duration=1.0, grip_close=grip_close)
            print("\n[TELEOP] Reset to start pose.\n")

        elif key == 'g':
            grip_close = not grip_close
            run(env, pose, duration=0.5, grip_close=grip_close)

        elif key in key_map:
            idx, delta = key_map[key]
            pose[idx] += delta
            run(env, pose, duration=step / 0.05, grip_close=grip_close)

    return pose


# ================= Usage =================
from droid.robot_env import RobotEnv

action_space = "cartesian_position"
gripper_action_space = "position"


imsize = 224

camera_kwargs = dict(
    hand_camera=dict(image=True, depth=True, resolution=(1280, 720), resize_func='cv2')
)

env = RobotEnv(
    action_space=action_space,
    gripper_action_space=gripper_action_space,
    camera_kwargs=camera_kwargs
)

start = [0.4350, 0.0, 0.5, -3.1354, 0.0117, -0.0087]
final_pose = teleop(env, start, step=0.01, rot_step=0.05)
print("Final pose:", final_pose)