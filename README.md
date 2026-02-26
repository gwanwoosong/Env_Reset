# Environment_Reset 

## 1. Installation

DROID 


## 2. Environment Setting
Before running, verify that the robot can see the camera and that the IDs are correct:

```python
from droid.robot_env import RobotEnv

env = RobotEnv(camera_kwargs=dict(hand_camera=dict(image=True, resolution=(1280, 720))))
print("Available Camera Keys:", env.get_observation()['image'].keys())
```

## 3. Values to Change
Open `config.yaml` to adjust the following:
* `base_x`: Anchor X-position of your workspace.
* `grasp_z`: Height for approaching the object (Default: 0.19).
* `place_z`: Height for releasing the object (Default: 0.185).
* `edge_offset_y/x`: Precision offsets for grasping towel corners.

## 4. How to Run

**Teddy-Bear Reset:**
`python main.py --task teddy-bear --iters 5`

**Towel Flattening (Fling):**
`python main.py --task towel-flattening --iters 5`

**Towel Folding (Unfold):**
`python main.py --task towel-folding --iters 1`