import json
import os
import numpy as np
import torch
from collections import OrderedDict
from copy import deepcopy

from droid.robot_env import RobotEnv

action_space = "cartesian_position"
gripper_action_space = "position"


imsize = 256

camera_kwargs = dict(
    hand_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
    varied_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
)

env = RobotEnv(
    action_space=action_space,
    gripper_action_space=gripper_action_space,
    camera_kwargs=camera_kwargs
)