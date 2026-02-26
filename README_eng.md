# Environment_Reset 

## 1. Installation

### 1.1 Install Dependencies (SAM3)

This project depends on **SAM3**.  
Please follow the steps below to install it properly.

#### Step 1: Request Model Access

Before installing SAM3, make sure to request access to the official HuggingFace model:

https://huggingface.co/facebook/sam3

You may need to log in and accept the usage terms.


#### Step 2: Create a Conda Environment

```bash
conda create -n sam3 python=3.12
conda activate sam3
```
#### Step 3: Install PyTorch (CUDA 12.8)
⚠️ The PyTorch version may differ depending on your CUDA setup.
Please adjust the CUDA version accordingly if needed.

``` 
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

#### Step 4: Install SAM3

```
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### 1.2 Install Dependencies (DROID)
Please follow the official installation guide:

https://droid-dataset.github.io/droid/

Complete the DROID setup according to the documentation before running this project.


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

```
python main.py
```
