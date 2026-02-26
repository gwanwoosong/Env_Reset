# 환경 초기화 프로그램


## 1. 설치 가이드

### SAM3 설정

본 프로젝트는 SAM3 (Segment Anything Model 3)를 사용합니다.  
아래 절차에 따라 SAM3를 설치해 주십시오.

#### 1) 모델 접근 권한 요청

SAM3 HuggingFace 모델에 대한 접근 권한을 먼저 요청해야 합니다.

https://huggingface.co/facebook/sam3



#### 2) Conda 환경 생성

권장: 별도의 conda 환경을 생성하여 설치하십시오.

```bash
conda create -n sam3 python=3.12
conda activate sam3
```
#### 3) PyTorch 설치 (CUDA 12.8 기준)

⚠️ 사용 중인 CUDA 버전에 따라 설치 명령어가 달라질 수 있습니다.
아래는 CUDA 12.8 기준 예시입니다.
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

#### 4) SAM3 설치
```
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### DROID 설정
먼저 DROID 공식 문서를 참고하여 기본적인 환경 설정을 완료해 주십시오.
[공식 DROID 문서](https://droid-dataset.github.io/droid/)



## 2. 환경 설정
프로그램을 실행하기 전, 로봇이 카메라를 정상적으로 인식하는지 확인해야 합니다. 아래 코드를 실행하여 연결된 카메라의 키(ID) 값을 확인하십시오.
``` python
from droid.robot_env import RobotEnv
env = RobotEnv(camera_kwargs=dict(hand_camera=dict(image=True, resolution=(1280, 720))))
print("사용 가능한 카메라 키(ID):", env.get_observation()['image'].keys())
```

## 3. 설정값 수정
config.yaml 파일을 열어 본인의 실험 환경에 맞춰 아래 파라미터들을 조정하십시오.

* `base_x`: 로봇의 X축 기준 위치.
* `grasp_z`: 물체를 잡기 위한 접근 높이 (기본값: 0.19m).
* `place_z`: 물체를 놓을 때의 안전 높이 (기본값: 0.185m). 
* `edge_offset`: 수건 모서리 등을 잡기 위한 정교한 오프셋 조정.

## 4. 실행 방법
설정이 완료되었다면 아래 명령어를 입력하여 프로그램을 실행합니다.
```
python main.py
```

태스크별 상세 설명 
1. Teddy Bear
목적: 흩어진 인형을 정해진 초기 위치로 옮깁니다.
동작: 인형의 정중앙을 잡고, 설정된 거리만큼 옆으로 이동한 뒤 바닥에 부드럽게 내려놓습니다.

2. Towel Flattenin
목적: 뭉쳐있는 수건을 공중에서 털어 넓게 펼칩니다.

동작: 수건을 잡고 위로 들어 올린 뒤, 고속 원형 궤적을 그리며 털어냅니다. 원심력을 극대화하기 위해 궤적의 끝부분에서 즉시 수건을 놓습니다.

3. Towel Folding
목적: 수건의 모서리를 잡아 바닥에 평평하게 펼쳐 놓습니다.
동작: 수건의 끝부분을 잡고, 부드러운 반원 형태의 아크 궤적을 그리며 바닥에 수건을 펴줍니다.