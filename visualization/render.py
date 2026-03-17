import io
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

arr = np.load(
    "/home/rllab2/gwanwoo/GraspVLA/visualization/trial-20250507120350_data.npy",
    allow_pickle=True,
                )

data=arr.item()
req=data["request"]

def to_numpy_image(x):

    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Image.Image):
        return np.array(x)
    if isinstance(x, list) and isinstance(x[0], bytes):
        img = Image.open(io.BytesIO(x[0])).convert("RGB")
        return np.array(img)

    # handle raw bytes
    if isinstance(x, bytes):
        img = Image.open(io.BytesIO(x)).convert("RGB")
        return np.array(img)
front = to_numpy_image(req["front_view_image"])
side = to_numpy_image(req["side_view_image"])

print("front:", front.shape, front.dtype)
print("side :", side.shape, side.dtype)

plt.figure()
plt.imshow(front)
plt.title("Front View")
plt.axis("off") 

plt.figure()
plt.imshow(side)
plt.title("Side View")
plt.axis("off")

plt.show()