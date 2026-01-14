import pyzed.sl as sl
import math
import numpy as np

from PIL import Image

# ... [Your existing SAM3 and imports setup] ...
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
# image = Image.open("test_image.png").convert("RGB")

# inference_state = processor.set_image(image)
# Prompt the model with text
# output = processor.set_text_prompt(state=inference_state, prompt="cloth")

# Get the masks, bounding boxes, and scores
# masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
# print(boxes)

def get_zed_point_cloud_at_pixel(box, point_cloud_mat):
    """
    Reads the exact X, Y, Z from the ZED Point Cloud at the box center.
    
    Args:
        box: [x1, y1, x2, y2]
        point_cloud_mat: sl.Mat object containing MEASURE.XYZ
        
    Returns:
        (x, y, z) in METERS (default ZED unit) or None if invalid.
    """
    # 1. Calculate Center Pixel
    x1, y1, x2, y2 = box
    u_center = int((x1 + x2) / 2)
    v_center = int((y1 + y2) / 2)

    # 2. Query the ZED SDK
    # point_cloud_mat.get_value(x, y) returns (ERROR_CODE, [x, y, z, rgba])
    err, point3D = point_cloud_mat.get_value(u_center, v_center)

    if err == sl.ERROR_CODE.SUCCESS:
        x, y, z, _ = point3D
        
        # Check for invalid values (nan or inf)
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            return x, y, z
        else:
            return None # Depth missing at this specific pixel
    else:
        return None

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================

# 1. Initialize ZED Camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.set_from_serial_number(14013996)
init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Use ULTRA or NEURAL for best accuracy
init_params.coordinate_units = sl.UNIT.METER   # Results will be in Meters
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # Standard Robotics
# OR sl.COORDINATE_SYSTEM.IMAGE (Y Down, Z Forward) if you prefer computer vision coords

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED:", err)
    exit()

# 2. Prepare ZED Data Containers
image_zed = sl.Mat()
point_cloud = sl.Mat()
runtime_params = sl.RuntimeParameters()

print("Camera Open. grabbing frame...")

if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    # A. Retrieve RGB Image for SAM3
    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    # Get numpy array from ZED Mat (RGBA -> RGB)
    image_np = image_zed.get_data()[:, :, :3] 
    # Convert to PIL for your existing SAM3 code
    image_pil = Image.fromarray(image_np)
    image_pil.show()

    # --- RUN SAM3 INFERENCE HERE ---
    inference_state = processor.set_image(image_pil)
    output = processor.set_text_prompt(state=inference_state, prompt="cloth")
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    # -------------------------------

    # B. Retrieve Point Cloud (X, Y, Z, Color)
    # This is more accurate than just Depth because it corrects for lens distortion
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

    print("\n--- ZED Mini Real-Time Readings ---")
    
    for i, box in enumerate(boxes):
        # Clean up box format
        b = box.detach().cpu().numpy() if hasattr(box, 'detach') else box
        # (Add your normalization/xywh check logic here if needed)

        # C. Get 3D Coordinates directly from ZED SDK
        coords = get_zed_point_cloud_at_pixel(b, point_cloud)

        if coords:
            rx, ry, rz = coords
            print(f"Object {i} | 2D Center: ({int((b[0]+b[2])/2)}, {int((b[1]+b[3])/2)})")
            print(f"         | 3D Point : X={rx:.3f}m, Y={ry:.3f}m, Z={rz:.3f}m")
        else:
            print(f"Object {i} | Depth invalid/occluded.")

# Close camera
zed.close()