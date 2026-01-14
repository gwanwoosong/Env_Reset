import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("test_image.png").convert("RGB")

inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="cloth")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(boxes)

import numpy as np
from PIL import ImageDraw, ImageFont

zed_intrinsics = {
    'fx': 700.0,  # Focal length x
    'fy': 700.0,  # Focal length y
    'cx': 640.0,  # Optical center x (usually W/2)
    'cy': 360.0   # Optical center y (usually H/2)
}

def get_zed_3d_coordinates(box, depth_map, intrinsics):
    """
    Calculates the 3D (X, Y, Z) coordinates of the bounding box midpoint.
    
    Args:
        box: [x1, y1, x2, y2]
        depth_map: 2D numpy array containing depth values (aligned with image).
        intrinsics: Dict with 'fx', 'fy', 'cx', 'cy'.
        
    Returns:
        (X, Y, Z) tuple representing real-world coordinates.
    """
    # 1. Calculate the 2D midpoint of the bounding box
    x1, y1, x2, y2 = box
    u_center = (x1 + x2) / 2
    v_center = (y1 + y2) / 2
    
    # 2. Extract Depth (Z) at the midpoint
    # Ensure indices are within bounds and integers
    h, w = depth_map.shape
    u_idx = int(np.clip(u_center, 0, w - 1))
    v_idx = int(np.clip(v_center, 0, h - 1))
    
    Z = depth_map[v_idx, u_idx]
    
    # Check for invalid depth (ZED often uses nan or inf for invalid points)
    if np.isnan(Z) or np.isinf(Z) or Z <= 0:
        print(f"Warning: Invalid depth at ({u_idx}, {v_idx})")
        return None

    # 3. Apply Pinhole Camera Formula to get X and Y
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    X = (u_center - cx) * Z / fx
    Y = (v_center - cy) * Z / fy
    
    return (X, Y, Z)


def draw_boxes_pil(image, boxes, scores=None, color="red", width=4, save_path="boxed.png"):
    """
    image: PIL.Image
    boxes: Tensor/np/list, shape (N,4)
    scores: optional, shape (N,)
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    boxes = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    if scores is not None:
        scores = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)

    W, H = img.size

    def looks_normalized(b):
        # if coords are mostly <= 1.5, treat as normalized
        return np.max(b) <= 1.5

    for i, b in enumerate(boxes):
        b = b.astype(float)

        # Decide if normalized
        if looks_normalized(b):
            b = b * np.array([W, H, W, H], dtype=float)

        # Decide xyxy vs xywh (heuristic)
        x1, y1, x2, y2 = b.tolist()
        if x2 <= x1 or y2 <= y1:
            # treat as xywh
            x, y, w, h = b.tolist()
            x1, y1, x2, y2 = x, y, x + w, y + h

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        if scores is not None:
            label = f"{scores[i]:.3f}"
            # simple label background
            tw, th = draw.textlength(label), 14
            draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 8, y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - th - 2)), label, fill="white")

    img.save(save_path)
    return img

# --- usage after your SAM3 call ---
# masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
boxed = draw_boxes_pil(image, boxes, scores=scores, save_path="boxed.png")
boxed.show()
print("Saved:", "boxed.png")

depth_map = np.ones((720, 1280)) * 2.5

print("\n--- 3D Coordinates (ZED Mini Frame) ---")
for i, box in enumerate(boxes):
    # Convert tensor to list if needed
    b = box.detach().cpu().numpy() if hasattr(box, 'detach') else box
    
    # Handle box normalization/format logic from your existing code if necessary
    # (Assuming 'b' here is [x1, y1, x2, y2] in pixels based on your existing drawing logic)
    
    coords_3d = get_zed_3d_coordinates(b, depth_map, zed_intrinsics)
    
    if coords_3d:
        X, Y, Z = coords_3d
        print(f"Object {i}: Midpoint Pixel=({(b[0]+b[2])/2:.1f}, {(b[1]+b[3])/2:.1f})")
        print(f"          3D Position  = X: {X:.3f}m, Y: {Y:.3f}m, Z: {Z:.3f}m")