"""
Configuration for the table reset policy (T-L-001).

Zone coordinates, grid slots, AnyGrasp parameters, robot poses, and constants.
Zone boundaries are calibrated once before experiments and stored here.
"""

import numpy as np


# =============================================================================
# Camera Intrinsics (ZED, 1280x720)
# =============================================================================
CAMERA_INTRINSICS = dict(
    fx=732.277771,
    fy=732.277771,
    cx=614.49353027,
    cy=352.48422241,
)
CAMERA_RESOLUTION = (1280, 720)
CAMERA_ID = "11049903_left"
DEPTH_SCALE = 1000.0  # depth image unit: mm -> m

# =============================================================================
# EEF-to-Camera Transform (from hand-eye calibration)
# =============================================================================
E_T_C = np.array([
    [-0.04530642,  0.93352235,  0.35564497, -0.04881182],
    [-0.99895582, -0.04024085, -0.02163219,  0.05729848],
    [-0.00588267, -0.35625369,  0.93437075,  0.05056935],
    [ 0.0,         0.0,         0.0,         1.0],
])

# Camera is mounted upside-down: 180 deg rotation around optical (Z) axis
R_Z_180_4x4 = np.array([
    [-1, 0, 0, 0],
    [ 0,-1, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, 0, 0, 1],
], dtype=float)

# =============================================================================
# Gripper Parameters
# =============================================================================
GRIPPER_LENGTH = 0.15          # gripper finger length offset (m)
GRIPPER_Y_OFFSET = -0.035      # gprime_T_g lateral offset

# =============================================================================
# AnyGrasp Configuration
# =============================================================================
ANYGRASP_CHECKPOINT = "../log/checkpoint_detection.tar"
ANYGRASP_MAX_GRIPPER_WIDTH = 0.1
ANYGRASP_GRIPPER_HEIGHT = 0.03
ANYGRASP_TOP_DOWN = True
ANYGRASP_LIMS = [-0.5, 0.5, -0.5, 0.5, 0.02, 1.5]  # [xmin,xmax,ymin,ymax,zmin,zmax]

# =============================================================================
# Robot Poses (in base frame: [x, y, z, roll, pitch, yaw])
# =============================================================================
TOP_POSE = [
    0.4350, 0.0, 0.75,
    -3.135387735082638, 0.0117385168564208, -0.008736370437155985,
]

HOME_POSE = [
    0.4350, 0.0, 0.4,
    -3.135387735082638, 0.0117385168564208, -0.008736370437155985,
]

Z_SAFETY_FLOOR = 0.23  # minimum z height (m) to avoid table collision

# =============================================================================
# Motion Parameters
# =============================================================================
MOTION_HZ = 15          # control frequency
GRASP_CLOSE_DURATION = 2.0   # seconds to hold gripper closed
GRASP_OPEN_DURATION = 2.0    # seconds to hold gripper open
LIFT_DURATION = 2.0
MOVE_DURATION = 3.0

# =============================================================================
# Zone Definitions (in robot base frame, calibrated before experiments)
#
# These are PLACEHOLDER values. Run calibration to fill in actual coordinates.
# Each zone is defined as [x_min, x_max, y_min, y_max] in base frame.
# =============================================================================
YELLOW_PLATE_ZONE = [0.35, 0.55, -0.30, -0.10]   # left side (food target)
BLUE_TRAY_ZONE    = [0.35, 0.55,  0.10,  0.30]    # right side (tool target)
ALLOWED_ZONE      = [0.35, 0.55, -0.10,  0.10]    # center (initial placement)

# Surface heights (z in base frame) for plate/tray — used to filter out
# the container itself when sending zone point clouds to AnyGrasp.
YELLOW_PLATE_SURFACE_Z = 0.20   # placeholder, calibrate
BLUE_TRAY_SURFACE_Z    = 0.20   # placeholder, calibrate

# =============================================================================
# Allowed Zone Grid (3x3)
#
# 7 objects placed in 7 of 9 slots. Slots are numbered left-to-right,
# top-to-bottom (row-major). Coordinates are slot centers in base frame.
# =============================================================================
def _generate_grid_slots(zone_bounds, rows=3, cols=3, place_z=0.25):
    """Generate grid slot center coordinates within the Allowed Zone."""
    x_min, x_max, y_min, y_max = zone_bounds
    x_centers = np.linspace(x_min + 0.03, x_max - 0.03, rows)
    y_centers = np.linspace(y_min + 0.03, y_max - 0.03, cols)
    slots = []
    for x in x_centers:
        for y in y_centers:
            slots.append([x, y, place_z])
    return slots

GRID_SLOTS = _generate_grid_slots(ALLOWED_ZONE)

# =============================================================================
# SAM3 Configuration
# =============================================================================
SAM3_OBJECT_PROMPT = "bread, orange, apple, twine, screwdriver, lock, pen"
SAM3_ALL_PROMPT = "bread, orange, apple, twine, screwdriver, lock, pen, yellow plate, blue tray"
SAM3_SCORE_THRESHOLD = 0.3

# Object names for reference
FOOD_OBJECTS = ["bread", "orange", "apple"]
TOOL_OBJECTS = ["twine", "screwdriver", "lock", "pen"]
ALL_OBJECTS = FOOD_OBJECTS + TOOL_OBJECTS
CONTAINER_NAMES = ["yellow plate", "blue tray"]

# =============================================================================
# Reset Pipeline Parameters
# =============================================================================
MAX_VERIFICATION_LOOPS = 3   # Phase 4<->5 max iterations
MAX_GRASP_RETRIES = 3        # per-object grasp retry limit
