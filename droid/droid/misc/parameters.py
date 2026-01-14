import os
from cv2 import aruco

# Robot Params #
nuc_ip = "172.16.0.1"
robot_ip = "172.16.0.2"
laptop_ip = "172.16.0.5"
sudo_password = "robot"
robot_type = "fr3"  # 'panda' or 'fr3'
# robot_serial_number = "309969-2538505"
# robot_serial_number = "309969-2538505"
robot_serial_number = "290102-47707"
#

# Camera ID's #
hand_camera_id = "99999999"
varied_camera_1_id = "99999999"
varied_camera_2_id = "99999999"
# hand_camera_id = "14013996"
# varied_camera_1_id = "32492097"
# varied_camera_2_id = "99999999"

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = "C12VSd4V67upAVz3i3UExatd2GgtXN"

# Code Version [DONT CHANGE] #
droid_version = "1.3"

