import os
import sys

import cv2
import glm
import random
import numpy as np
import backgroundsubtractor
import xml.etree.ElementTree as ET
import calibration
import voxel_construction
block_size = 1.0

# Parameters for voxel positions function
# Initialization with loading videos and training background models
initialized = False
videos = []
bg_models = []
# Background model parameters for every camera
# figure_threshold, figure_inner_threshold,
# apply_opening_pre, apply_closing_pre, apply_opening_post, apply_closing_post
cam_bg_model_params = [
    [5000, 115, False, False, True, True],
    [5000, 115, False, False, True, True],
    [5000, 175, False, True, True, True],
    [5000, 115, False, False, False, True]
]
# Currently loaded frames and their index
current_frames = []
frame_count = 0
previous_masks = []
# Lookup table for voxels
lookup_table = None
voxel_points = None


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    list_videos = []
    # list_videos(cv2.VideoCapture(os.path.join("data", "cam" + str(cam + 1))))

    voxel_3d_points = voxel_construction.create_voxel_volume(width, height * 2, depth)
    lookup_table = voxel_construction.create_lookup_table(voxel_3d_points, 4, "data", "config.xml")

    # Extract foreground mask from video frame for each camera
    foregrounds = []
    cam_allframes = []
    for cam in range(1, 5):
        background_model = backgroundsubtractor.find_background(os.path.join("data", "cam" + str(cam)),
                                                                "background.avi", history=500, threshold=300,
                                                                detect_shadow=False)

        removed=backgroundsubtractor.substract_background(os.path.join("data", "cam" + str(cam)), "video.avi",
                                                  background_model, 10, True,
                                                  False)
        contoured_foreground = backgroundsubtractor.contouring(removed, outer_threshold=1000, inner_threshold=130)
        foregrounds.append(contoured_foreground)
        cap = cv2.VideoCapture( os.path.join(os.path.join("data", "cam" + str(cam)), "video.avi"))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            break
        cam_allframes.append(np.array(frames)[0])
    #voxel_visible,voxels_visible_colors= voxel_construction.visible_voxels_coloring(lookup_table, foregrounds,cam_allframes )
    voxel_visible,voxels_visible_colors= voxel_construction.visible_voxels_coloring(lookup_table, foregrounds,cam_allframes )

    allvisible_positions=[]
    colors=[]
    scaling_factor=64
    threshold=4 # if all on for the specific camera view
    for voxel, camera in voxel_visible.items():

        if sum(camera.values()) >= 4:
            # Swap y and z and flip sign of y
            x = voxel[0] / scaling_factor
            y = - (voxel[2] / scaling_factor)
            z = voxel[1] / scaling_factor
            allvisible_positions.append([x, y, z])


            # Use color of only 2nd camera (front) and convert to 0-1
            colors.append(voxels_visible_colors[voxel][2][::-1] / 255.0)

    return allvisible_positions, colors



def get_cam_positions():
    """
        Calculates positions of cameras with rotation and translation vectors. Swaps Y and Z axis to convert OpenCV
        3D coordinate system to OpenGL and makes the new Y negative to face the viewer.

        :return: returns position for every camera and color vector for every camera
        """
    tree = ET.parse('./data/checkerboard.xml')
    row = 0
    col = 0
    chessboard_square_size = 0
    root = tree.getroot()
    for child in root:
        if child.tag == 'CheckerBoardWidth':
            col = int(child.text)
        if child.tag == 'CheckerBoardHeight':
            row = int(child.text)
        if child.tag == 'CheckerBoardSquareSize':
            chessboard_square_size = int(child.text)


    scale = 1.0 / chessboard_square_size

    # Get all camera positions
    camera_positions = []
    for camera in range(4):
        # Get camera rotation and translation
        _, _, rvecs, tvecs = voxel_construction.load_config_info(os.path.join("data", "cam" + str(camera + 1)),
                                                    "config.xml")
        rmtx, _ = cv2.Rodrigues(rvecs)

        # Get camera position
        position = -np.matrix(rmtx).T * np.matrix(tvecs) * scale

        # Swap Y and Z axis for OpenGL system and make new Y negative to face the viewer
        camera_positions.append([position[0][0], -position[2][0], position[1][0]])

    return camera_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room

    cam_rotations = []
    for camera in range(4):
        # Get camera rotation
        _, _, rvecs, _ = voxel_construction.load_config_info(os.path.join("data", "cam" + str(camera + 1)),
                                                "config.xml")

    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
