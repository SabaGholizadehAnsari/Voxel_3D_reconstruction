import numpy as np
import sys

import cv2
import os
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import utils


def load_config_info(config_info_path="data/cam", config_input_filename="config.xml"):
    """
    Loads intrinsic (camera matrix, distortion coefficients) and extrinsic (rotation vector, translation vector) camera
    parameters from config file.

    :param config_info_path: config xml file directory path
    :param config_input_filename: config xml file name
    :return: camera matrix
    """
    # Select tags for loaded nodes and their types
    node_tags = ["CameraMatrix", "DistortionCoeffs", "Rotation", "Translation"]
    node_types = ["mat" for _ in range(len(node_tags))]

    # Load nodes
    nodes = utils.load_xml_nodes(config_info_path, config_input_filename, node_tags, node_types)

    # Parse config
    mtx = nodes.get("CameraMatrix")
    dist = nodes.get("DistortionCoeffs")
    rvecs = nodes.get("Rotation")
    tvecs = nodes.get("Translation")

    return mtx, dist, rvecs, tvecs


def create_voxel_volume(num_voxels_x=128, num_voxels_y=128, num_voxels_z=128, x_min=-512, x_max=1024, y_min=-1024,
                        y_max=1024, z_min=-2048, z_max=512):
    """
    Creates voxel volume points given dimensions and linear spaces.

    :param num_voxels_x: number of voxels in x range
    :param num_voxels_y: number of voxels in y range
    :param num_voxels_z: number of voxels in z range
    :param x_min: min x for sampling x range
    :param x_max: max x for sampling x range
    :param y_min: min y for sampling y range
    :param y_max: max y for sampling y range
    :param z_min: min z for sampling z range
    :param z_max: max z for sampling z range
    :return: voxel volume points
    """
    # Sample ranges
    x_range = np.linspace(x_min, x_max, num=num_voxels_x)
    y_range = np.linspace(y_min, y_max, num=num_voxels_y)
    z_range = np.linspace(z_min, z_max, num=num_voxels_z)

    # Generate points
    voxel_points = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
    return voxel_points


def create_lookup_table(voxel_points, num_cameras, cam_input_path="data", config_input_filename="config.xml"):
    lookup_table = {camera: [] for camera in range(1, num_cameras + 1)}
    for cam in range(1, num_cameras + 1):

        mtx, dist, rvecs, tvecs = load_config_info(os.path.join("data", "cam" + str(cam)), "config.xml")
        image_points, _ = cv2.projectPoints(voxel_points, rvecs, tvecs, mtx, dist)


        for voxel_p, img_p in zip(voxel_points, image_points):
            x, y = img_p[0]
            lookup_table[cam].append((tuple(map(int, voxel_p)), (x, y)))

    return lookup_table


def visible_voxels_coloring(lookup_table, cam_foregrounds, cam_allframes):
    voxel_visible = {}
    voxels_visible_colors = {}
    for cam, list in lookup_table.items():
        cam_index = cam - 1
        for voxel, (x, y) in list:

            # if x and y is within foregrounds dimension
            if 0 <= y < cam_foregrounds[cam_index].shape[0] and 0 <= x < cam_foregrounds[cam_index].shape[1]:

                if cam_foregrounds[cam_index][int(y), int(x)] > 0:
                    if voxel not in voxel_visible:
                        voxel_visible[voxel] = {}
                    voxel_visible[voxel][cam] = True
                    color = cam_allframes[cam_index][int(y), int(x), :]
                    if voxel not in voxels_visible_colors:
                        voxels_visible_colors[voxel] = {}
                    voxels_visible_colors[voxel][cam] = np.array(color)

    return voxel_visible, voxels_visible_colors
