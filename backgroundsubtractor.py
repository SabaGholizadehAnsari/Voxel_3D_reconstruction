# This is a sample Python script.
import numpy as np
import sys
path_to_module = "C:/Users/Gholi002/AppData/Local/r-miniconda/envs/myenvironment/Lib/site-packages"
sys.path.append(path_to_module)
import cv2
import os
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def find_background(video_path, file_name, history, threshold, detect_shadow):
    cap = cv2.VideoCapture(os.path.join(video_path, file_name))
    back_model = cv2.createBackgroundSubtractorMOG2(history, threshold, detect_shadow)
    while (1):
        ret, img = cap.read()
        if ret:
            img = img.astype(float).astype('uint8')
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            back_model.apply(hsv_img, learningRate=-1)
        if not ret:
            break

        cap.release();
    cv2.destroyAllWindows();
    return back_model


def substract_background(video_path, file_name, bg_model, num_frames, opening, closing):
    cap = cv2.VideoCapture(os.path.join(video_path, file_name))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices to extract
    frame_indices = [int(i * total_frames / (num_frames - 1)) for i in range(num_frames)]
    flag = True
    # Extract frames
    frame_count = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if frame_count in frame_indices:

            img = img.astype(float).astype('uint8')
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bg_mask = bg_model.apply(hsv_img, learningRate=-1)
            frame_count += 1
            # Morphological closing: first dilation,then erosion => useful for filling small holes (make it white) in an image
            # while preserving the shape and size of large holes and objects in the image
            if closing:
                kernel = np.ones((3, 3), np.uint8)

                # closing the image
                bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE,
                                           kernel, iterations=1)
            # Morphological opening:  first erosion, then dilation => useful for removing small objects and thin lines from an image
            # while preserving the shape and size of larger objects in the image.
            if opening:
                # define the kernel
                kernel = np.ones((3, 3), np.uint8)
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

                # opening the image
                bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN,
                                           kernel, iterations=1)

        cap.release()
    return bg_mask


def calculate_average_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    alpha = 0.0001  # You can adjust this value to control the weight of each frame

    ret, frame = cap.read()
    average_frame = frame.astype(float)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.accumulateWeighted(frame, average_frame, alpha)

    average_frame = average_frame.astype('uint8')
    cap.release()
    return average_frame


def contouring(bg_mask, outer_threshold, inner_threshold):
    contours, hierarchy = cv2.findContours(bg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    foreground = np.zeros(bg_mask.shape, dtype=np.uint8)

    # Fill large contours
    for idx, contour in enumerate(contours):
        # Contour accepted if its area is larger than the threshold
        if cv2.contourArea(contour) >= outer_threshold:
            # Draw contour and fill area
            cv2.drawContours(foreground, [contour], -1, 255)  # find and draw contours
            cv2.fillPoly(foreground, [contour], 255)  # fill the counter with white 255
            # For every thresholded figure we need to fill back black areas inside it
            # Look at the first child (if it exists) as first inner contour
            inner_idx = hierarchy[0][idx][2]
            while inner_idx != -1:
                # Inner contour accepted if its area is larger than figure inner threshold
                if cv2.contourArea(contours[inner_idx], True) >= inner_threshold:
                    # Draw inner contour and fill area
                    cv2.fillPoly(foreground, [contours[inner_idx]], 0)
                    cv2.drawContours(foreground, [contours[inner_idx]], -1, 255)
                # Next inner contour at the same hierarchy level as this inner contour
                inner_idx = hierarchy[0][inner_idx][0]
    return foreground

def main():
    data_path = "data"
    plots_path = "plots"
    cam1_path = os.path.join(data_path, "cam1")
    cam2_path = os.path.join(data_path, "cam2")
    cam3_path = os.path.join(data_path, "cam3")
    cam4_path = os.path.join(data_path, "cam4")
    cam_paths = [cam1_path, cam2_path, cam3_path, cam4_path]

    for cam in range(0, len(cam_paths)):
        # MOG2 history=500, var_threshold=16, detect_shadows=True, learning_rate=-1
        background_model = find_background(cam_paths[cam], "background.avi", history=500, threshold=300,
                                           detect_shadow=False)
        foreground = substract_background(cam_paths[cam], "video.avi", background_model, 10, True,
                                          False)  # closing make it worse nd opening makee it better less white noises.
        cv2.imwrite(os.path.join(cam_paths[cam], f"bgmask_opening_{cam}.jpg"), foreground)
        contoured_foreground = contouring(foreground, outer_threshold=1000, inner_threshold=130)
        cv2.imwrite(os.path.join(cam_paths[cam], f"bgmask_opening_contouring_{cam}.jpg"), contoured_foreground)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
