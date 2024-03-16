import numpy as np
import sys
import cv2
import os
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def extract_frames(video_path, output_dir, num_frames):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices to extract
    frame_indices = [int(i * total_frames / (num_frames - 1)) for i in range(num_frames)]

    # Extract frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in frame_indices:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_indices.remove(frame_count)  # Remove extracted frame index
            if len(frame_indices) == 0:
                break
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"{num_frames} frames extracted from {video_path} to {output_dir}")


def camera_intrinsic_calibration(file_path):
    images = glob.glob(file_path)
    # images=glob.glob('./Cam1extracted_frames_extrinsic/*.jpg')

    tree = ET.parse('./data/checkerboard.xml')
    row = 0
    col = 0
    squar_size = 0
    root = tree.getroot()
    for child in root:
        if child.tag == 'CheckerBoardWidth':
            col = int(child.text)
        if child.tag == 'CheckerBoardHeight':
            row = int(child.text)
        if child.tag == 'CheckerBoardSquareSize':
            squar_size = int(child.text)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    board_size = (col, row)
    objpoints = []
    imagepoints = []
    objp = np.zeros((col * row, 3), np.float32)
    objp[:, :2] = squar_size * np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    count = 0

    for fname in images:

        img = cv2.imread(fname)
        # img = cv2.resize(img, (500,500))
        points = []
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corner = cv2.findChessboardCorners(gray_img, board_size,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            corner2 = cv2.cornerSubPix(gray_img, corner, (11, 11), (-1, -1), criteria)
            # corners2 = np.array([[corner for [corner] in corner2]])

            imagepoints.append(corner2)

            img = cv2.drawChessboardCorners(img, board_size, corner, ret)
        else:
            os.remove(fname)
            # objpoints.append(objp)
            # cv2.imshow('image', img)
            # cv2.setMouseCallback('image', click_event, param=points)
            # Wait for four mouse clicks
            # while len(points) < 4:
            #   cv2.waitKey(10)

            # corners=calculate_inner_corners(np.array(points), rows=row-1, cols=col-1,squar_size=squar_size)
            # cv2.waitKey(10)
            # corner2 = cv2.cornerSubPix(gray_img, corners, (11,11),(-1,-1), criteria)
            # imagepoints.append(corners)
            # img=cv2.drawChessboardCorners(img,board_size,corners,True)
        cv2.imshow('image', img)
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imagepoints, gray_img.shape[::-1], None, None)
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    np.save("camera matrix for cam4", mtx)
    # np.save("rotation matric for cam1", rvecs)
    # np.save("translation vector for cam1", tvecs)
    np.save("dist for cam4", dist)

    # np.save("Distortion coefficients for all img", coef)
    return mtx, dist


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        param.append((x,y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
        str(y), (x,y), font,
        1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    if event==cv2.EVENT_RBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
        str(g) + ',' + str(r),
        (x,y), font, 1,
        (255, 255, 0), 2)
        param.append((x,y))
        cv2.imshow('image', img)


def calculate_inner_corners2(corners, rows, cols, sort_corners=False):
    # Sort corners to (top-left, top-right, bottom-right, bottom-left)

    if sort_corners:
        corners = sort_corners_clockwise(corners, origin="top-left")
    else:
        corners = np.array(corners, dtype="float32")

    # Calculate the maximum width and height
    max_width = max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[3] - corners[2]))
    max_height = max(np.linalg.norm(corners[2] - corners[1]), np.linalg.norm(corners[3] - corners[0]))

    # Use maximum width and height to form destination coordinates for perspective transform
    dest_corners = np.float32([[0, 0], [max_width - 1, 0],
                               [max_width - 1, max_height - 1], [0, max_height - 1]])

    # Adjustment for cases where the corners given are outer corners
    horizontal_adjust = 0
    vertical_adjust = 0

    # Horizontal and vertical step calculation using chessboard shape
    horizontal_step = max_width / (cols - 1)
    vertical_step = max_height / (rows - 1)

    interpolated_row = []
    interpolated_points = []
    # Perform perspective transform for accuracy improvement
    p_matrix = cv2.getPerspectiveTransform(corners, dest_corners)

    # Get inverse matrix for projecting points from the transformed space back to the original image space
    inverted_p_matrix = np.linalg.inv(p_matrix)

    # Compute each projected point
    for y in range(0, rows):
        for x in range(0, cols):
            # Calculate the position of the current point relative to the grid using homogenous coordinates
            point = np.array([horizontal_adjust + x * horizontal_step,
                              vertical_adjust + y * vertical_step,
                              1])

            # Multiply with inverse matrix to project point from transformed space back to original image space
            point = np.matmul(inverted_p_matrix, point)

            # Divide point by its Z
            point /= point[2]

            # Append the X and Y of point to the list of interpolated points in row
            interpolated_row.append(point[:2])
        # Append interpolated points in row to interpolated points
        interpolated_points.append(interpolated_row)
        interpolated_row = []

    # If change_point_order is True then point order will start from bottom-left and end on top-right
    # moving through rows before changing column
    # if False then point order will start at top-left and end on bottom-right
    # moving through columns before changing row as already saved
    interpolated_points = np.array(interpolated_points, dtype="float32")

    # Return (MxN, 1, 2) array to match automatic corner detection output
    return np.reshape(interpolated_points, (-1, 1, 2))

def sort_corners_clockwise(corners, origin="top-left"):
    """
    Sorts given corner coordinates in clockwise order.

    :param corners: array of corner points ([x, y])
    :param origin: which corner point starts the clockwise order (bottom-left, top-left, top-right, or bottom-right)
    :return: returns array of sorted corners
    """
    # Calculate the centroid of the corners
    centroid = np.mean(corners, axis=0)

    # Sort corners and determine their relative position to the centroid
    top = sorted([corner for corner in corners if corner[1] < centroid[1]], key=lambda point: point[0])
    bottom = sorted([corner for corner in corners if corner[1] >= centroid[1]], key=lambda point: point[0],
                    reverse=True)

    # Sort top and bottom corners depending on first element
    if origin == "top-left":
        return np.array(top + bottom, dtype="float32")
    elif origin == "top-right":
        return np.array([top[1]] + bottom + [top[0]], dtype="float32")
    elif origin == "bottom-right":
        return np.array(bottom + top, dtype="float32")
    else:
        return np.array([bottom[1]] + top + [bottom[0]], dtype="float32")


def camera_extrinsic_calibration(file_name, camera_matrix, distortion_coeffs):
    global img
    images = glob.glob(file_name)

    tree = ET.parse('./data/checkerboard.xml')
    row = 0
    col = 0
    squar_size = 0
    root = tree.getroot()
    for child in root:
        if child.tag == 'CheckerBoardWidth':
            col = int(child.text)
        if child.tag == 'CheckerBoardHeight':
            row = int(child.text)
        if child.tag == 'CheckerBoardSquareSize':
            squar_size = int(child.text)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    board_size = (col, row)
    objpoints = []
    imagepoints = []
    objp = np.zeros((col * row, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * squar_size

    count = 0

    for fname in images:

        img = cv2.imread(fname)
        points = []
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        count += 1
        objpoints.append(objp)
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event, param=points)
        while len(points) < 4:
            cv2.waitKey(10)

        # corners=calculate_inner_corners(np.array(points), rows=row, cols=col,squar_size=squar_size)

        corners = calculate_inner_corners2(np.array(points), rows=row, cols=col, sort_corners=False)
        cv2.waitKey(10)
        # corner2 = cv2.cornerSubPix(gray_img, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imagepoints.append(corners)
        img = cv2.drawChessboardCorners(img, board_size, corners, True)
        cv2.imshow('image', img)
        cv2.waitKey(100)
    cv2.destroyAllWindows()

    objpoints = np.array(objpoints).astype('float32')
    imagepoints = np.array(imagepoints).astype('float32')  # .reshape(count,48,2)
    objpoints = objpoints.reshape(-1, 3)  # Reshape objpoints to (N, 3)
    imagepoints = imagepoints.reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(objpoints, imagepoints, camera_matrix, distortion_coeffs)
    return rvec, tvec, imagepoints


def main():
    data_path = "data"
    plots_path = "plots"
    cam1_path = os.path.join(data_path, "cam1")
    cam2_path = os.path.join(data_path, "cam2")
    cam3_path = os.path.join(data_path, "cam3")
    cam4_path = os.path.join(data_path, "cam4")
    cam_paths = [cam1_path, cam2_path, cam3_path, cam4_path]
    intrinsics = False
    extrinsics = True
    for cam in range(0, len(cam_paths)):
        if intrinsics:
            dir_str = "Cam" + str(cam + 1) + "extracted_frames_intrinsic"
            output_dir = dir_str
            num_frames = 50  # Number of frames to extract
            extract_frames(os.path.join(cam_paths[cam], 'intrinsics.avi'), output_dir, num_frames)
            mtx, dist = camera_intrinsic_calibration(os.path.join(dir_str, '*.jpg'))

            tree = ET.parse(os.path.join(cam_paths[cam], 'intrinsics.xml'))
            root = tree.getroot()
            camera_matrix = root.find(".//CameraMatrix/data")
            camera_matrix.text = '\n'.join([' '.join(map(str, row)) for row in mtx])

            distortion_coeffs = root.find(".//DistortionCoeffs/data")
            distortion_coeffs.text = '\n'.join([' '.join(map(str, row)) for row in dist])

            # Write the updated XML tree to file
            tree.write(os.path.join(cam_paths[cam], 'config.xml'))
        if extrinsics:
            print("......extrinsics.....")
            # load from XML
            tree = ET.parse(os.path.join(cam_paths[cam], 'config.xml'))
            root = tree.getroot()
            camera_matrix_elem = root.find(".//CameraMatrix")

            rows = int(camera_matrix_elem.find("rows").text)
            cols = int(camera_matrix_elem.find("cols").text)
            data_text = camera_matrix_elem.find("data").text

            camera_matrix = np.fromstring(data_text, dtype=float, sep=' ').reshape(rows, cols)
            camera_matrix = np.float32(camera_matrix).reshape((3, 3))

            distortion_coeffs_elem = root.find(".//DistortionCoeffs")
            rows = int(distortion_coeffs_elem.find("rows").text)
            cols = int(distortion_coeffs_elem.find("cols").text)
            data_text = distortion_coeffs_elem.find("data").text
            distortion_coeffs = np.fromstring(data_text, dtype=float, sep=' ').reshape(rows, cols)
            distortion_coeffs = np.float32(distortion_coeffs).reshape((1, 5))
            dir_str = "Cam" + str(cam + 1) + "extracted_frames_extrinsic"
            output_dir = dir_str
            extract_frames(os.path.join(cam_paths[cam], 'checkerboard.avi'), output_dir, 5)
            rvec, tvec, image_points = camera_extrinsic_calibration(os.path.join(dir_str, '*.jpg'), camera_matrix,
                                                                    distortion_coeffs)
            # save calibration
            file_handle = cv2.FileStorage(os.path.join(cam_paths[cam], 'config.xml'), cv2.FileStorage_WRITE)
            file_handle.write('CameraMatrix', camera_matrix)
            file_handle.write('DistortionCoeffs', distortion_coeffs)
            file_handle.write('Rotation', rvec)
            file_handle.write('Translation', tvec)
            file_handle.release()
            print(f"rvec:", rvec)
            print(f"tvec:", tvec)
            # tree = ET.parse()
            if cam == 0:
                img = cv2.imread('./Cam1extracted_frames_extrinsic/frame_0.jpg')
                axis_length = 400  # Length of the axes
                axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]])
                img_points, jac = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, distortion_coeffs)
                img_points = img_points.astype(int).reshape(-1, 2)
                origin_corner = image_points[0].astype(np.int32)
                cv2.arrowedLine(img, origin_corner, tuple(img_points[0].ravel()), color=(0, 0, 255), thickness=2)
                # Draw green arrow for vertical axis
                cv2.arrowedLine(img, origin_corner, tuple(img_points[1].ravel()), color=(0, 255, 0), thickness=2)
                # Draw blue arrow for Z axis
                cv2.arrowedLine(img, origin_corner, tuple(img_points[2].ravel()), color=(255, 0, 0), thickness=2)

                plt.figure(figsize=(5, 5))
                # cv2.drawFrameAxes(img, camera_matrix, distortion_coeffs,np.array(rvec),np.array(tvec), axis_length)
                cv2.imshow('image', img)
                cv2.waitKey(10000)
                # draw_axes_on_chessboard(img, image_points, camera_matrix, distortion_coeffs, np.array(rvec),np.array(tvec), chessboard_square_size=151,
                # chessboard_square_span=1)
            if cam == 1:
                img = cv2.imread('./Cam2extracted_frames_extrinsic/frame_0.jpg')
                axis_length = 400  # Length of the axes
                axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]])
                img_points, jac = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, distortion_coeffs)
                img_points = img_points.astype(int).reshape(-1, 2)
                origin_corner = image_points[0].astype(np.int32)
                cv2.arrowedLine(img, origin_corner, tuple(img_points[0].ravel()), color=(0, 0, 255), thickness=2)
                # Draw green arrow for vertical axis
                cv2.arrowedLine(img, origin_corner, tuple(img_points[1].ravel()), color=(0, 255, 0), thickness=2)
                # Draw blue arrow for Z axis
                cv2.arrowedLine(img, origin_corner, tuple(img_points[2].ravel()), color=(255, 0, 0), thickness=2)

                plt.figure(figsize=(5, 5))
                # cv2.drawFrameAxes(img, camera_matrix, distortion_coeffs,np.array(rvec),np.array(tvec), axis_length)
                cv2.imshow('image', img)
                cv2.waitKey(10000)
            if cam == 2:
                img = cv2.imread('./Cam3extracted_frames_extrinsic/frame_0.jpg')
                axis_length = 400  # Length of the axes
                axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]])
                img_points, jac = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, distortion_coeffs)
                img_points = img_points.astype(int).reshape(-1, 2)
                origin_corner = image_points[0].astype(np.int32)
                cv2.arrowedLine(img, origin_corner, tuple(img_points[0].ravel()), color=(0, 0, 255), thickness=2)
                # Draw green arrow for vertical axis
                cv2.arrowedLine(img, origin_corner, tuple(img_points[1].ravel()), color=(0, 255, 0), thickness=2)
                # Draw blue arrow for Z axis
                cv2.arrowedLine(img, origin_corner, tuple(img_points[2].ravel()), color=(255, 0, 0), thickness=2)

                plt.figure(figsize=(5, 5))
                # cv2.drawFrameAxes(img, camera_matrix, distortion_coeffs,np.array(rvec),np.array(tvec), axis_length)
                cv2.imshow('image', img)
                cv2.waitKey(10000)
            if cam == 3:
                img = cv2.imread('./Cam4extracted_frames_extrinsic/frame_0.jpg')
                axis_length = 400  # Length of the axes
                axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]])
                img_points, jac = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, distortion_coeffs)
                img_points = img_points.astype(int).reshape(-1, 2)
                origin_corner = image_points[0].astype(np.int32)
                cv2.arrowedLine(img, origin_corner, tuple(img_points[0].ravel()), color=(0, 0, 255), thickness=2)
                # Draw green arrow for vertical axis
                cv2.arrowedLine(img, origin_corner, tuple(img_points[1].ravel()), color=(0, 255, 0), thickness=2)
                # Draw blue arrow for Z axis
                cv2.arrowedLine(img, origin_corner, tuple(img_points[2].ravel()), color=(255, 0, 0), thickness=2)

                plt.figure(figsize=(5, 5))
                # cv2.drawFrameAxes(img, camera_matrix, distortion_coeffs,np.array(rvec),np.array(tvec), axis_length)
                cv2.imshow('image', img)
                cv2.waitKey(10000)
            # root = tree.getroot()
    cv2.destroyAllWindows
            # tvecs = root.find(".//tvecs/data")
            # tvecs.text = '\n'.join([' '.join(map(str, row)) for row in tvec])
            # rvecs = root.find(".//rvecs/data")
            # rvecs.text = '\n'.join([' '.join(map(str, row)) for row in rvec])
            # Write the updated XML tree to file
            # tree.write(os.path.join(cam_paths[cam],'config.xml'))

    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()