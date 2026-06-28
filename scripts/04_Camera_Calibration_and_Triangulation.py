import cv2
import numpy as np
import math
import os
import glob
from skimage.filters import difference_of_gaussians, threshold_otsu
from itertools import product, imap


def get_frames_list(starttime, endtime):
    """
    Helper function that delivers a list of all videoframes in between two video timestamps.

    :param starttime is given as a string in mm:ss-format.
    :param endtime is given as a string in mm:ss-format.

    :return: frames_list: a list of the video frames between the the timestamps
    """
    startframe = math.floor((float(starttime[0:2])*60+float(starttime[-2:len(starttime)]))*59.94)
    endframe = math.ceil((float(endtime[0:2])*60+float(endtime[-2:len(endtime)]))*59.94)
    frames_number = endframe - startframe
    frames_list = list(range(startframe, endframe, 1))
    print('There are ' + str(frames_number) + ' frames')
    return frames_list


def multiple_frames_list(tuple_list):
    """
    Wrapper function that performs get_frames_list for multiple time periods.

    :param tuple_list: is a list containing the individual time periods in the format "('mm:ss', 'mm:ss')".

    :return: res_list: a list of frames lying between the multiple time periods.
    """
    res_list = []
    for i in range(len(tuple_list)):
        new_list = get_frames_list(str(tuple_list[i][0]), str(tuple_list[i][1]))
        res_list.extend(new_list)
    return res_list


def extract_images(video, save_folder, frames_list=None):
    """
    Helper function that extracts .jpg-images from a video stream with the option to extract only selected ones.

    :param video: Name of the video from which the images should be extracted. i.e. "DSC_0537_calibrate.mov".
    :param save_folder: folder in which the images should be saved.
    :param frames_list: Optional - Return only the images of selected frames. Speeds up the function a bit.
    """
    cam = cv2.VideoCapture(video)
    frameno = 0
    save_path = os.path.join(os.getcwd(), save_folder)

    if frames_list == None:
        while(True):
            ret, frame = cam.read()
            if ret:
                # corr_frame = frameno + 13887
                name = os.path.basename(save_path) + '_' + str(frameno) + '.png'
                cv2.imwrite(os.path.join(save_path, name), frame)
                frameno += 1
            else:
                break
        cam.release()
        cv2.destroyAllWindows()
    else:
        while (True):
            ret, frame = cam.read()
            if frameno in frames_list:
                if ret:
                    # corr_frame = frameno + 13887
                    name = os.path.basename(save_path) + '_' + str(frameno) + '.png'
                    cv2.imwrite(os.path.join(save_path, name), frame)
                else:
                    break
            else:
                pass
            frameno += 1
        cam.release()
        cv2.destroyAllWindows()


def move_selected_frames(frames_list, source_folder_name, destination_folder_name):
    """
    Helper function that moves a number of selected images given by a frame number from one folder to another.

    :param frames_list: List of the frame-numbers of the images.
    :param source_folder_name: Name of the folder the images currently lie in.
    :param destination_folder_name: Name of the folder the images should be moved to.
    """
    source_folder = os.path.join(os.getcwd(), source_folder_name)
    destination_folder = os.path.join(os.getcwd(), destination_folder_name)
    files = os.listdir(source_folder)
    prefix = os.path.basename(source_folder) + '_'

    for f in files:
        if os.path.splitext(f)[0].replace(prefix, '') == "selected":
            pass
        if os.path.splitext(f)[0].replace(prefix, '') == "jsons":
            pass
        elif int(os.path.splitext(f)[0].replace(prefix, '')) in frames_list:
            print(os.path.splitext(f)[0].replace(prefix, ''))
            os.rename(source_folder + "/" + f, destination_folder + "/" + f)


def find_corners(img_folder):
    """
    Helper function that finds the checkerboard corners in a single calibration image.

    :param img_folder: Address of the selected single calibration image.
    """
    img_path = os.path.join(os.getcwd(), img_folder)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (4,3), cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        img = cv2.drawChessboardCorners(img_path, (4,3), corners, ret)
        cv2.imshow('img', img)


def calibrate(img_folder):
    """
    Function that calibrates multiple pre-selected images with calibrateCamera of OpenCV.

    :param img_folder: Folder that contains the pre-selected images.

    :return: mtx: The calibrated camera-matrix.
    :return: dist: The calibrated distortion coefficients.
    :return: rvecs: A 3x1 rotation vector.
    :return: tvecs: A 3x1 translation vector.
    """
    img_path = os.path.join(os.getcwd(), img_folder)
    # defining the inner dimensions of the checkerboard
    CHECKERBOARD = (4, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # vector 3d
    objpoints = []
    # vector 2d
    imgpoints = []
    # all corners
    all_corners = []

    # Defining world coordinates for 3d points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Loop over images
    images = glob.glob(img_path)
    for fname in images:
        # print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # area = cv2.countNonZero(gray)

        # Adjust brightness and contrast
        alpha = 0.6  # Contrast
        beta = -50  # Brightness
        adjusted_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        img_normalized = gray / gray.max()  # normalize

        img_filtered = difference_of_gaussians(img_normalized, 0.01, 100)
        normalized_filtered = img_filtered / img_filtered.max()

        # thresh = threshold_otsu(normalized_filtered)
        # threshold_filtered = normalized_filtered > thresh

        # Zoom in by cropping
        # Define the ROI (x, y, width, height)
        # x, y, w, h = 750, 500, 150, 150  # Adjust these values based on your image
        # cropped_image = adjusted_image[y:y + h, x:x + w]

        # Optionally resize the cropped image for better visibility
        # zoomed_image = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # zoomed_image = cv2.resize(adjusted_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Apply Gaussian blur
        # blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

        ret, img1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        img2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        img3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        cv2.imshow('New', img3)
        cv2.waitKey(20)

        # print(cv2.checkChessboard(img3, CHECKERBOARD))

        # find corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # print(cv2.findChessboardCorners(adjusted_image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE))

        if ret == True:
            print(fname)
            objpoints.append(objp)
            # refining pixel coordinates for given 2d point
            corners2 = cv2.cornerSubPix(adjusted_image, corners, (4, 3), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(10)
            all_corners.append(corners2)

        else:
            # break
            points = []
            image = gray
            objpoints.append(objp)

            def select_points(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    # Draw a small circle on selected points
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                    cv2.imshow("Select Corners", image)

            cv2.imshow("Select Corners", image)
            cv2.setMouseCallback("Select Corners", select_points)
            print("Click on the checkerboard corners in order (left to right, top to bottom).")
            print("Press 's' to save and exit.")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s") and len(points) > 0:
                    break
            corners2 = np.array(points)
            corners2 = np.float32(corners2[:, np.newaxis, :])
            imgpoints.append(corners2)
            all_corners.append(corners2)

    cv2.destroyAllWindows()

    # h, w = img.shape[:2]

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # zipped = zip(mtx, dist, rvecs, tvecs)
    # np.save('Output_MAH04889_calibrate', zipped)

    mean_error = 0

    for i in range(len(objpoints)):
        imgPoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)
        mean_error += error

    print("Total Error: {}".format(mean_error/len(objpoints)))

    return mtx, dist, rvecs, tvecs

### https://www.youtube.com/watch?v=3h7wgR5fYik


def write_calibration_arrays_to_txt(img_folder, filename):
    """
    Wrapper function for the calibrate-function. Saves all params of it to single output-file.
    Therefore, unused in later stages of the project.

    :param img_folder: Folder containing the images needed to calibrate the camera.
    :param filename: Name of the output-file. Usually in the format "Output_CameraName".
    """
    img_path = os.path.join(os.getcwd(), img_folder)
    mtx, dist, rvecs, tvecs = calibrate(img_path)

    with open(filename, 'w') as f:
        f.write('Camera Matrix:\n\n')
        np.savetxt(f, mtx)
        f.write('\n\n')
        f.write('Distortion Parameters:\n\n')
        np.savetxt(f, dist)
        f.write('\n\n')
        f.write('Rotation Vectors:\n\n')
        for i in range(len(rvecs)):
            np.savetxt(f, rvecs[i])
            f.write('\n\n')
        f.write('Translation Vectors:\n\n')
        for i in range(len(tvecs)):
            np.savetxt(f, tvecs[i])
            f.write('\n\n')


def undistort_images(img_folder, mtx, dist):
    """
    Function that undistorts the images of a camera and saves the undistorted copies in the same folder.

    :param img_folder: Folder containing the images to be undistorted.
    :param mtx: Matrix of the Camera used to record the images.
    :param dist: Distortion coefficients of the same camera.

    mtx and dist are both gained from calibrate one step before.
    """
    img_path = os.path.join(os.getcwd(), img_folder)
    images = glob.glob(img_path)
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(img, mtx, None, newCameraMatrix)
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        newname = os.path.splitext(fname)[0] + '_result.jpg'
        cv2.imwrite(newname, dst)


def stereo_calibrate(mtx1, dist1, mtx2, dist2, img_folder1, img_folder2):
    """
    Given the successful calibration of both cameras on their own, this function calibrates the cameras in conjunction.

    :param mtx1: Camera Matrix of the first Camera.
    :param dist1: Distortion Coefficient of the first Camera.
    :param mtx2: Camera Matrix of the second Camera.
    :param dist2: Distortion Coefficient of the second Camera.
    :param img_folder1: Name of the folder containing the calibrated images recorded with the first camera.
    :param img_folder2: Name of the folder containing the calibrated images recorded with the second camera.

    :return: R: Output rotation matrix.
    :return: T: Output translation vector.

    For more details on the Output Parameters you can read the OpenCV documentation:
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5
    """
    img_path1 = os.path.join(os.getcwd(), img_folder1)
    img_path2 = os.path.join(os.getcwd(), img_folder2)
    # read the synched frames
    c1_images_names = glob.glob(img_path1)
    c2_images_names = glob.glob(img_path2)

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)

        _im = cv2.imread(im2, 1)
        c2_images.append(_im)

    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 4 # number of checkerboard rows.
    columns = 3 # number of checkerboard columns.
    world_scaling = 1. # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    # Adjust brightness and contrast
    alpha = 0.53  # Contrast
    beta = -82  # Brightness

    for frame1, frame2 in zip(c1_images, c2_images):
        cv2.imshow('img', frame1)
        cv2.waitKey(10)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        bw1 = cv2.convertScaleAbs(gray1, alpha=alpha, beta=beta)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        bw2 = cv2.convertScaleAbs(gray2, alpha=alpha, beta=beta)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            print("yes!")
            corners1 = cv2.cornerSubPix(bw1, corners1, (4, 3), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(bw2, corners2, (4, 3), (-1, -1), criteria)

            cv2.drawChessboardCorners(frame1, (4,3), corners1, c_ret1)
            cv2.imshow('img', frame1)

            cv2.drawChessboardCorners(frame2, (4,3), corners2, c_ret2)
            cv2.imshow('img2', frame2)
            cv2.waitKey(1000)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

        else:
            print(c_ret1, c_ret2)
            cv2.imshow('img', frame1)
            corners1 = cv2.cornerSubPix(gray1, corners1, (4, 4), (-1, -1), criteria)

            points = []
            image = gray2

            def select_points(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    # Draw a small circle on selected points
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                    cv2.imshow("Select Corners", image)

            cv2.imshow("Select Corners", image)
            cv2.setMouseCallback("Select Corners", select_points)
            print("Click on the checkerboard corners in order (left to right, top to bottom).")
            print("Press 's' to save and exit.")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s") and len(points) > 0:
                    break
            corners2 = np.array(points)
            corners2 = np.float32(corners2[:, np.newaxis, :])

            cv2.drawChessboardCorners(frame1, (4, 3), corners1, c_ret1)
            cv2.imshow('img', frame1)

            cv2.drawChessboardCorners(frame2, (4, 3), corners2, c_ret2)
            cv2.imshow('img2', frame2)
            cv2.waitKey(1000)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)



    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,
                                              dist1, mtx2, dist2, (4, 3),
                                              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001),
                                              flags=cv2.CALIB_FIX_INTRINSIC)

    print(ret)
    return R, T


if __name__ == "__main__":
    # 1.) Get Calibration Frames and Shooting Frames
    # TODO: Define the CALIBRATION_TIMESPAN and LIST_OF_SHOOTING_TIMEFRAMES based on video recordings, and the necessary paths.
    CALIBRATION_TIMESPAN = ('XX:XX', 'XX:XX')
    LIST_OF_SHOOTING_TIMEFRAMES = [('XX:XX', 'XX:XX')]
    FIRST_CALIBRATION_VIDEO_NAME = ''
    SECOND_CALIBRATION_VIDEO_NAME = ''
    FIRST_VIDEO_NAME = ''
    SECOND_VIDEO_NAME = ''
    FIRST_VIDEO_CALIBRATION_PATH = ''
    SECOND_VIDEO_CALIBRATION_PATH = ''
    FIRST_VIDEO_SHOTS_IMAGES_PATH = ''
    SECOND_VIDEO_SHOTS_IMAGES_PATH = ''

    calibration_frames_list = get_frames_list(CALIBRATION_TIMESPAN)
    shooting_frames_list = multiple_frames_list(LIST_OF_SHOOTING_TIMEFRAMES)

    extract_images(FIRST_CALIBRATION_VIDEO_NAME, FIRST_VIDEO_CALIBRATION_PATH, calibration_frames_list)
    extract_images(SECOND_CALIBRATION_VIDEO_NAME, SECOND_VIDEO_CALIBRATION_PATH, calibration_frames_list)
    extract_images(FIRST_VIDEO_NAME, FIRST_VIDEO_SHOTS_IMAGES_PATH, shooting_frames_list)
    extract_images(SECOND_VIDEO_NAME, SECOND_VIDEO_SHOTS_IMAGES_PATH, shooting_frames_list)
    ####################################################################################################################

    # 2.) Handpick the correct frames for calibration and defined moments of the shooting motion
    # (support leg placement and strike)
    #TODO: Write handpicked frame names into lists

    CALIBRATION_FRAMES = []
    SHOT_FRAMES = []

    FIRST_VIDEO_CALIBRATION_SELECTED_PATH = FIRST_VIDEO_CALIBRATION_PATH + '/selected'
    SECOND_VIDEO_CALIBRATION_SELECTED_PATH = SECOND_VIDEO_CALIBRATION_PATH + '/selected'
    FIRST_VIDEO_SHOTS_IMAGES_SELECTED_PATH = FIRST_VIDEO_SHOTS_IMAGES_PATH + '/selected'
    SECOND_VIDEO_SHOTS_IMAGES_SELECTED_PATH = SECOND_VIDEO_SHOTS_IMAGES_PATH + '/selected'

    move_selected_frames(CALIBRATION_FRAMES, FIRST_VIDEO_CALIBRATION_PATH, FIRST_VIDEO_CALIBRATION_SELECTED_PATH)
    move_selected_frames(CALIBRATION_FRAMES, SECOND_VIDEO_CALIBRATION_PATH, SECOND_VIDEO_CALIBRATION_SELECTED_PATH)
    move_selected_frames(SHOT_FRAMES, FIRST_VIDEO_SHOTS_IMAGES_PATH, FIRST_VIDEO_SHOTS_IMAGES_SELECTED_PATH)
    move_selected_frames(SHOT_FRAMES, SECOND_VIDEO_SHOTS_IMAGES_PATH, SECOND_VIDEO_SHOTS_IMAGES_SELECTED_PATH)
    ####################################################################################################################

    # 3.) Calibrate the cameras with the selected calibration frames

    mtx1, dist1, rvecs1, tvecs1 = calibrate(FIRST_VIDEO_CALIBRATION_SELECTED_PATH + '/*.jpg')
    np.save('mtx_DSC_0644_calibrate.npy', mtx1, allow_pickle=True)
    mtx2, dist2, rvecs2, tvecs2 = calibrate(SECOND_VIDEO_CALIBRATION_SELECTED_PATH + '/*.jpg')
    np.save('mtx_MAH05312_calibrate.npy', mtx2, allow_pickle=True)
    ####################################################################################################################

    # 4.) Undistort calibration frames and move them to results folder
    undistort_images(FIRST_VIDEO_CALIBRATION_SELECTED_PATH + '/*.jpg', mtx1, dist1)
    undistort_images(SECOND_VIDEO_CALIBRATION_SELECTED_PATH + '/*.jpg', mtx2, dist2)

    FIRST_VIDEO_CALIBRATION_RESULT_PATH = FIRST_VIDEO_CALIBRATION_PATH + '/results'
    SECOND_VIDEO_CALIBRATION_RESULT_PATH = SECOND_VIDEO_CALIBRATION_PATH + '/results'

    move_selected_frames(list(imap('_'.join, product(SHOT_FRAMES, '_result'))), FIRST_VIDEO_CALIBRATION_PATH, FIRST_VIDEO_CALIBRATION_RESULT_PATH)
    move_selected_frames(list(imap('_'.join, product(SHOT_FRAMES, '_result'))), SECOND_VIDEO_CALIBRATION_PATH, SECOND_VIDEO_CALIBRATION_RESULT_PATH)
    ####################################################################################################################

    # 5.) Stereo Calibration for Triangulation in later stages of the pipeline
    R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'DSC_0644_calibrate/results/*.jpg', 'MAH05312_calibrate/results/*.jpg')
    np.save('R_Measure1.npy', R, allow_pickle=True)
    np.save('T_Measure1.npy', T, allow_pickle=True)
