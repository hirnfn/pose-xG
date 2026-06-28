import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import os


def read_json(file, person_index):
    """
    Helper function that reads a .json-file that was the output of OpenPose.

    :param file: .json-file from OpenPose that contains the point coordinates of the estimation.
    :param person_index: Depending on how many people are visible in the picture, OpenPose does multiple estimations.
                         Unfortunately, there is no option to automatically detect the right index.
                         I used the function 'draw_2D' further below to establish which estimation is the one in question.

    :return: lst: a list containing all the coordinates of the estimated points.
    """
    with open(file) as json_file:
        json_data = json.load(json_file)
        # x = np.reshape(json_data['people'][person_index]['pose_keypoints_2d'], (25, 3)) #somehow changes order
        # y = np.delete(x, 2, 0).T
        pose_data = json_data['people'][person_index]['pose_keypoints_2d']
        ran = int(len(pose_data)/3)
        lst = []
        for i in range(ran):
            z = [pose_data[i*3], pose_data[i*3+1]]
            lst.append(z)
        return lst


def write_3D_points_to_json(filename, data):
    """
    Helper function that saves the triangulated 3D points in a .json-file.

    :param filename: Name of the file in which the points should be saved.
    :param data: The coordinates of the triangulated 3D points
    """
    dic = {}
    for i in range(data.shape[0]):
        dic[i] = list(data[i])
    file = filename + '.json'
    with open(file, 'w') as f:
        json.dump(dic, f)


def triangulate(mtx1, mtx2, R, T,  picture1, picture2, check1=None, check2=None):
    """
    Function that triangulates the 2D points of the images from to camera angles into 3D points.

    :param mtx1: Camera Matrix of the first Camera.
    :param mtx2: Camera Matrix of the second Camera.
    :param R: Output rotation matrix.
    :param T: Output translation vector.
    :param picture1: Picture taken from the first camera angle.
    :param picture2: Picture taken from the second camera angle.
    :param check1: Estimated coordinates for all the body keypoints from picture1.
    :param check2: Estimated coordinates for all the body keypoints from picture2.

    :return: p3ds: Triangulated 3D points.
    """
    if check1 is None:
        uvs1 = [[458, 86], [451, 164], [287, 181],
                [196, 383], [297, 444], [564, 194],
                [562, 375], [596, 520], [329, 620],
                [488, 622], [432, 52], [489, 56]]
        uvs2 = [[540, 311], [603, 359], [542, 378],
                [525, 507], [485, 542], [691, 352],
                [752, 488], [711, 605], [549, 651],
                [651, 663], [526, 293], [542, 290]]
    else:
        uvs1 = check1
        uvs2 = check2

    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)

    frame1 = cv2.imread(picture1)
    frame2 = cv2.imread(picture2)

    # plt.imshow(frame1[:, :, [2, 1, 0]])
    # plt.scatter(uvs1[:, 0], uvs1[:, 1])
    # plt.show()  # this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this.

    # plt.imshow(frame2[:, :, [2, 1, 0]])
    # plt.scatter(uvs2[:, 0], uvs2[:, 1])
    # plt.show()  # this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this

    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2

    def DLT(P1, P2, point1, point2):

        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        print('Triangulated point: ')
        print(Vh[3, 0:3] / Vh[3, 3])
        return Vh[3, 0:3] / Vh[3, 3]

    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)

    from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim3d(-15, 5)
    # ax.set_ylim3d(-10, 10)
    # ax.set_zlim3d(10, 30)

    # connections = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [1, 9], [2, 8], [5, 9], [8, 9], [0, 10], [0, 11]]

    '''
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head to ears
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    '''
    #OpenPose
    connections = [
        [0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10],
        [10, 11], [11, 22], [11, 24], [12, 13], [13, 14], [14, 19], [14, 21], [15, 17], [16, 18], [19, 20], [22, 23]
    ]

    for _c in connections:
        print(p3ds[_c[0]])
        print(p3ds[_c[1]])
        # ax.plot(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]], ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]], zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]], c='red')

    # Create a 3D scatter plot for each point
    # ax.scatter(xs=p3ds[:, 0], ys=p3ds[:, 1], zs=p3ds[:, 2], c='blue', marker='o')

    # ax.set_title('This figure can be rotated.')
    # uncomment to see the triangulated pose. This may cause a crash if youre also using cv.imshow() above.
    # plt.show()
    return p3ds


def triangulate_all_images(img_folder1, img_folder2, json_path_name1, json_path_name2, mtx1, mtx2, R, T, shots_list):
    """
    WARNING: This function only works on images where only one person is visible. Ergo the trial study.
             For images containing multiple people, i used triangulate one by one.

    Wrapper function that performs 'triangulate' on multiple images in a loop.

    :param img_folder1: Path of the folder containing the pictures taken from the first camera angle.
    :param img_folder2: Path of the folder containing the pictures taken from the second camera angle.
    :param json_path_name1: Path of the folder containing the .json-files corresponding to the images in img_folder1.
    :param json_path_name2: Path of the folder containing the .json-files corresponding to the images in img_folder2.
    :param mtx1: Camera Matrix of the first Camera.
    :param mtx2: Camera Matrix of the second Camera.
    :param R: Output rotation matrix.
    :param T: Output translation vector.
    :param shots_list: List of the image frames to be triangulated.
    """
    img_path1 = os.path.join(os.getcwd(), img_folder1)
    img_path2 = os.path.join(os.getcwd(), img_folder2)
    json_path1 = os.path.join(os.getcwd(), json_path_name1)
    json_path2 = os.path.join(os.getcwd(), json_path_name2)
    for i in range(len(shots_list)):
        picture1 = img_path1 + str(shots_list[i]) + '.jpg'
        picture2 = img_path2 + str(shots_list[i]) + '.jpg'
        check1_path = json_path1 + str(shots_list[i]) + '_keypoints.json'
        check1 = read_json(check1_path, 0)
        check2_path = json_path2 + str(shots_list[i]) + '_keypoints.json'
        check2 = read_json(check2_path, 0)
        json_path = os.path.join(os.getcwd(), 'Triangulated_Points/3D_Points_' + str(shots_list[i]))
        pts = triangulate(mtx1, mtx2, R, T, picture1, picture2, check1, check2)
        write_3D_points_to_json(json_path, pts)
        plot_human_pose_3d(pts) # for check


def plot_human_pose_3d(keypoints):
    """
    Helper function that plots the 3D points gained from 'triangulate'.

    :param keypoints: 3D points returned from 'triangulate'
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    z = keypoints[:, 2]

    '''
     # Connections between keypoints in COCO format
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head to ears
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    '''

    # OpenPose
    connections = [
        [0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10],
        [10, 11], [11, 22], [11, 24], [12, 13], [13, 14], [14, 19], [14, 21], [15, 17], [16, 18], [19, 20], [22, 23]
    ]

    # Plot keypoints
    ax.scatter(x, y, z, c='red', marker='o')

    # Plot connections
    for i, j in connections:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Find the overall range across all axes
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()

    # Determine the maximum range and set limits based on that
    max_range = max(x_range, y_range, z_range)
    center = [(x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2]

    ax.set_xlim([center[0] - max_range, center[0] + max_range])
    ax.set_ylim([center[1] - max_range, center[1] + max_range])
    ax.set_zlim([center[2] - max_range, center[2] + max_range])

    plt.show()


def draw_2D(keypoints):
    """
    Helper function that plots the 2D points gained from OpenPose.

    :param keypoints: 2D points contained in the .json-file
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.yaxis.set_inverted(True)

    x = []
    y = []
    for i in range(len(keypoints)):
        x.append(keypoints[i][0])
        y.append(keypoints[i][1])

    connections = [
        [0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10],
        [10, 11], [11, 22], [11, 24], [12, 13], [13, 14], [14, 19], [14, 21], [15, 17], [16, 18], [19, 20], [22, 23]
    ]

    # Plot keypoints
    ax.scatter(x, y, c='red', marker='o')

    # Plot connections
    for i, j in connections:
        ax.plot([x[i], x[j]], [y[i], y[j]], 'blue')

    plt.show()


if __name__ == "__main__":
    # Run this script for each shot in the SHOT_FRAMES list from 04_Camera_Calibration_and_triangulation.
    SHOT_FRAME = ''
    FIRST_VIDEO_SHOTS_IMAGES_SELECTED_PATH = ''
    SECOND_VIDEO_SHOTS_IMAGES_SELECTED_PATH = ''
    FIRST_VIDEO_SHOTS_IMAGES_JSON_PATH = ''
    SECOND_VIDEO_SHOTS_IMAGES_JSON_PATH = ''

    picture1 = os.path.join(os.getcwd(), FIRST_VIDEO_SHOTS_IMAGES_SELECTED_PATH + SHOT_FRAME + '.jpg')
    picture2 = os.path.join(os.getcwd(), SECOND_VIDEO_SHOTS_IMAGES_SELECTED_PATH + SHOT_FRAME + '.jpg')

    #TODO: ich draw_2D interrupts the script and the user has to decide if the pose depicted aligns with that of the shoooter. Otherwise he has to choose another person_index.
    check1 = read_json(os.path.join(os.getcwd(), FIRST_VIDEO_SHOTS_IMAGES_JSON_PATH + SHOT_FRAME + '_keypoints.json'), 0)
    draw_2D(check1)
    check2 = read_json(os.path.join(os.getcwd(), SECOND_VIDEO_SHOTS_IMAGES_JSON_PATH + SHOT_FRAME + '_keypoints.json'), 0)
    draw_2D(check2)

    R = np.load('R_Measure1.npy', allow_pickle=True)
    T = np.load('T_Measure1.npy', allow_pickle=True)
    mtx1 = np.load('mtx_DSC_0644_calibrate.npy', allow_pickle=True)
    mtx2 = np.load('mtx_MAH05312_calibrate.npy', allow_pickle=True)
    pts = triangulate(mtx1, mtx2, R, T, picture1, picture2, check1, check2)
    write_3D_points_to_json("Triangulated_Points/3D_Points_" + SHOT_FRAME, pts)
    plot_human_pose_3d(pts)