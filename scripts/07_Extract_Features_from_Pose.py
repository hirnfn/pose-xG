import json
import os
import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def read_json(file):
    """
    Helper function that reads the 3D points of the previous step '06_Triangulation' from a .json-file.

    :param file: .json-file that contains the 3D points
    """
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


def calculate_angle_from_two_points(a, b):
    """
    Wrapper function that specifically calculates the inclination angle.

    :param a: Coordinates of the knee.
    :param b: Coordinates of the ankle.

    :return: An angle in °.
    """
    c = [a[0], b[1], a[2]]
    return calculate_angle_from_three_points(a, b, c)


def calculate_angle_from_three_points(a, b, c):
    """
    Function that calculates the angle between three points.

    :param a: First point.
    :param b: Second point.
    :param c: Third point.

    :return: An angle in °.
    """
    vector1 = np.array(a) - np.array(b)
    vector2 = np.array(c) - np.array(b)
    v1 = np.array(vector1) / np.linalg.norm(vector1)
    v2 = np.array(vector2) / np.linalg.norm(vector2)
    dot = np.dot(v1, v2)
    return math.acos(dot) * 180 / math.pi


def calculate_angle_from_four_points(a, b, c, d):
    """
    Function that specifically calculates the arm angle.

    :param a: Coordinates of the sternum.
    :param b: Coordinates of the mid point of the hips.
    :param c: Coordinates of the shoulder.
    :param d: Coordinates of the elbow.

    :return: An angle in °.
    """
    vector1 = np.array(b) - np.array(a)
    vector2 = np.array(d) - np.array(c)
    v1 = np.array(vector1) / np.linalg.norm(vector1)
    v2 = np.array(vector2) / np.linalg.norm(vector2)
    dot = np.dot(v1, v2)
    return math.acos(dot) * 180 / math.pi


def calculate_trajectory(a, b, c, d, e, f):
    """
    Function that calculates the angle of a vector towards the normal vector of a plain.
    Used for the trunk angle (where the plain is the ground)
    and the support leg toe trajectory (where the plain is the upper body).

    :param a: Coordinates of the first corner of the plain.
    :param b: Coordinates of the second corner of the plain.
    :param c: Coordinates of the third corner of the plain.
    :param d: Coordinates of the fourth corner of the plain.
    :param e: Coordinates of the first point of the vector.
    :param f: Coordinates of the second point of the vector.

    :return: An angle in °.
    """
    vector1 = np.array(b) - np.array(a)
    vector2 = np.array(d) - np.array(c)
    v1 = np.array(vector1) / np.linalg.norm(vector1)
    v2 = np.array(vector2) / np.linalg.norm(vector2)
    normal = np.cross(v1, v2)
    vector3 = np.array(f) - np.array(e)
    v3 = np.array(vector3) / np.linalg.norm(vector3)
    dot = np.dot(normal, v3)
    return math.acos(dot) * 180 / math.pi


def calculate_left_foot(array, pts1, pts2, i):
    """
    Helper function that calculates all the features for the left foot.

    :param array: Empty array to be filled with the values.
    :param pts1: 3D coordinates when the support leg is placed.
    :param pts2: 3D coordinates when the ball is struck.
    :param i: loop index variable.
    """
    # Knee Angle at support foot placement
    array[i][0] = calculate_angle_from_three_points(pts1['9'], pts1['10'], pts1['11'])
    # Support Leg Toe Trajectory
    array[i][1] = calculate_trajectory(pts1['8'], pts1['1'], pts1['5'], pts1['2'], pts1['11'], pts1['22'])
    # Arm Angle at support foot placement
    array[i][2] = calculate_angle_from_four_points(pts1['1'], pts1['8'], pts1['2'], pts1['3'])
    # Knee Angle at strike
    array[i][3] = calculate_angle_from_three_points(pts2['9'], pts2['10'], pts2['11'])
    # Trunk Angle at strike
    array[i][4] = calculate_trajectory([0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], pts1['1'], pts1['8'])
    # Ankle Angle at strike
    array[i][5] = calculate_angle_from_three_points(pts2['10'], pts2['11'], pts2['22'])
    # Body Inclination Angle at strike
    array[i][6] = calculate_angle_from_two_points(pts2['10'], pts2['11'])


def calculate_right_foot(array, pts1, pts2, i):
    """
    Helper function that calculates all the features for the right foot.

    :param array: Empty array to be filled with the values.
    :param pts1: 3D coordinates when the support leg is placed.
    :param pts2: 3D coordinates when the ball is struck.
    :param i: loop index variable.
    """
    # Knee Angle at support foot placement
    array[i][0] = calculate_angle_from_three_points(pts1['12'], pts1['13'], pts1['14'])
    # Support Leg Toe Trajectory
    array[i][1] = calculate_trajectory(pts1['8'], pts1['1'], pts1['5'], pts1['2'], pts1['14'], pts1['19'])
    # Arm Angle at support foot placement
    array[i][2] = calculate_angle_from_four_points(pts1['1'], pts1['8'], pts1['5'], pts1['6'])
    # Knee Angle at strike
    array[i][3] = calculate_angle_from_three_points(pts2['12'], pts2['13'], pts2['14'])
    # Trunk Angle at strike
    array[i][4] = calculate_trajectory([0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], pts1['1'], pts1['8'])
    # Ankle Angle at strike
    array[i][5] = calculate_angle_from_three_points(pts2['10'], pts2['11'], pts2['22'])
    # Body Inclination Angle at strike
    array[i][6] = calculate_angle_from_two_points(pts2['13'], pts2['14'])


def calculate_features(folder_name, shots_list, dataset):
    """
    Wrapper function that calculates the features for a list of shots and saves them in a excel-file.

    :param folder_name: Name of the folder containing the .json-files with the 3D points.
    :param shots_list: A List of the frame names of the shots to be analyzed.
    """
    json_path = os.path.join(os.getcwd(), folder_name)
    rows = int(len(shots_list)/2)
    columns = ['knee_angle_support', 'toe_trajectory_support', 'arm_angle_support', 'knee_angle_strike', 'trunk_angle_strike', 'ankle_angle_strike', 'inclination_angle_strike']
    array = np.empty((rows, 7))

    df = pd.read_excel(dataset)
    new_dataset = os.path.splitext(dataset)[0] + '_Bio.xlsx'

    for i in range(rows):
        pts1 = read_json(os.path.join(json_path, '3D_Points_' + str(shots_list[2 * i]) + '.json'))
        pts2 = read_json(os.path.join(json_path, '3D_Points_' + str(shots_list[2 * i + 1]) + '.json'))

        # print(df.iloc[i]['Body Part'])
        if df.iloc[i]['Body Part'] == 'Strong Foot':
            calculate_right_foot(array, pts1, pts2, i)
        elif df.iloc[i]['Body Part'] == 'Weak Foot':
            calculate_left_foot(array, pts1, pts2, i)

    df2 = pd.DataFrame(array, columns=columns)
    result = pd.concat([df, df2], axis=1)
    result.to_csv('data.csv')
    result.to_excel(new_dataset)


def json_dict_to_array(keypoints):
    """
    Helper function that transforms the coordinates from a dictionary into an array to be plotted.

    :param keypoints: A dictionary conatining the keypoints.
    :return: res: An array containing the same keypoints.
    """
    quantity = len(keypoints)
    res = np.empty((quantity, 3))
    for i in range(quantity):
        res[i] = keypoints[str(i)]
    return res


def plot_human_pose_3d(keypoints):
    """
    Direct copy of the same function in 06_Triangulation. Used to check the footedness of the player.

    :param keypoints: 3D points from triangulation.
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

    # connections = [[8, 12], [12, 13], [13, 14], [14, 19], [14, 21]]

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


if __name__ == "__main__":
    #TODO: Copy SHOT_FRAMES from 04_Camera_Calibration_and_Triangulation
    SHOT_FRAMES = []
    OUTPUT_EXCEL_NAME = 'Data.xlsx'
    calculate_features('Triangulated_Points', SHOT_FRAMES, OUTPUT_EXCEL_NAME)