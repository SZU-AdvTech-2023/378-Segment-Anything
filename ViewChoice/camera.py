import numpy as np
from numpy.linalg import linalg
import exifread
import pprint
import csv

print = pprint.pprint


# 如果知道相机传感器实际大小则使用原有焦距就行
# 如果不知道相机传感器实际大小则需要使用等效焦距
# 35mm [36mm, 24mm]


def load_camera_params(intrinsic_file, pose_file):
    intrinsic = np.loadtxt(intrinsic_file)
    pose = np.loadtxt(pose_file)

    rotation_matrix = pose[:3, :3]
    transform = pose[3, :3]

    return intrinsic, rotation_matrix, transform


def get_camera_world_position(rotation, transform):
    return -np.cross(np.transpose(rotation), transform)


def get_image_o_world_position(intrinsic, pixel_len, f):
    u0 = intrinsic[0, 3]
    v0 = intrinsic[1, 3]
    xc = u0 * pixel_len
    yc = v0 * pixel_len
    return np.array([xc, yc, f])


def get_image_center_world_position(intrinsic, f):
    return np.array([0, 0, f])


def camera_look_at_world(intrinsic, rotation, transform, f):
    camera_world_position = get_camera_world_position(rotation, transform)
    image_center_world_position = get_image_center_world_position(intrinsic, f)
    return image_center_world_position - camera_world_position


def get_image_exif(image_path):
    f = open(image_path, 'rb')
    tags = None
    try:
        tags = exifread.process_file(f)
    except Exception:
        print('Image does not contain EXIF...')
    f.close()
    return tags


def get_camera_matrix(x, y, altitude, yaw, pitch, roll):
    yaw = -138.6378 * np.pi / 180.0
    pitch = 37.6569 * np.pi / 180.0
    roll = -30.5921 * np.pi / 180.0
    # 横滚旋转
    P = np.eye(4, dtype=np.float32)
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
    # 俯仰旋转
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    # 航向旋转
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    # 相机姿态矩阵
    P[:3, :3] = np.dot(R_roll, np.dot(R_pitch, R_yaw))
    P[:3, 3] = np.array([x, y, altitude])
    return P


def read_csv(path):
    params = {}
    image_list = []
    P_list = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            P = get_camera_matrix(x=row['x'],
                                  y=row['y'],
                                  altitude=row['alt'],
                                  yaw=row['heading'],
                                  pitch=row['pitch'],
                                  roll=row['roll'])
            row['P'] = P
            image_list.append(row['#name'])
            P_list.append(P)
            params[row['#name']] = row
    return params, image_list, np.array(P_list)


if __name__ == '__main__':
    # params, image_list, P_list = read_csv('F:/L7rc/L7_calibreations.csv')
    # print(P_list)
    # print(image_list)
    print('This is camera.py...')
