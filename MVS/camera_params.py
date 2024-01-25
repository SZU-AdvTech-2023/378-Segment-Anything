import numpy as np
import math


def cal_intrinsic_from_xml():
    sensor_size_mm = 18.434999942779541  # 传感器对角线长度，单位：毫米
    image_width_pixels = 5280  # 图像宽度，单位：像素
    image_height_pixels = 3956  # 图像高度，单位：像素
    focal_length_mm = 12.289999961853027
    cx = 2646.1197706076591  # 主点x坐标
    cy = 1965.8168313920294  # 主点y坐标

    # 计算传感器宽度和高度的像素尺寸
    sensor_width_mm = sensor_size_mm / math.sqrt(1 + (image_height_pixels / image_width_pixels) ** 2)
    sensor_height_mm = sensor_size_mm / math.sqrt(1 + (image_width_pixels / image_height_pixels) ** 2)

    pixel_size_mm_x = sensor_width_mm / image_width_pixels
    pixel_size_mm_y = sensor_height_mm / image_height_pixels

    # 将焦距转换为像素单位
    fx = focal_length_mm / pixel_size_mm_x
    fy = focal_length_mm / pixel_size_mm_y

    # 构造内参矩阵
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1.]], dtype=np.float64)
    np.savetxt('F:/SegAndAlign/MVS/intrinsic.txt', K)


def cal_dist_from_xml():
    dist = np.array([-0.1028036652219434,
                     -0.0088418412928753486,
                     -0.0073012629880329959,
                     8.8006033773138309e-05,
                     0.00054663837530288383], dtype=np.float64)
    np.savetxt('F:/SegAndAlign/MVS/distortion.txt', dist)


def cal_pose_mat():
    rotation = np.array([-0.47622854119999836,
                         -0.8793153336623637,
                         0.0033046229389276371,
                         0.00012167193868914709,
                         -0.0038240461336510045,
                         -0.99999268090676985,
                         0.87932153490202991,
                         -0.47622465355902621,
                         0.001928107913076238], dtype=np.float64).reshape(3, 3)
    t = np.array([113.49999308333334, 22.748213194444446, 78.943000793457031], dtype=np.float64)
    extrinsic = np.eye(4, 4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = t
    np.savetxt('F:/SegAndAlign/MVS/DJI_20230516152244_0017_V_POSE.txt', extrinsic)


if __name__ == '__main__':
    cal_pose_mat()
    print('This is camera_params.py...')
