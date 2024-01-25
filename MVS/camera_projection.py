import numpy as np
import cv2


class Camera(object):
    def __init__(self, K, R, t, dist=None):
        # 3x3 matrix
        self.K = K
        # 3x3 matrix
        self.R = R
        # 3x1 matrix
        self.t = t

        self.dist = dist

    def cam_location_w(self):
        return (-self.R.T) @ self.t

    def cam_orientation_w(self):
        return self.R[2, :]

    def projection(self, point_w):
        # 世界坐标系到相机坐标系
        X_c = self.R @ point_w + self.t
        # 相机坐标系到图像坐标系
        x = X_c[0] / X_c[2]
        y = X_c[1] / X_c[2]
        # TODO 去除径向畸变
        r2 = x * x + y * y

        # 图像坐标系到像素平面
        x = self.K @ X_c
        factor = X_c[2] * x[2]
        x = x / factor
        return x


if __name__ == '__main__':
    print('This is camera_projection.py...')
