import trimesh
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

# file path
file_path = {
    'image': 'F:/L7/01.原始照片/images/',
    'intrinsic': 'F:/L7/相机参数/intrinsics/',
    'poses': 'F:/L7/相机参数/poses/'
}

SensorSize = [23.5, 15.7]
Resolution = [6000, 4000]
PixelSize = SensorSize[1] / Resolution[1]
print(PixelSize)

# w:0.003916666666666666
# h:0.003925

images_path = [file_path['image'] + _ for _ in os.listdir(file_path['image'])]
intrinsic_path = [file_path['intrinsic'] + _ for _ in os.listdir(file_path['intrinsic'])]
poses_path = [file_path['poses'] + _ for _ in os.listdir(file_path['poses'])]

intrinsics = [np.loadtxt(intrinsic) for intrinsic in intrinsic_path]
poses = [np.loadtxt(pos) for pos in poses_path]


# # print(len(intrinsics))
# location = [poses[_][:3, 3] for _ in range(len(poses))]
# location = np.array(location)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# X = location[:, 0]
# Y = location[:, 1]
# Z = location[:, 2]
# cmap = plt.get_cmap('tab10')
# norm = plt.Normalize(vmin=-3, vmax=3)
# ax.scatter(X, Y, Z, c=Z, cmap='rainbow')
# plt.show()


def param_process(pos, intrinsic):
    R = pos[:3, :3]
    t = pos[:3, 3]
    camera_center = -np.dot(R.T, t)

    Z = -R[:, 2]  # look_at
    X = R[:, 0]  # right
    Y = np.cross(Z, X)  # view_up
    # camera_center, right, view_up, look_at
    return camera_center, X, Y, Z


