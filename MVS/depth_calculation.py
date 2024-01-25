import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def compute_depth_maps(img1_path, img2_path, K1, T1, D1, K2, T2, D2):
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 畸变校正
    img1_undistorted = cv2.undistort(img1, K1, D1, None, K1)
    img2_undistorted = cv2.undistort(img2, K2, D2, None, K2)
    # 特征检测和匹配
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_undistorted, None)
    kp2, des2 = sift.detectAndCompute(img2_undistorted, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    # 提取匹配点坐标
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    # 三角测量
    points_3d = triangulate_points(pts1, pts2, K1, T1, K2, T2)
    save_point_cloud(points_3d, 'out.pcd')
    # 生成深度图
    # depth_map1 = create_depth_map(points_3d, K1, T1, img1_undistorted.shape)
    # depth_map2 = create_depth_map(points_3d, K2, T2, img2_undistorted.shape)

    # return depth_map1, depth_map2


def triangulate_points(pts1, pts2, K1, T1, K2, T2):
    # 构建投影矩阵
    P1 = K1 @ T1[:3, :4]
    P2 = K2 @ T2[:3, :4]
    # 三角测量
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
    return points_4d[:3, :].T


def create_depth_map(points_3d, K, T, img_shape):
    # 初始化深度图
    depth_map = np.zeros(img_shape[:2])

    # 投影点到图像平面
    R = T[:3, :3]
    for point in points_3d:
        point_img = K @ (R @ point[:3] + T[:3, 3])
        x, y, z = int(point_img[0] / point_img[2]), int(point_img[1] / point_img[2]), point_img[2]
        if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
            depth_map[y, x] = z
    return depth_map


def save_point_cloud(points_3d, filename):
    # 创建一个Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    # 将NumPy数组转换为Open3D的Vector3dVector格式
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    # 保存点云为PLY文件
    o3d.io.write_point_cloud(filename, pcd)


def visualize_depth_map(depth_map):
    # 深度图中可能存在负值，表示无效或无法估计的深度
    # 将这些值设置为NaN，以便在可视化时忽略它们
    depth_map[depth_map < 0] = np.nan
    # 创建一个绘图
    plt.figure(figsize=(20, 20))
    # 显示深度图，可以选择一个合适的颜色映射
    plt.imshow(depth_map, cmap='plasma')  # 'plasma'是一个良好的颜色映射，用于深度数据
    # 添加颜色条，显示深度值的范围
    plt.colorbar()
    # 显示图像
    plt.show()


# 假设depth_map是你的深度图矩阵
# visualize_depth_map(depth_map)


if __name__ == '__main__':
    source_image_path = 'F:/SegAndAlign/MVS/source_images/'
    poses_path = 'F:/SegAndAlign/MVS/poses/'
    dist = np.loadtxt('F:/SegAndAlign/MVS/distortion.txt')
    intrinsic = np.loadtxt('F:/SegAndAlign/MVS/intrinsic.txt')

    img1_path = source_image_path + 'DJI_20230516152243_0016_V.JPG'
    img2_path = source_image_path + 'DJI_20230516152244_0017_V.JPG'
    pose1 = np.loadtxt(poses_path + 'DJI_20230516152243_0016_V_POSE.txt')
    pose2 = np.loadtxt(poses_path + 'DJI_20230516152244_0017_V_POSE.txt')

    compute_depth_maps(img1_path, img2_path,
                       K1=intrinsic, K2=intrinsic,
                       D1=dist, D2=dist,
                       T1=pose1, T2=pose2)
    # visualize_depth_map(depth[0])
    # visualize_depth_map(depth[1])
    print('This is depth_calculation.py...')
