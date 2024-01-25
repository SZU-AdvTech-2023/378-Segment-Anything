import queue
import os
import glob

import numpy as np
import open3d as o3d
from camera import *
import cv2


def get_geometry_center(coordinates):
    # coordinates: [n, 3], [n, 2]
    center = np.mean(coordinates, axis=0)
    return center


def get_similarity(camera_location, look_at, geometry_center):
    L1_distance = np.abs(camera_location - geometry_center)
    distance = np.sqrt(np.dot(L1_distance, L1_distance))
    nlook_at = look_at / np.linalg.norm(look_at)
    ndirection = (geometry_center - camera_location) / np.linalg.norm(geometry_center - camera_location)
    orientation_similarity = np.dot(nlook_at, ndirection)
    return distance, orientation_similarity


def pc_reprojection(mesh, intrinsic, pose, image, save_path):
    vertices = mesh.vertices
    normals = np.array(mesh.vertex_normals)
    R = pose[:3, :3]
    t = np.array(pose[:3, 3])
    t_rep = np.repeat(t, vertices.shape[0], axis=0)

    # 计算相机位姿朝向信息
    camera_look_at = np.array([np.dot(R, np.array([0.0, 0.0, 1.0]))])
    look_at_rep = np.repeat(camera_look_at, normals.shape[0], axis=0)

    # 顶点法向点乘相机朝向，用于后续的不可见点剔除
    dot_product = np.sum(normals * look_at_rep, axis=1)
    indices = np.where(dot_product > 0.0)

    # 世界坐标系转换到相机坐标系
    points_cam = np.dot(R, vertices.T) + t_rep.T
    # 相机坐标系到像素坐标系
    points_img = (intrinsic.dot(points_cam)).T

    scalar = np.ones_like(points_img)
    scalar[:, 0] = points_img[:, 2]
    scalar[:, 1] = points_img[:, 2]
    scalar[:, 2] = points_img[:, 2]
    points_img = points_img / scalar
    uv = points_img[:, :2]
    uv = uv[indices]

    points_cam = np.dot(R, vertices.T) + t_rep.T
    points_img = (intrinsic.dot(-points_cam)).T
    scalar = np.ones_like(points_img)
    scalar[:, 0] = points_img[:, 2]
    scalar[:, 1] = points_img[:, 2]
    scalar[:, 2] = points_img[:, 2]
    points_img = points_img / scalar
    uv = points_img[:, :2]
    proj_points = [_ for _ in uv if (_[0] > 0 and _[0] < 4000 and _[1] > 0 and _[1] < 6000)]
    proj_points = np.array(proj_points, dtype=np.int32)
    image = cv2.rotate(image, cv2.ROTATE_180)
    for p in range(len(proj_points)):
        cv2.circle(image, proj_points[p], radius=2, color=[0, 0, 255], thickness=3)
    # cv2.namedWindow("WINDOW", flags=cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("WINDOW", image)
    # cv2.waitKey(0)
    image = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(save_path, image)


def opencv_reproj(mesh, intrinsic, pose, dist, image, save_path):
    # 获取三维文件的顶点信息和顶点法向信息
    vertices = np.array(mesh.vertices)
    normals = np.array(mesh.vertex_normals)
    # 读取对应的相机内外参矩阵
    # K, R, t
    R = pose[:3, :3]
    t = np.array([pose[:3, 3]])

    # <K1>-0.0017061420153339907</K1>
    # <K2>-0.065791258580520359</K2>
    # <K3>0.053818937401014802</K3>
    # <P1>-0.00066751638634341044</P1>
    # <P2>0.00042294512275782402</P2>
    # opencv test
    cvproj_points = cv2.projectPoints(vertices, R, t, intrinsic, distCoeffs=dist)[0].squeeze()
    cvproj_points = np.array(cvproj_points, dtype=np.int32)

    image = cv2.rotate(image, cv2.ROTATE_180)
    for p in range(len(cvproj_points)):
        cv2.circle(image, cvproj_points[p], radius=3, color=[0, 0, 255], thickness=3)
    image = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(save_path, image)


def view_choice(mesh_file, images, intrinsics, poses, threshold=0.9, max_distance=200.0):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = np.array(mesh.vertices)
    # triangles = np.array(mesh.triangles)
    mesh.compute_vertex_normals()
    # vertex_normals = np.array(mesh.vertex_normals) / np.linalg.norm(np.array(mesh.vertex_normals), axis=1,
    #                                                                 keepdims=True)
    triangle_normals = np.array(mesh.triangle_normals) / np.linalg.norm(np.array(mesh.triangle_normals), axis=1,
                                                                        keepdims=True)
    avg_normal = np.mean(triangle_normals, axis=0) / np.linalg.norm(np.mean(triangle_normals, axis=0))
    center_point = np.mean(vertices, axis=0)
    Rs = np.array([p[:3, :3] for p in poses])
    ts = np.array([p[:3, 3] for p in poses])

    cam_location = []
    for _ in range(len(ts)):
        cam_location.append(np.dot(-Rs[_].T, ts[_]))
    cam_location = np.array(cam_location)

    # dirs = np.array([[center_point[0] - t[0], center_point[1] - t[1], center_point[2] - t[2]] for t in ts])
    dirs = np.array([[center_point[0] - t[0], center_point[1] - t[1], center_point[2] - t[2]] for t in cam_location])

    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    cam_look_at = [r[2, :3] for r in Rs]
    # distance = [np.sqrt((t[0] - center_point[0]) ** 2 + (t[1] - center_point[1]) ** 2 + (t[2] - center_point[2]) ** 2)
    #             for t in ts]
    distance = [np.sqrt((t[0] - center_point[0]) ** 2 + (t[1] - center_point[1]) ** 2 + (t[2] - center_point[2]) ** 2)
                for t in cam_location]

    cos_similarity = np.sum(np.array(dirs) * np.array(cam_look_at), axis=1)
    normal_filter = [np.dot(avg_normal, look_at) for look_at in cam_look_at]
    print(f'Min dis:{min(distance)}, mean dis:{np.mean(distance)}')
    print(f'Max cos:{max(cos_similarity)}, mean cos:{np.mean(cos_similarity)}')
    distance = np.where(np.array(distance) < max_distance)
    cos_similarity = np.where(np.array(cos_similarity) > threshold)
    normal_filter = np.where(np.array(normal_filter) < 0)

    same = np.intersect1d(distance, cos_similarity)
    same = np.intersect1d(same, normal_filter)
    result = [images[_] for _ in same]
    return result


def ray_tracing(mesh_file, intrinsic, pose, image):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    # 输入参数
    R = np.array(pose[:3, :3])  # 旋转矩阵
    t = np.array(pose[:3, 3])  # 平移向量
    h, w, c = image.shape
    # 计算相机光心位置
    camera_pos = t
    # 遍历每个像素
    for u in range(w):
        for v in range(h):
            # 根据内参计算归一化坐标
            x = (u - intrinsic[0, 2]) / intrinsic[0, 0]
            y = (v - intrinsic[1, 2]) / intrinsic[1, 1]
            # 计算归一化光线方向
            dx = x
            dy = y
            dz = 1

            # 用外参转到世界坐标系
            dx_world = R[0, 0] * dx + R[0, 1] * dy + R[0, 2] * dz
            dy_world = R[1, 0] * dx + R[1, 1] * dy + R[1, 2] * dz
            dz_world = R[2, 0] * dx + R[2, 1] * dy + R[2, 2] * dz

            # 输出光线方向
            direction = np.array([dx_world, dy_world, dz_world])
            print(direction)


def generate_ray(intrinsic, pose, image):
    # 图像的分辨率
    resy, resx, c = image.shape
    # 相机内参 3x3
    K = intrinsic
    K_inv = np.linalg.inv(K)
    # 外参矩阵的逆 4x4
    R_inv = np.linalg.inv(pose)
    t = pose[:3, 3]
    # 生成像素网格
    y_range = np.arange(0, resy, dtype=np.float32)
    x_range = np.arange(0, resx, dtype=np.float32)
    pixely, pixelx = np.meshgrid(y_range, x_range)
    # TODO Why z = 1 ?
    pixelz = np.ones_like(pixely)
    pixel = np.stack([pixelx, pixely, pixelz], axis=2).view([-1, 3])
    pixel_p = K_inv @ pixel.T

    pixel_world_p = R_inv[:3, :3] @ pixel_p + R_inv[:3, 3:4]
    ray_origin = R_inv[:3, 3:4]  # [3x1]
    ray_dir = pixel_world_p - ray_origin
    ray_dir = ray_dir.T  # [nx3]
    ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=1, keepdims=True)
    ray_origin = ray_origin.T.expand_as(ray_dir)
    return ray_origin, ray_dir


if __name__ == '__main__':
    # point projection
    # 三维数据
    # mesh = o3d.io.read_point_cloud('F:/1682561966.799244033.pcd', print_progress=True)
    # points = np.array(mesh.points)
    # 获取三维文件的顶点信息和顶点法向信息
    # vertices = np.array(mesh.vertices)
    # normals = np.array(mesh.vertex_normals)
    # o3d.io.write_triangle_mesh(f'F:/save.ply', mesh)
    # # 读取对应的相机内外参矩阵
    # # K, R, t
    # intrinsic = np.loadtxt('./DataProcess/intrinsics.txt')
    # P = np.loadtxt('F:/L7/相机参数/poses/6z00453.txt')
    # R = P[:3, :3]
    # R_inv = np.linalg.inv(R)
    # t = np.array([P[:3, 3]])
    # t_rep = np.repeat(t, vertices.shape[0], axis=0)
    #
    # dist = np.array([-0.0017061420153339907,
    #                  -0.065791258580520359,
    #                  0.053818937401014802,
    #                  -0.00066751638634341044,
    #                  0.00042294512275782402])
    # # <K1>-0.0017061420153339907</K1>
    # # <K2>-0.065791258580520359</K2>
    # # <K3>0.053818937401014802</K3>
    # # <P1>-0.00066751638634341044</P1>
    # # <P2>0.00042294512275782402</P2>
    # # opencv test
    # cvproj_points = cv2.projectPoints(vertices, R, t, intrinsic, distCoeffs=dist)[0].squeeze()
    # cvproj_points = np.array(cvproj_points, dtype=np.int32)
    #
    # # 计算相机位姿朝向信息
    # camera_look_at = np.array([np.dot(R, np.array([0.0, 0.0, 1.0]))])
    # look_at_rep = np.repeat(camera_look_at, normals.shape[0], axis=0)
    # # 顶点法向点乘相机朝向，用于后续的不可见点剔除
    # dot_product = np.sum(normals * look_at_rep, axis=1)
    # indices = np.where(dot_product > 0.0)
    #
    # # 世界坐标系转换到相机坐标系
    # points_cam = np.dot(R, vertices.T) + t_rep.T
    # # 相机坐标系到像素坐标系
    # points_img = (intrinsic.dot(points_cam)).T
    #
    # scalar = np.ones_like(points_img)
    # scalar[:, 0] = points_img[:, 2]
    # scalar[:, 1] = points_img[:, 2]
    # scalar[:, 2] = points_img[:, 2]
    # points_img = points_img / scalar
    # uv = points_img[:, :2]
    # uv = uv[indices]
    #
    # # proj_points = [_ for _ in uv if (_[0] > 0 and _[0] < 4000 and _[1] > 0 and _[1] < 6000)]
    # proj_points = np.array(uv, dtype=np.int32)
    # image = cv2.imread('F:/L7Images/6z00453.JPG')
    # image = cv2.rotate(image, cv2.ROTATE_180)
    # for p in range(len(cvproj_points)):
    #     cv2.circle(image, cvproj_points[p], radius=3, color=[0, 0, 255], thickness=3)
    # # cv2.namedWindow("WINDOW", flags=cv2.WINDOW_KEEPRATIO)
    # # cv2.imshow("WINDOW", image)
    # # cv2.waitKey(0)
    # image = cv2.rotate(image, cv2.ROTATE_180)
    # cv2.imwrite('./DataProcess/reproj_6z00453.jpg', image)

    # # plane cluster
    # mesh = o3d.io.read_triangle_mesh('./DataProcess/abstract_mesh/9th_simplifyed_mesh.obj')
    # vertices = np.array(mesh.vertices)
    # triangles = np.array(mesh.triangles)
    #
    # mesh.compute_vertex_normals()
    # vertex_normals = np.array(mesh.vertex_normals) / np.linalg.norm(np.array(mesh.vertex_normals), axis=1,
    #                                                                 keepdims=True)
    # triangle_normals = np.array(mesh.triangle_normals) / np.linalg.norm(np.array(mesh.triangle_normals), axis=1,
    #                                                                     keepdims=True)
    #
    # normal_cluster = []
    # index_cluster = [0]
    # normal_queue = queue.Queue(maxsize=10000000)
    # for _ in triangle_normals:
    #     normal_queue.put(_)
    # normal_cluster.append(normal_queue.get())
    # while not normal_queue.empty():
    #     label = True
    #     buff = normal_queue.get()
    #     for idx, _ in enumerate(normal_cluster):
    #         if np.dot(buff, _) > 0.98:
    #             index_cluster.append(idx)
    #             label = False
    #             break
    #         else:
    #             continue
    #     if label:
    #         index_cluster.append(len(normal_cluster))
    #         normal_cluster.append(buff)
    #
    # TriMesh = {}
    # for idx, triangle in enumerate(triangles):
    #     triangle_ele = {
    #         'vertices': [vertices[triangle[0]],
    #                      vertices[triangle[1]],
    #                      vertices[triangle[2]]],
    #         'triangle': triangle,
    #         'vertex_normals': [triangle_normals[triangle[0]],
    #                            triangle_normals[triangle[1]],
    #                            triangle_normals[triangle[2]]],
    #         'triangle_normal': triangle_normals[idx],
    #         'label': index_cluster[idx],
    #         'cos_similarity': 0.0
    #     }
    #     TriMesh[f'{idx}'] = triangle_ele
    #
    # num_of_cluster = len(normal_cluster)
    # for cls in range(num_of_cluster):
    #
    #     indices = np.where(np.array(index_cluster) == cls)[0]
    #
    #     mesh_plane = o3d.geometry.TriangleMesh()
    #     plane_vertices = []
    #     plane_triangle = []
    #     vertices_normal = []
    #     triangles_normal = []
    #     i = 0
    #     for _ in indices:
    #         plane_vertices.append(TriMesh[f'{_}']['vertices'][0])
    #         plane_vertices.append(TriMesh[f'{_}']['vertices'][1])
    #         plane_vertices.append(TriMesh[f'{_}']['vertices'][2])
    #         vertices_normal.append(TriMesh[f'{_}']['vertex_normals'][0])
    #         vertices_normal.append(TriMesh[f'{_}']['vertex_normals'][1])
    #         vertices_normal.append(TriMesh[f'{_}']['vertex_normals'][2])
    #         plane_triangle.append([np.uint(i * 3), np.uint(i * 3 + 1), np.uint(i * 3 + 2)])
    #         triangles_normal.append(TriMesh[f'{_}']['triangle_normal'])
    #         i += 1
    #     mesh_plane.vertices = o3d.utility.Vector3dVector(plane_vertices)
    #     mesh_plane.triangles = o3d.utility.Vector3iVector(plane_triangle)
    #     mesh_plane.vertex_normals = o3d.utility.Vector3dVector(vertices_normal)
    #     mesh_plane.triangle_normals = o3d.utility.Vector3dVector(triangles_normal)
    #     colors = [[np.random.rand(),
    #                np.random.rand(),
    #                np.random.rand()]] * len(mesh_plane.vertices)
    #     mesh_plane.vertex_colors = o3d.utility.Vector3dVector(colors)
    #     o3d.io.write_triangle_mesh(f'./DataProcess/plane_{cls}.ply', mesh_plane)

    # # view choice
    # # view_choice(mesh_file, images, intrinsics, poses, threshold=0.7, max_distance=100.0):
    # mesh_file = './DataProcess/plane_cluster/9th/plane_7.ply'
    # images = os.listdir('F:/L7/01SourceImages/images')
    # images = ['F:/L7/01SourceImages/images/' + _ for _ in images]
    # poses = os.listdir('F:/L7/相机参数/poses')
    # poses = [np.loadtxt('F:/L7/相机参数/poses/' + _) for _ in poses]
    # intrinsic = np.loadtxt('./DataProcess/intrinsics.txt')
    # res = view_choice(mesh_file, images, intrinsic, poses, threshold=0.98, max_distance=120)

    # ray tracing
    # ray_tracing(mesh_file=, intrinsic=, pose=, image=)

    print('This is meshdel.py...')
