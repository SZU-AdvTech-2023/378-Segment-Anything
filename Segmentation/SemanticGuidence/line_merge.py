import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import line_filter as lf


# 线段被按角度划分，给定一个传入的角度 theta，然后传入的线段集合 (theta-1, theta+1)
# 平行的判定为：夹角小于某阈值


def calculate_angles(line_segments):
    """ Calculate the angle of each line segment with respect to the x-axis. """
    delta_y = line_segments[:, 3] - line_segments[:, 1]
    delta_x = line_segments[:, 2] - line_segments[:, 0]
    angles = np.arctan2(delta_y, delta_x)
    return np.mod(angles, np.pi)  # Ensure the angles are within [0, π]


def calculate_angle(segment):
    """ Calculate the angle of each line segment with respect to the x-axis. """
    delta_y = segment[3] - segment[1]
    delta_x = segment[2] - segment[0]
    angle = np.arctan2(delta_y, delta_x)
    return np.mod(angle, np.pi)  # Ensure the angles are within [0, π]


def calculate_length_and_angle(segment):
    """
    Calculate the length and angle of a line segment.

    Parameters:
    - segment: A NumPy array representing the line segment as [x1, y1, x2, y2].

    Returns:
    - length: The length of the line segment.
    - angle: The angle of the line segment in radians, within the range [0, pi].
    """
    # Calculate the length of the segment
    length = np.linalg.norm(segment[:2] - segment[2:])

    # Calculate the angle of the segment
    dx = segment[2] - segment[0]
    dy = segment[3] - segment[1]
    angle = np.arctan2(dy, dx) % np.pi  # Ensure the angle is within [0, pi]

    return length, angle


def calculate_segment_lengths(segments):
    """
    Calculate the lengths of line segments.
    Parameters:
    - segments: An array of shape (n, 4) where each row represents a line segment in the form [x1, y1, x2, y2].
    Returns:
    - lengths: An array of shape (n,) containing the length of each line segment.
    """
    # Calculate the lengths using Euclidean distance
    delta = segments[:, 2:4] - segments[:, 0:2]
    lengths = np.sqrt(np.sum(delta ** 2, axis=1))
    return lengths


def distance_between_segments(seg1, seg2):
    # 计算两线段中点
    mid1 = (seg1[:2] + seg1[2:]) / 2
    mid2 = (seg2[:2] + seg2[2:]) / 2
    # 计算线段方向向量
    dir1 = np.array([-seg1[1] + seg1[3], seg1[0] - seg1[2]])
    # 计算单位法向量
    norm1 = dir1 / np.linalg.norm(dir1)
    # 计算两线段中点的连线投影到线段1法线的长度
    distance = np.abs(np.dot(mid2 - mid1, norm1))
    return distance


def calculate_parallelism_and_distances(line_segments, angle_threshold):
    n = line_segments.shape[0]
    angles = np.arctan2(line_segments[:, 3] - line_segments[:, 1],
                        line_segments[:, 2] - line_segments[:, 0]) % np.pi
    angle_diffs = np.abs(np.subtract.outer(angles, angles))
    angle_diffs = np.minimum(angle_diffs, np.pi - angle_diffs)
    parallel_matrix = (angle_diffs < angle_threshold).astype(int)
    np.fill_diagonal(parallel_matrix, 0)
    # 只计算上三角矩阵部分以避免重复计算
    distance_matrix = np.full((n, n), np.inf)
    ix_upper_tri = np.triu_indices(n, k=1)  # 获取上三角矩阵的索引
    for i, j in zip(*ix_upper_tri):
        if parallel_matrix[i, j]:
            distance_matrix[i, j] = distance_between_segments(line_segments[i], line_segments[j])
    # 使距离矩阵对称
    distance_matrix = distance_matrix + distance_matrix.T - np.diag(np.diag(distance_matrix))
    return parallel_matrix, distance_matrix


def find_closest_farthest_endpoints(L1, L2):
    # Extract endpoints for L1 and L2
    endpoints_L1 = np.array([L1[:2], L1[2:]])
    endpoints_L2 = np.array([L2[:2], L2[2:]])

    # Compute all distances between endpoints of L1 and L2
    dist_matrix = np.linalg.norm(endpoints_L1[:, np.newaxis, :] - endpoints_L2[np.newaxis, :, :], axis=2)
    # Find the indices of the minimum distance
    min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    closest_points = (endpoints_L1[min_dist_idx[0]], endpoints_L2[min_dist_idx[1]])
    min_distance = dist_matrix[min_dist_idx]
    max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    farthest_points = (endpoints_L1[max_dist_idx[0]], endpoints_L2[max_dist_idx[1]])
    max_distance = dist_matrix[max_dist_idx]
    # Adjust indices to be 1-based instead of 0-based
    closest_points_indices = (min_dist_idx[0], min_dist_idx[1])

    return closest_points, farthest_points, min_distance, max_distance, closest_points_indices


def merge_two_lines(l1, l2, len1, len2, angle1, angle2, ts, t_theta):
    closest_points, farthest_points, min_distance, max_distance, closest_points_indices = find_closest_farthest_endpoints(
        l1, l2)
    c1, c2, d = closest_points[0], closest_points[1], min_distance
    if d > ts:
        return None
    lamb = len2 / len1 + d / ts
    tao_theta_star = t_theta * (1 - 1 / (1 + np.power(np.e, -2 * (lamb - 1.5))))
    theta = np.abs(angle1 - angle2)
    if max_distance < len1:
        return None
    if theta < tao_theta_star or theta > (np.pi - tao_theta_star):
        # 关键点：如何进行融合
        f1, f2 = farthest_points[0], farthest_points[1]
        merge_angel = calculate_angle([f1[0], f1[1], f2[0], f2[1]])
        if np.abs(angle1 - merge_angel) > (0.5 * t_theta):
            return None
        else:
            return np.array([f1[0], f1[1], f2[0], f2[1]])
    else:
        return None


def LSM():
    # load lines from LSD
    lines = np.loadtxt('F:/SegAndAlign/MVS/line_detect_and_filter/lines_filtered_txt/image0000_filtered_lines.txt',
                       dtype=np.float32)
    # cal angles
    angles = calculate_angles(lines)
    # cal lens
    lens = calculate_segment_lengths(lines)

    # sort from lens
    sorted_from_len_idx = np.argsort(lens)
    sorted_lens = lens[sorted_from_len_idx]
    sorted_angles = angles[sorted_from_len_idx]
    sorted_lines = lines[sorted_from_len_idx]  # L-space

    # hyper parameters
    n_ori = len(sorted_lines)  # 线段总数，当数量不变时则合并完成
    c_s = 0.02  # 距离阈值，当两线段之间的最短端点距离小于阈值则判定为空间邻近
    tao_theta = 0.5 * np.pi / 180.0  # 角度阈值，当两线段的锐夹角小于阈值则判定为角度邻近

    # 线段以依据长度进行降序排序   长-->短
    temp_n = n_ori
    i = 0
    while (True):
        for idx in range(n_ori):
            # 选取主线段
            main_angle = sorted_angles[idx]
            main_len = sorted_lens[idx]
            tao_s = c_s * main_len
            # angle filter idx  筛选出满足角度邻近的线的索引
            angle_filter_idx = np.argwhere(
                np.logical_or(
                    np.abs(sorted_angles - main_angle) < tao_theta,
                    np.abs(sorted_angles + main_angle - np.pi) < tao_theta
                )
            )
            angle_filter_idx = angle_filter_idx[angle_filter_idx != idx]
            P_l = sorted_lines[angle_filter_idx]
            P_angle = sorted_angles[angle_filter_idx]
            P_len = sorted_lens[angle_filter_idx]
            for id, line in enumerate(P_l):
                merge_line = merge_two_lines(l1=sorted_lines[idx], l2=P_l[id],
                                             len1=sorted_lens[idx], len2=P_len[id],
                                             angle1=sorted_angles[idx], angle2=P_angle[id],
                                             ts=tao_s, t_theta=tao_theta)
                if merge_line is not None:
                    main_idx = angle_filter_idx[id]
                    sorted_lines[idx] = merge_line
                    sorted_angles[idx], sorted_lens[idx] = calculate_length_and_angle(merge_line)
                    sorted_lines[main_idx:-1] = sorted_lines[main_idx + 1:]
                    sorted_lens[main_idx:-1] = sorted_lens[main_idx + 1:]
                    sorted_angles[main_idx:-1] = sorted_angles[main_idx + 1:]
                    temp_n -= 1
                    i += 1
                    print('Do {:08d} times merge.....'.format(i))
        if n_ori == temp_n:
            break
        else:
            n_ori = temp_n
    merge = sorted_lines[:n_ori]
    print('Before merging: {:06d} lines...'.format(len(lines)))
    print('After merging: {:06d} lines...'.format(len(merge)))
    np.savetxt('merge.txt', merge)
    return merge


if __name__ == '__main__':
    # root = 'F:/SegAndAlign/MVS/'
    # image_root_path = root + 'source_images/'
    # txt_root_path = root + 'line_detect_and_filter/lines_txt/'
    # image_names = [image_root_path + _ for _ in os.listdir(image_root_path)]
    # lsd_txt_names = [txt_root_path + _ for _ in os.listdir(txt_root_path)]
    # lines = LSM()  # LSM algorithm
    # # visualize and save
    # image = cv2.imread('F:/SegAndAlign/MVS/source_images/DJI_20230516152238_0012_V.JPG')
    # binary = np.zeros_like(image)
    # lf.draw_lines(lines, image, 'merge.jpg')
    # lf.draw_lines(lines, binary, 'merge_binary.jpg')
    # # parallel_matrix, distance_matrix = calculate_parallelism_and_distances(lines, np.radians(0.5))

    # LSM matlab visual
    image = cv2.imread('F:/SegAndAlign/MVS/source_images/DJI_20230516152238_0012_V.JPG')
    binary = np.zeros_like(image)
    lines = np.loadtxt('E:/00_Code/MWorks/output.txt')[:, :4].astype(np.float32)
    lf.draw_lines(lines, image, 'merge.jpg')
    lf.draw_lines(lines, binary, 'binary_merge.jpg')
    ###########

    print('This is line_merge.py...')
