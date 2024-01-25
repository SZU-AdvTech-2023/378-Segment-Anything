import os
from FeatureDetection import feature
import cv2
import numpy as np
import matplotlib.pyplot as plt


def cal_angle(lines):
    """
    lines: [x1,y1,x2,y2]
    """
    delta_x = lines[:, 2] - lines[:, 0]
    delta_y = lines[:, 3] - lines[:, 1]
    angle = np.arctan2(delta_y, delta_x)
    angle = np.mod(angle, np.pi)
    return angle


def draw_statistic(angles, save_path=None):
    # 统计直方图数据
    hist, bins = np.histogram(angles, bins=30, range=(0, np.pi))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # 创建极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # 绘制条形图
    ax.bar(bin_centers, hist, width=(np.pi / 30), color='r', alpha=0.75)
    # 设置角度范围是0到π，0度位于顶部
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    # 设置角度标签
    ax.set_xticks(np.linspace(0, np.pi, 5))
    ax.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
    # 移除极径标签
    ax.set_yticklabels([])
    # 显示图形
    plt.show()
    if save_path is not None:
        plt.savefig(save_path, ax)


def draw_lines(lines, image, save_path=None):
    drawer = cv2.createLineSegmentDetector()
    draw_image = drawer.drawSegments(image, lines)
    if save_path is not None:
        cv2.imwrite(save_path, draw_image)


def filter_line_segments_by_angular_distribution(line_segments, segment_angles, num_angle_bins, min_ratio_threshold):
    """
    Filter line segments based on their angular distribution across specified intervals.

    Args:
    line_segments (np.ndarray): Array of shape (n_segments, 4) with each line segment as [x1, y1, x2, y2].
    segment_angles (np.ndarray): Array of shape (n_segments,) with the angle in radians for each line segment.
    num_angle_bins (int): Number of intervals to divide the angle range [0, pi] into.
    min_ratio_threshold (float): Minimum ratio threshold of segments within a bin to be kept.

    Returns:
    np.ndarray: Filtered array of line segments.
    np.ndarray: Array of angles corresponding to the filtered line segments.
    """
    # Define bins for the histogram
    angle_bins = np.linspace(0, np.pi, num=num_angle_bins + 1)
    # Compute histogram of angles
    angle_hist, _ = np.histogram(segment_angles, bins=angle_bins)
    # Calculate ratio of segments per bin
    segment_ratios = angle_hist / len(segment_angles)
    # Identify bins that meet the minimum ratio threshold
    valid_bins_mask = segment_ratios >= min_ratio_threshold
    # Find indices of valid angles
    valid_angle_indices = np.digitize(segment_angles, bins=angle_bins[:-1], right=True) - 1
    valid_segments_mask = valid_bins_mask[valid_angle_indices]
    # Filter segments and angles
    filtered_segments = line_segments[valid_segments_mask]
    filtered_angles = segment_angles[valid_segments_mask]
    return filtered_segments, filtered_angles


if __name__ == '__main__':
    image_root_path = 'F:/SegAndAlign/MVS/source_images/'
    images_names = [image_root_path + _ for _ in os.listdir(image_root_path)]
    num_of_images = len(images_names)
    root_save_path = 'F:/SegAndAlign/MVS/line_detect_and_filter/'
    image_save_path = root_save_path + 'line_draw/'
    txt_save_path = root_save_path + 'lines_txt/'
    filtered_txt_save_path = root_save_path + 'lines_filtered_txt/'

    # params experience
    sigma_scale = 0.95  # 这个参数在目标图像数据中对检测基本没有影响
    ang_th = 10  # 参数越大，检测到的线段越多，太小的话会导致很多线框检测不到，建议 [18-30] 如果有好的融合过滤算法，数值可以大一点
    density_th = 0.4  # 基本没什么影响，保持默认值即可

    line_detector = feature.LSDLineDetect(sigma_scale=0.6, ang_th=25)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if not os.path.exists(txt_save_path):
        os.makedirs(txt_save_path)
    if not os.path.exists(filtered_txt_save_path):
        os.makedirs(filtered_txt_save_path)
    for idx, _ in enumerate(images_names):
        image = cv2.imread(_)
        filtered_image = cv2.imread(_)
        black_back = np.zeros_like(image)
        filtered_black_back = np.zeros_like(image)
        # line detect
        lines, width, conf, nfa = line_detector.detect(image)
        lines = lines.squeeze()
        angles = cal_angle(lines)
        lines_with_width = np.concatenate([lines, width], axis=1)

        # without any filter
        # origin lines
        line_detector.draw_lines(image, lines,
                                 save_path=image_save_path + 'ori_image/',
                                 label='image{:04d}_lsd_'.format(idx))
        line_detector.draw_lines(black_back, lines,
                                 save_path=image_save_path + 'binary_image/',
                                 label='binary_image{:04d}_lsd_'.format(idx))
        np.savetxt(txt_save_path + 'image{:04d}_ori_lines_with_width.txt'.format(idx), lines_with_width)

        # # filter from width, lens
        # delta = lines[:, 2:4] - lines[:, 0:2]
        # lengths = np.sqrt(np.sum(delta ** 2, axis=1))
        # wid_keep_idx = np.where(width < 40)[0]
        # len_keep_idx = np.where(lengths > 15)[0]
        # # 两数组交集且去重
        # combine_keep_idx = np.intersect1d(wid_keep_idx, len_keep_idx)
        # # # 两数组并集且去重
        # # combine_filter_idx = np.union1d(wid_keep_idx, len_keep_idx)
        #
        # # do filter
        # filtered_lines = lines[combine_keep_idx]
        # filtered_angles = angles[combine_keep_idx]
        #
        # # filter from angle
        # filtered_lines, filtered_angles = filter_line_segments_by_angular_distribution(filtered_lines, filtered_angles,
        #                                                                                360, 0.0005)
        # np.savetxt(filtered_txt_save_path + 'image{:04d}_filtered_lines.txt'.format(idx), filtered_lines)
        # # draw filtered
        # draw_lines(filtered_lines, filtered_image,
        #            image_save_path + 'ori_image/image{:04d}_filtered.jpg'.format(idx))
        # draw_lines(filtered_lines, filtered_black_back,
        #            image_save_path + 'binary_image/image{:04d}_filtered.jpg'.format(idx))

    print('This is line_filter.py...')
