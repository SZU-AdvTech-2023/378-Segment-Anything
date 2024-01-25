import numpy as np
import os
from FeatureDetection import feature
import cv2
from line_merge import calculate_segment_lengths


def main(filter_step, m_out=None, detect=True, len_threshold=10, record_ori=True, record_filter=True, with_width=True):
    image_root_path = 'F:/SegAndAlign/MVS/source_images/'
    images_names = [image_root_path + _ for _ in os.listdir(image_root_path)]
    idx = 0
    image = cv2.imread(images_names[idx])
    save_path = 'F:/SegAndAlign/MVS/seg_and_merge/image_{:04d}/'.format(idx)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if detect:
        lsd_detector = feature.LSDLineDetect()
        lines, width, conf, nfa = lsd_detector.detect(image)
        lines = lines.squeeze()
        if record_ori:
            if with_width:
                records = np.concatenate([lines, width], axis=1)
                np.savetxt(save_path + 'lines_with_width.txt', records)
            else:
                np.savetxt(save_path + 'lines.txt', lines)
        lens = calculate_segment_lengths(lines)
        keep_idx = np.where(lens > len_threshold)
        lines = lines[keep_idx]
        width = width[keep_idx]
        if record_filter:
            if with_width:
                records = np.concatenate([lines, width], axis=1)
                np.savetxt(save_path + 'lines_with_width_step{:04d}.txt'.format(filter_step), records)
            else:
                np.savetxt(save_path + 'lines_step{:04d}.txt'.format(filter_step), lines)


def filter(txt, len_threshold, save_path):
    lines = np.loadtxt(txt)
    lens = calculate_segment_lengths(lines[:, :4])
    keep_idx = np.where(lens > len_threshold)
    lines = lines[keep_idx]
    np.savetxt(save_path, lines)


def draw_lines(lines, image, save_path=None):
    drawer = cv2.createLineSegmentDetector()
    draw_image = drawer.drawSegments(image, lines)
    if save_path is not None:
        cv2.imwrite(save_path, draw_image)


if __name__ == '__main__':
    # main(filter_step=1, len_threshold=10)
    image = cv2.imread('F:/SegAndAlign/MVS/source_images/DJI_20230516152238_0012_V.JPG')
    ori_lines = np.loadtxt('F:/SegAndAlign/MVS/seg_and_merge/image_0000/lines_with_width.txt').astype(np.float32)
    filter_lines = np.loadtxt('F:/SegAndAlign/MVS/seg_and_merge/image_0000/lines_with_width_step0001.txt').astype(
        np.float32)
    merge_lines = np.loadtxt('F:/SegAndAlign/MVS/seg_and_merge/image_0000/output.txt').astype(np.float32)
    draw_lines(ori_lines[:, :4], np.zeros_like(image), 'F:/SegAndAlign/MVS/seg_and_merge/image_0000/LSD_detection.JPG')
    draw_lines(filter_lines[:, :4], np.zeros_like(image), 'F:/SegAndAlign/MVS/seg_and_merge/image_0000/filter_01.JPG')
    draw_lines(merge_lines[:, :4], np.zeros_like(image), 'F:/SegAndAlign/MVS/seg_and_merge/image_0000/merge_01.JPG')
    print('This is filter_and_merge.py')
