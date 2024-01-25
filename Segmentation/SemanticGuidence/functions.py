import cv2
import numpy as np


def remove_white_holes(binary_mask, kernel_size=3, kernel_shape=cv2.MORPH_RECT, iterations=3):
    # Create the kernel for morphological operation, the size can be tuned
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    # Apply morphological closing (dilate then erode)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return closed_mask


def fill_small_holes(mask, area_threshold):
    """
    填充小于特定面积阈值的白色空洞。

    Args:
        mask (np.array): 二值mask图像，黑色区域为0，白色区域为255。
        area_threshold (int): 空洞面积的阈值。

    Returns:
        filled_mask (np.array): 填充后的mask图像。
    """
    # 复制mask以进行修改
    filled_mask = mask.copy()
    # 查找所有连通区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            cv2.drawContours(filled_mask, [cnt], 0, 0, -1)
    return filled_mask


def txt_to_list(txt_file):
    with open(txt_file, 'r') as f:
        res = f.readlines()
    results = []
    for _ in res:
        results.append(_[:-1].replace("image_expand", "binary_mask"))
    return results


def cal_warp_mat(txt_file, confidence_filter=None):
    # txt--> keypoint0[x,y] keypoint1[x,y] conf[v]
    info = np.loadtxt(txt_file)
    keypoints0 = info[:, :2]
    keypoints1 = info[:, 2:4]
    conf = info[:, - 1]
    if confidence_filter is not None:
        index = np.where(conf > confidence_filter)
        keypoints0 = keypoints0[index]
        keypoints1 = keypoints1[index]
    M, _ = cv2.findHomography(keypoints1, keypoints0, cv2.RANSAC, maxIters=10)
    return M
