import cv2
import numpy as np
import os

if __name__ == '__main__':

    image_root_path = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/DataProcess/original_datas/for_optical_flow/'
    save_root_path = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/Results/optical_flow_results/'
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    images = os.listdir(image_root_path)
    image_files = [image_root_path + _ for _ in images]
    # 读取两个连续帧的图像
    frame1 = cv2.imread(image_files[0])
    frame2 = cv2.imread(image_files[1])

    # 将图像转换为灰度
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 定义特征点的初始位置（可以使用角点检测算法来检测特征点）
    # 这里使用 Shi-Tomasi 角点检测算法
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    # 使用光流法计算特征点的新位置
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # 仅保留那些光流跟踪成功的特征点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 在第一帧图像上绘制特征点
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame1 = cv2.circle(frame1, (int(c), int(d)), 5, (0, 0, 255), -1)
        frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (0, 0, 255), -1)

    # 绘制光流跟踪的线
    mask = np.zeros_like(frame1)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
        frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (0, 0, 255), -1)
    img = cv2.add(frame2, mask)

    # 显示结果
    cv2.imwrite(save_root_path + 'LK_flow_map.png', img)

    # # 计算光流矢量
    # flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #
    # # 计算角度和大小
    # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #
    # # 将角度映射到颜色的方向
    # hue = angle * 180 / np.pi / 2
    #
    # # 将大小映射到颜色的强度
    # value = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    #
    # # 创建 HSV 图像
    # heatmap = np.zeros_like(frame1)
    # heatmap[..., 0] = hue
    # hue = (hue * 100.) % 180
    # heatmap[..., 1] = 180
    # heatmap[..., 2] = value
    # value = value / 10.
    #
    # # 将 HSV 转换为 BGR
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_HSV2BGR)
    #
    # cv2.imwrite(save_root_path + 'Farneback_flow_map.png', heatmap)
