import os.path

import matplotlib.pyplot as plt
import numpy as np


# arrays -> [nums, c, h, w]
def plot_arrays(arrays, cols=1, squeeze=None, titles=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    arrays = np.split(arrays, arrays.shape[0], axis=0)
    num_arrays = len(arrays)
    rows = int(np.ceil(num_arrays / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(12, 6), sharex=True, sharey=True)

    for i, array in enumerate(arrays):
        if squeeze:
            array = np.squeeze(array)
        array = array.permute(1, 2, 0)
        ax_curr = ax if cols == 1 else ax[i // cols, i % cols]
        if titles:
            ax_curr.set_title(titles[i])
        if array.ndim == 3:
            ax_curr.imshow(array)
        elif array.ndim == 2:
            ax_curr.imshow(array)

    plt.tight_layout()
    plt.show()


def visualize_array_heatmap(array, save_path, label):
    """
    使用热图颜色映射可视化一个numpy二维数组。
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.imshow(array, cmap='hot')  # 使用热图颜色映射
    plt.colorbar()  # 显示颜色条
    plt.savefig(save_path + label + '_heat_map.jpg')
