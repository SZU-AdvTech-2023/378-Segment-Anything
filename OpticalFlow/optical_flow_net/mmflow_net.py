from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow
import mmcv
import cv2
import os
import sys
import numpy as np


# FlowNet: Learning Optical Flow with Convolutional Networks (5 ckpts)
# FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks (7 ckpts)
# PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume (6 ckpts)
# LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation (9 ckpts)
# A Lightweight Optical Flow CNN-Revisiting Data Fidelity and Regularization (8 ckpts)
# Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation (5 ckpts)
# MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask (4 ckpts)
# RAFT: Recurrent All-Pairs Field Transforms for Optical Flow (5 ckpts)
# GMA: Learning to Estimate Hidden Motions with Global Motion Aggregation (13 ckpts)


def optical_flow_sequences(model, image_root_path, save_root_path, save_flo=False):
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    images = os.listdir(image_root_path)
    image_files = [image_root_path + _ for _ in images]

    for i in range(len(image_files) - 1):
        img1 = image_files[i]
        img2 = image_files[i + 1]
        img1 = cv2.imread(img1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(img2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, dsize=None, fx=0.8, fy=0.8)
        img2 = cv2.resize(img2, dsize=None, fx=0.8, fy=0.8)
        result = inference_model(model, img1, img2)
        if save_flo:
            write_flow(result, flow_file=save_root_path + f'optical_flow_{i}.flo')
        flow_map = visualize_flow(result, save_file=save_root_path + f'flow_map_{i}.png')


def optical_flow_2images(model, image01_path, image02_path, save_path, save_name, save_flo=False, resize=0.8):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image01 = cv2.imread(image01_path)
    image01 = cv2.cvtColor(image01, cv2.COLOR_BGR2RGB)
    image02 = cv2.imread(image02_path)
    image02 = cv2.cvtColor(image02, cv2.COLOR_BGR2RGB)
    if resize is not None:
        image01 = cv2.resize(image01, dsize=None, fx=resize, fy=resize)
        image02 = cv2.resize(image02, dsize=None, fx=resize, fy=resize)
    result = inference_model(model, image01, image02)
    if save_flo:
        write_flow(result, flow_file=save_path + save_name + '.flo')
    flow_map = visualize_flow(result, save_file=save_path + save_name + '.png')


if __name__ == '__main__':
    config_file = 'configs/raft/raft_8x2_100k_flyingthings3d_400x720.py'
    config_pth = 'checkpoints/raft_8x2_100k_flyingthings3d_400x720.pth'

    model = init_model(config_file, config_pth, device='cuda:0')
    images_root_path = 'F:/5buildings/Multi-view_images/'
    images_name = os.listdir(images_root_path)
    images_path = [images_root_path + _ for _ in images_name]
    indices = np.arange(len(images_name), dtype=np.int8)
    indices_matrix = indices.reshape(-1, 4)
    # print(indices_matrix)
    for row in range(indices_matrix.shape[0]):
        for col in range(indices_matrix.shape[1] - 1):
            print('Calculate the optical from image {:s} to {:s}...'.format(images_name[indices_matrix[row, col]],
                                                                            images_name[
                                                                                indices_matrix[row, col + 1]]))
            save_name = images_name[indices_matrix[row, col]].split('.')[0] + '_to_' + \
                        images_name[indices_matrix[row, col + 1]].split('.')[0]
            optical_flow_2images(model=model,
                                 image01_path=images_path[indices_matrix[row, col]],
                                 image02_path=images_path[indices_matrix[row, col + 1]],
                                 save_path='F:/5buildings/raft_optical_flows/',
                                 save_name=save_name, resize=0.4)

    # image_root_path = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/DataProcess/original_datas/for_optical_flow/'
    # save_root_path = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/Results/optical_flow_results/'
    print('This is mmflow_net.py...')
