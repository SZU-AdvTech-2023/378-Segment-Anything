# coding=utf-8

import os
import time
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset as dataset
from Net import Net


class Inference(object):
    def __init__(self, Network, images_list):
        self.cfg = dataset.Config(dataset='PMD', datapath=None,
                                  snapshot='E:/00_Code/PyCharmProjects/UrbanSceneNet/GlassDetection/HetNet/checkpoints/pmd-model-best',
                                  mode='test')
        self.data = dataset.InferenceData(self.cfg, images_list)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, pin_memory=True)
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            i = 0
            for image in self.loader:
                image = image.cuda().float()
                torch.cuda.synchronize()
                out, out_edge = self.net(image)
                torch.cuda.synchronize()
                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                edge = (torch.sigmoid(out_edge[0, 0]) * 255).cpu().numpy()

                save_path = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/Results/galss_detec_results/HetNet_results/PMD/mask/'
                save_edge = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/Results/galss_detec_results/HetNet_results/PMD/edge/'

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not os.path.exists(save_edge):
                    os.makedirs(save_edge)

                cv2.imwrite(save_path + 'HetNet_MSD_mask_{:03d}.png'.format(i), np.round(pred))
                cv2.imwrite(save_edge + 'HetNet_MSD_edge_{:03d}.png'.format(i), np.round(edge))
                i += 1


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg = Dataset.Config(dataset='PMD', datapath=path, snapshot='./PMD-msd-model-best/msd-model-best',
                                  mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            cost_time = list()
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                out, out_edge = self.net(image, shape)
                torch.cuda.synchronize()
                cost_time.append(time.perf_counter() - start_time)
                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                edge = (torch.sigmoid(out_edge[0, 0]) * 255).cpu().numpy()

                save_path = './map-PMD/PMD/' + self.cfg.datapath.split('/')[-1]
                save_edge = './map-PMD/Edge/' + self.cfg.datapath.split('/')[-1]

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not os.path.exists(save_edge):
                    os.makedirs(save_edge)

                cv2.imwrite(save_path + '/' + name[0] + '.png', np.round(pred))
                cv2.imwrite(save_edge + '/' + name[0] + '_edge.png', np.round(edge))

            cost_time.pop(0)
            print('Mean running time is: ', np.mean(cost_time))
            print("FPS is: ", len(self.loader.dataset) / np.sum(cost_time))


if __name__ == '__main__':
    images_path = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/DataProcess/original_datas/for_glass_detec/'
    image_list = os.listdir(images_path)
    images_list = [images_path + _ for _ in image_list]
    inf = Inference(Net, images_list)
    inf.save()
    # for path in ['/home/crh/MirrorDataset/PMD/']:
    #     test = Test(dataset, Net, path)
    #     test.save()
    print('HetNet test.py...')
