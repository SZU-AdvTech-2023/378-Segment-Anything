from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
import numpy as np
import cv2
from PIL import ImageFilter
from PIL import Image
import random
import platform

system_type = platform.system()
# 'Windows' 'Linux'
if system_type == 'Windows':
    from data.data_process import *

if system_type == 'Linux':
    import sys

    sys.path.append('/home/ubuntu/workdic/UrbanSceneNet/urban-scene-seg-net/DataProcess')
    from data_process import *


# 使用无监督方式判断是否存在玻璃，传统的正负样本不知道可不可行
# 把一张图片分割成多个patch，每一个patch作为一个样本传入网络
# 使用infoNCE loss
# 裁剪的尺寸可能需要设计好

# 对深度图做同样裁剪，然后检测玻璃区域
# TODO 联合训练还是使用RGB图像单独训练玻璃检测网络

# 假设图像尺寸[6000, 4000] -> resize 0.5 -> [3000, 2000] ~ 30 patches
# 目前将图像 resize=0.5 之后保存为 512X512 的patch，overlap=32

class GlassDiscDataSet(Dataset):
    def __init__(self, txt_file, transform=None):
        super(GlassDiscDataSet, self).__init__()
        self.image_list = read_image_from_txt(txt_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # img = cv2.imread(self.image_list[idx])
        img = Image.open(self.image_list[idx])
        # img = img.transpose((2, 0, 1))
        img_q = augmentation(img)
        img_k = augmentation(img)
        return img_q, img_k


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


normalize = tf.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
augmentation = tf.Compose([
    tf.RandomResizedCrop(224, scale=(0.3, 0.6)),
    tf.RandomApply(
        [tf.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
    ),
    tf.RandomGrayscale(p=0.2),
    tf.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    tf.RandomHorizontalFlip(),
    tf.ToTensor(),
    normalize,
])
