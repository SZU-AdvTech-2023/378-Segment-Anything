import torch
from mmpretrain import get_model
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
import torch.nn as nn
from PIL import Image
import platform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2

system_type = platform.system()
# 'Windows' 'Linux'
if system_type == 'Windows':
    from data.data_process import *

if system_type == 'Linux':
    import sys

    sys.path.append('/home/ubuntu/workdic/UrbanSceneNet/urban-scene-seg-net/DataProcess')
    from data_process import *

normalize = tf.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
inference_transform = tf.Compose([
    tf.Resize((224, 224)),
    tf.ToTensor(),
    normalize,
])


class DINOInferneceDatset(Dataset):
    def __init__(self, txt_file, transform=inference_transform):
        super(DINOInferneceDatset, self).__init__()
        self.image_list = read_image_from_txt(txt_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # img = cv2.imread(self.image_list[idx])
        img = Image.open(self.image_list[idx])
        img = self.transform(img)
        return img


def inference_use_dino(device='cuda'):
    dataset = DINOInferneceDatset(
        txt_file='/DataProcess/txt/UrbanScenePatch/eval_image_win.txt')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    representation = []
    model = get_model('vit-small-p14_dinov2-pre_3rdparty', pretrained=True, device=device)
    model.eval()
    for idx, img in enumerate(dataloader):
        with torch.no_grad():
            print('Inference image--{:06d}.....'.format(idx))
            img = img.to(device)
            img_rep = model(img)[0].T.squeeze()
            img_rep = nn.functional.normalize(img_rep, dim=0)
            np_arr = img_rep.cpu().numpy()
            representation.append(np_arr)

    reps = np.array(representation)
    np.savetxt('results/representations_for_eval.txt', reps)


def kmeans_cluster(representation_txt, n_cluster=5, max_iter=300):
    # 生成示例数据集,包含100个样本,每个样本是一个300维向量
    X = np.loadtxt(representation_txt)
    # 定义k-means聚类对象,k为聚类中心数
    km = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=max_iter, n_init=10)
    # 进行聚类
    km.fit(X)
    # 获得每个样本的聚类标签
    labels = km.labels_
    # 获得聚类中心
    centers = km.cluster_centers_

    # 循环每个簇
    for i in range(len(np.unique(labels))):
        # 获取该簇的所有样本索引
        indices = np.where(labels == i)[0]
        np.savetxt('./results/eval_cluster_label_{:02d}_ncluster{:02d}.txt'.format(i, n_cluster), indices, fmt='%d')
        # 写入txt文件
    # # 颜色映射,为每一个簇指定一个颜色
    # colors = ['red', 'green', 'blue', 'cyan', 'magenta']
    # tsne = TSNE(n_components=2).fit_transform(X)
    # # 绘制每个样本点
    # for i in range(len(labels)):
    #     plt.plot(X[i, 0], X[i, 1], '.', color=colors[labels[i]])
    #     # 记录样本索引
    #     # plt.text(X[i, 0], X[i, 1], str(i))
    # # 绘制簇中心
    # for i, c in enumerate(centers):
    #     plt.plot(c[0], c[1], 'o', color=colors[i], markersize=12)
    # plt.show()


def index_to_image(index_txt, image_txt, write_name):
    images = read_image_from_txt(image_txt)
    indices = np.loadtxt(index_txt, dtype=np.int64)
    cluster_images = [images[_] for _ in indices]
    with open(write_name, 'w') as f:
        f.writelines('\n'.join(cluster_images))
    f.close()


def image_rewrite(image_txt, write_path, resize=224):
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    images = read_image_from_txt(image_txt)
    for idx, _ in enumerate(images):
        image = cv2.imread(_)
        image = cv2.resize(image, dsize=(resize, resize))
        cv2.imwrite(write_path + '_{:05d}.jpg'.format(idx), image)


def main(image_txt, cluster=6):
    kmeans_cluster('./results/representations_for_eval.txt',
                   n_cluster=cluster)
    for _ in range(cluster):
        index_to_image('./results/eval_cluster_label_{:02d}_ncluster{:02d}.txt'.format(_, cluster),
                       image_txt, write_name='./results/cluster{:02d}_ncluster{:02d}_images.txt'.format(_, cluster))
    txt_file = ['./results/cluster{:02d}_ncluster{:02d}_images.txt'.format(_, cluster) for _ in range(cluster)]
    for idx, _ in enumerate(txt_file):
        image_rewrite(_, 'F:/cluster/cls_{:02d}/cluster_{:02d}/'.format(cluster, idx))


if __name__ == '__main__':
    main('E:/00_Code/PyCharmProjects/UrbanSceneNet/DataProcess/txt/UrbanScenePatch/eval_image_win.txt', cluster=6)
