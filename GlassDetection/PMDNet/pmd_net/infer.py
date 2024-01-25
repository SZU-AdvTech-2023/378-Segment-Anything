import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import skimage.io
from config import msd_testing_root

from model.pmd import PMD

device_ids = [0]
torch.cuda.set_device(device_ids[0])

args = {
    'scale': 384,
}

img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# detectron
to_test = {'MSD': msd_testing_root}

to_pil = transforms.ToPILImage()


def main():
    net = PMD().cuda(device_ids[0])
    net.load_state_dict(
        torch.load('E:/00_Code/PyCharmProjects/UrbanSceneNet/GlassDetection/PMDNet/checkpoints/pmd.pth'))

    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            file_path = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/DataProcess/original_datas/for_glass_detec/'
            img_list = [file_path + _ for _ in os.listdir(file_path)]
            start = time.time()
            for idx, img_name in enumerate(img_list):
                print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))

                img = Image.open(os.path.join(root, 'image', img_name))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("{} is a gray image.".format(name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                # f_4, f_3, f_2, f_1, e = net(img_var)
                f_4, f_3, f_2, f_1, edge, final = net(img_var)
                # output = f.data.squeeze(0).cpu()
                # edge = e.data.squeeze(0).cpu()
                f_4 = f_4.data.squeeze(0).cpu()
                f_3 = f_3.data.squeeze(0).cpu()
                f_2 = f_2.data.squeeze(0).cpu()
                f_1 = f_1.data.squeeze(0).cpu()
                edge = edge.data.squeeze(0).cpu()
                final = final.data.squeeze(0).cpu()

                f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
                f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
                f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
                f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
                edge = np.array(transforms.Resize((h, w))(to_pil(edge)))
                final = np.array(transforms.Resize((h, w))(to_pil(final)))

                Image.fromarray(final).save(
                    'E:/00_Code/PyCharmProjects/UrbanSceneNet/Results/galss_detec_results/PMD_results/{:03d}.png'.format(
                        idx))

            end = time.time()
            print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()
