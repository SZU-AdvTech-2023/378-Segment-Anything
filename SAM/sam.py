from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import numpy as np
from visualizer.seg_map import single_instance, plot_everything

this_dir = os.path.abspath(os.path.curdir)


class SAM:
    def __init__(self, mode='seg_anything', device='cpu', model_type='vit_h', checkpoint_path=None):
        self.mode = mode
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.checkpoint = self.__check_point__(self.checkpoint_path)
        self.sam = sam_model_registry[self.model_type](self.checkpoint).to(self.device)

    def __check_point__(self, path=None):
        if path is not None:
            checkpoint = path
        else:
            if self.model_type == 'vit_b':
                return os.path.join(this_dir,
                                    'E:/00_Code/PyCharmProjects/UrbanSceneNet/SAM/checkpoints/sam_vit_b_01ec64.pth')
            if self.model_type == 'vit_h':
                return os.path.join(this_dir,
                                    'E:/00_Code/PyCharmProjects/UrbanSceneNet/SAM/checkpoints/sam_vit_h_4b8939.pth')
            else:
                return os.path.join(this_dir,
                                    'E:/00_Code/PyCharmProjects/UrbanSceneNet/SAM/checkpoints/sam_vit_l_0b3195.pth')
        return checkpoint

    def get_model(self):
        return self.sam

    def do_segment(self, image, label='Everything'):
        """Return
            segmentation (dict(str, any) or np.ndarray): The mask. If
            output_mode='binary_mask', is an array of shape HW. Otherwise,
            is a dictionary containing the RLE.
            bbox (list(float)): The box around the mask, in XYWH format.
            area (int): The area in pixels of the mask.
        dict_keys(['segmentation','area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
        """
        mask_generator = SamAutomaticMaskGenerator(self.sam, min_mask_region_area=0, points_per_side=128)
        seg_result = mask_generator.generate(image)
        if label == 'Everything' or label == 0:
            return seg_result
        if label == 'Geometry' or label == 1:
            return self._get_masks_(seg_result), self._get_bbox_(seg_result), self._get_areas_(seg_result)
        if label == 'OnlyMask' or label == 2:
            return self._get_masks_(seg_result)
        if label == 'OnlyBbox' or label == 3:
            return self._get_bbox_(seg_result)
        else:
            return seg_result

    @staticmethod
    def _get_masks_(seg_result):
        # all segmentation masks [number_masks, h, w]
        masks = [seg_result[_]['segmentation'] for _ in range(len(seg_result))]
        return np.array(masks)

    @staticmethod
    def _get_areas_(seg_result):
        # number of pixels
        areas = [seg_result[_]['area'] for _ in range(len(seg_result))]
        return np.array(areas)

    @staticmethod
    def _get_bbox_(seg_result):
        # [x, y, H, W]
        return [seg_result[_]['bbox'] for _ in range(len(seg_result))]


if __name__ == '__main__':
    # root_path = 'F:/5buildings/04/'
    # file_path = root_path + 'for_sam_08/'
    # image_resize = 0.5
    # img_list = [file_path + _ for _ in os.listdir(file_path)]
    # SAM = SAM(device='cuda')
    # for idx, _ in enumerate(img_list):
    #     image = cv2.imread(_)
    #     image = cv2.resize(image, dsize=None, fx=image_resize, fy=image_resize)
    #     # seg_map = SAM.seg_anything(image)
    #     seg_only_masks = SAM.only_masks(image)
    #     plot_everything(seg_only_masks, label=idx,
    #                     save=root_path + 'for_sam_08/')

    # test_image = 'F:/SAM/002.png'
    # SAM = SAM(device='cuda')
    # image = cv2.imread(test_image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    # masks, bboxes, areas = SAM.do_segment(image, label='Geometry')
    # single_instance(image, masks, bboxes, save_all='F:/SAM/002_masks/')
    # plot_everything(masks, save='F:/SAM/', label='002')
    print('This is sam.py...')
