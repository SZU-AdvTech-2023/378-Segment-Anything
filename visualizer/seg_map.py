import os.path
import cv2
import numpy as np
from Segmentation.SemanticGuidence import functions


# input [num_objects, h, w]
def single_instance(image, masks, remove_hole=True, expand=200, obj_index=None, save=None, save_all=None):
    num_objs = masks.shape[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    # create folders
    if save_all is not None:
        if not os.path.exists(save_all):
            os.makedirs(save_all)
        binary_mask_save = save_all + 'binary_mask/'
        image_mask_save = save_all + 'image__mask/'
        image_maks_crop_save = save_all + 'image_mask_crop/'
        image_crop_save = save_all + 'image_crop/'
        image_expand_save = save_all + 'image_expand/'
        os.mkdir(binary_mask_save)
        os.mkdir(image_mask_save)
        os.mkdir(image_maks_crop_save)
        os.mkdir(image_crop_save)
        os.mkdir(image_expand_save)
        # generate binary mask
        for i in range(num_objs):
            mask = masks[i]
            if isinstance(mask[0, 0], bool):
                pixels = np.where(mask)
            else:
                pixels = np.where(mask > 0.8)

            binary_mask = np.zeros_like(mask, dtype=np.uint8)
            image_mask = np.zeros_like(image, dtype=np.uint8)
            binary_mask[pixels] = 255
            # remove small hole
            if remove_hole:
                # binary_mask = functions.remove_white_holes(binary_mask)
                binary_mask = functions.fill_small_holes(binary_mask, 1600)
            pixels = np.where(binary_mask > 0.8)
            # define the bbox
            if len(pixels[0]) > 0:
                x_min, x_max = min(pixels[0]), max(pixels[0])
                y_min, y_max = min(pixels[1]), max(pixels[1])
                bbox = [x_min, y_min, x_max, y_max]
                # transfer the mask to image
                image_mask[pixels] = image[pixels]
                # crop mask for bbox
                mask_crop = image_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # crop image for bbox
                image_crop = image[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
                # expand the crop image
                image_expand = image[max(0, bbox[0] - expand):min(h, bbox[2] + expand),
                               max(0, bbox[1] - expand):min(w, bbox[3] + expand)]
                # save images
                # use the same name
                cv2.imwrite(binary_mask_save + '{:05d}.jpg'.format(i), binary_mask)
                cv2.imwrite(image_mask_save + '{:05d}.jpg'.format(i), image_mask)
                cv2.imwrite(image_maks_crop_save + '{:05d}.jpg'.format(i), mask_crop)
                cv2.imwrite(image_crop_save + '{:05d}.jpg'.format(i), image_crop)
                cv2.imwrite(image_expand_save + '{:05d}.jpg'.format(i), image_expand)
    if obj_index is None:
        obj_index = 0
    else:
        assert obj_index < num_objs
    mask = masks[obj_index]
    if isinstance(mask[0, 0], bool):
        pixels = np.where(mask)
    else:
        pixels = np.where(mask > 0.8)
    img = np.zeros_like(mask, dtype=np.uint8)
    img[pixels] = 255
    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)
        cv2.imwrite(save + f'{obj_index}th_mask.jpg', img)
    return img


def plot_everything(masks, save=None, label=None):
    num_objs, h, w = masks.shape
    colors = np.random.rand(num_objs, 3) * 255
    colors = colors.astype(np.uint8)
    img = np.zeros([h, w, 3], dtype=np.uint8)
    for i in range(num_objs):
        if isinstance(masks[i][0, 0], bool):
            pixels = np.where(masks[i])
            img[pixels] = colors[i]
        else:
            pixels = np.where(masks[i] > 0.8)
            img[pixels] = colors[i]
    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)
        if label is not None:
            cv2.imwrite(save + f'{label}_everything_map.jpg', img)
        else:
            cv2.imwrite(save + f'everything_map.jpg', img)
    return img


def plot_match(image_01_file, image_02_file, match_file, save_path, label):
    image_01 = cv2.imread(image_01_file)
    # image_01 = cv2.cvtColor(image_01, cv2.COLOR_BGR2RGB)
    image_02 = cv2.imread(image_02_file)
    # image_02 = cv2.cvtColor(image_02, cv2.COLOR_BGR2RGB)
    with open(match_file, 'r') as f:
        matches = f.readlines()
    f.close()
    num_of_matches = len(matches) // 2
    colors = np.random.rand(num_of_matches, 3) * 255
    colors = colors.astype(np.uint8)
    for i in range(num_of_matches):
        mask1 = cv2.imread(matches[i * 2][:-1], 0)
        mask2 = cv2.imread(matches[i * 2 + 1][:-1], 0)
        pixels1 = np.where(mask1 > 0.5)
        pixels2 = np.where(mask2 > 0.5)
        image_01[pixels1] = colors[i]
        image_02[pixels2] = colors[i]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path + label + '_01.jpg', image_01)
    cv2.imwrite(save_path + label + '_02.jpg', image_02)


def collect_masks(masks, image_file, save_path, label):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image = cv2.imread(image_file)
    h, w, c = image.shape
    binary = np.zeros((h, w), dtype=np.uint8)
    inverse_binary = np.ones((h, w), dtype=np.uint8) * 255
    # mask region equals to 1
    for _ in masks:
        mask = cv2.imread(_, 0)
        index = np.where(mask > 0.5)
        binary[index] = 255
        inverse_binary[index] = 0
    cv2.imwrite(save_path + label + '_glass_region_binary_mask.jpg', binary)
    cv2.imwrite(save_path + label + '_other_region_binary_mask.jpg', inverse_binary)
