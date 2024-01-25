import numpy as np
from use_clip import Clip
from SAM import sam
from fastSAM import fsam
import cv2
import os
from visualizer.seg_map import single_instance, plot_everything, plot_match, collect_masks
import visualizer.tensor_vis as tv
import features
import FeatureDetection.feature as ff
import functions
import FeatureDetection.feature_match as fm
import visualizer.match_vis as mv


def segmentation(source_image_root, save_root_path, expand=100, label='Geometry', resize=None, model='SAM'):
    ##########
    # SAM
    ##########
    source_images = [source_image_root + _ for _ in os.listdir(source_image_root)]
    if model == 'SAM':
        SAM = sam.SAM(device='cuda')
        for idx, img in enumerate(source_images):
            # load source images
            image = cv2.imread(img)
            # change the color space
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if resize is not None:
                image = cv2.resize(image, dsize=None, fx=resize, fy=resize)
            masks, bboxes, areas = SAM.do_segment(image, label=label)
            # 'binary_mask/' 'image_mask/' 'image_mask_crop/' 'image_crop/' 'image_expand/' for each image
            single_instance(image, masks, expand=expand, save_all=save_root_path + 'image_{:04d}/'.format(idx))
            # everything seg map for each image
            plot_everything(masks, save=save_root_path + 'seg_map/', label='image_{:04d}'.format(idx))
            print("Image num {:04d} has segmented...".format(idx + 1))
    else:
        SAM = fsam.FSAM(device='cuda')
        for idx, img in enumerate(source_images):
            # load source images
            image = cv2.imread(img)
            # change the color space
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = SAM.do_segment(img, prompt_class='everything', image_size=2048).cpu().numpy()
            # 'binary_mask/' 'image_mask/' 'image_mask_crop/' 'image_crop/' 'image_expand/' for each image
            single_instance(image, masks, expand=expand, save_all=save_root_path + 'image_{:04d}/'.format(idx))
            # everything seg map for each image
            plot_everything(masks, save=save_root_path + 'seg_map/', label='image_{:04d}'.format(idx))
            print("Image num {:04d} has segmented...".format(idx + 1))


def alignment(text, image_path, name_save, indices_save):
    ##########
    # CLIP
    ##########
    clip = Clip()
    text_features = clip.get_text_features(text)
    images = [image_path + _ for _ in os.listdir(image_path)]
    record = []
    indices = []
    for idx, image in enumerate(images):
        h, w, c = cv2.imread(image).shape
        scalar = float(h) / float(w)
        image_features = clip.get_image_features(image)
        image_score = clip.get_similarity_from_features(image_features, text_features).cpu().numpy()
        criterion = np.argmax(image_score)
        if (criterion < 5) and (scalar < 2. and scalar > 0.5):
            record.append(image + '\n')
            indices.append(int(idx))
    with open(name_save, 'w') as f:
        f.writelines(record)
    f.close()
    np.savetxt(indices_save, indices)


def vis_result(image, indices_record, mask_path, resize=None):
    if resize is not None:
        image = cv2.resize(image, None, fx=resize, fy=resize)
    indices = np.loadtxt(indices_record)
    masks = [mask_path + _ for _ in os.listdir(mask_path)]
    label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for _ in indices:
        mask = cv2.imread(masks[int(_)], 0)
        pixels = np.where(mask > 0.9)
        label[pixels] = 255
    labels = np.where(label < 1)
    image[labels] = [0, 0, 0]
    return image


def loftr_point_match(images_name_list, region_label,
                      save_root_path, match_save_path, txt_save_path,
                      resize=0.3, confi_filter=0.8, draw_each_pair=False):
    model = fm.LoFTR()
    num_of_images = len(images_name_list)
    for _ in range(num_of_images - 1):
        image0_path = images_name_list[_]
        image1_path = images_name_list[_ + 1]
        image0 = cv2.imread(image0_path)
        image1 = cv2.imread(image1_path)
        # TODO mask do not work
        mask0_path = save_root_path + 'seg_map/image_{:04d}_{:s}_binary_mask.jpg'.format(_, region_label)
        mask1_path = save_root_path + 'seg_map/image_{:04d}_{:s}_binary_mask.jpg'.format(_ + 1, region_label)
        image0 = cv2.resize(image0, dsize=None, fx=resize, fy=resize)
        image1 = cv2.resize(image1, dsize=None, fx=resize, fy=resize)
        res = model.do_match(image0_path, image1_path, mask0_path=None, mask1_path=None, resize=resize)
        keypoints0, keypoints1, conf = (res['keypoints0'].cpu().numpy(),
                                        res['keypoints1'].cpu().numpy(),
                                        res['confidence'].cpu().numpy())
        filter_index = np.where(conf > confi_filter)
        keypoints0, keypoints1 = keypoints0[filter_index], keypoints1[filter_index]
        conf = conf[filter_index]

        mask0 = cv2.imread(mask0_path, 0)
        mask1 = cv2.imread(mask1_path, 0)
        mask0 = cv2.resize(mask0, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
        mask1 = cv2.resize(mask1, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
        # 确保特征点坐标是整数，因为它们将被用作数组索引
        kp0 = keypoints0.astype(int)
        kp1 = keypoints1.astype(int)
        # 根据掩码剔除特征点对
        valid_kp0_mask = mask0[kp0[:, 1], kp0[:, 0]] > 100  # 取出kp1对应的mask1位置的值，检查是否为1
        valid_kp1_mask = mask1[kp1[:, 1], kp1[:, 0]] > 100  # 同理对kp2和mask2进行操作
        valid_mask = np.logical_and(valid_kp0_mask, valid_kp1_mask)  # 两个掩码取逻辑与操作，得到有效特征点对的掩码
        # 应用掩码过滤无效的特征点对  int for draw
        filtered_kp0 = kp0[valid_mask]
        filtered_kp1 = kp1[valid_mask]

        # float for record
        valid_keypoints0 = keypoints0[valid_mask]
        valid_keypoints1 = keypoints1[valid_mask]
        valid_conf = conf[valid_mask][..., None]
        record = np.concatenate([valid_keypoints0, valid_keypoints1, valid_conf], axis=1)
        np.savetxt(txt_save_path + '{:s}_image{:04d}_image{:04d}_match_points.txt'.format(region_label, _, _ + 1),
                   record)

        # point match visualize
        # filtered_kp1, filtered_kp2 和 filtered_confidence 现在包含了剔除了无效区域后的匹配特征点对和置信度
        mv.draw_matches(image0, image1, filtered_kp0, filtered_kp1,
                        point_radius=5, line_thickness=1,
                        save_path=match_save_path + 'all_matches/',
                        label='{:s}_image{:04d}_image{:04d}'.format(region_label, _, _ + 1))
        assert len(filtered_kp0) == len(filtered_kp1)
        # draw per pair
        if draw_each_pair:
            for i in range(len(filtered_kp1)):
                mv.draw_matches(image0, image1, [filtered_kp0[i]], [filtered_kp1[i]],
                                point_radius=8, line_thickness=3,
                                save_path=match_save_path + '{:s}_image{:04d}_image{:04d}_LoFTR/'.format(region_label,
                                                                                                         _, _ + 1),
                                label='image{:04d}_image{:04d}_{:04d}'.format(_, _ + 1, i))


def main(mask_expand=100, resize=None):
    ################
    # file system
    ################
    # source images folder
    root_path = 'F:/SegAndAlign/MVS/'
    source_image_root = root_path+'source_images/'
    source_images = [source_image_root + _ for _ in os.listdir(source_image_root)]
    num_of_images = len(os.listdir(source_image_root))
    print("Totally {:03d} images...".format(num_of_images))
    # path to save seg results
    save_root_path = root_path+'expand_{:04d}/'.format(mask_expand)
    # path to save txt files
    txt_save_path = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/Results/Matching/MVS/expand{:04d}_txt/'.format(mask_expand)
    # path to save match vis
    match_save_path = root_path+'expand_{:04d}/match_vis/'.format(mask_expand)
    region_label = 'other_region'  # 'glass_region' or 'other_region'

    ################
    # SAM or fastSAM
    ################
    segmentation(source_image_root, save_root_path, model='fSAM', expand=mask_expand, resize=resize)

    ################
    # CLIP
    ################
    # glass window, glass door
    text = ["glass", "window", "glass window", "glass door",
            "wall", "plants", "ground", "trees",
            "roof", "car", "pedestrian", "sky",
            "floor", "building"]
    for i in range(num_of_images):
        # use expand image to detect if there have glass
        image_path = save_root_path + 'image_{:04d}/image_expand/'.format(i)
        if not os.path.exists(txt_save_path):
            os.makedirs(txt_save_path)
        alignment(text, image_path,
                  txt_save_path + 'image_{:04d}_glass_detect_record.txt'.format(i),
                  txt_save_path + 'indices_image_{:04d}_glass_detect_record.txt'.format(i))
        images = [source_image_root + _ for _ in os.listdir(source_image_root)]
        source_img = cv2.imread(images[i])
        # 将 glass region 映射到原图中
        image = vis_result(source_img,
                           txt_save_path + 'indices_image_{:04d}_glass_detect_record.txt'.format(i),
                           save_root_path + '/image_{:04d}/binary_mask/'.format(i),
                           resize=resize)
        cv2.imwrite(save_root_path + 'seg_map/show_{:04d}.jpg'.format(i), image)

    ################
    # Patch-level Match
    ################
    # 使用 Patch 的特征进行 Patch 之间的匹配
    # after testing, SIFT have the best performance, but still not well
    for i in range(num_of_images):
        features.patch_description(txt_save_path + 'image_{:04d}_glass_detect_record.txt'.format(i),
                                   txt_save_path + 'image_{:04d}_features_record.txt'.format(i),
                                   model_label='SIFT', output_dims=512, patch_size=41)
    # patch match step
    for i in range(num_of_images - 1):
        features.match(feature_record1=txt_save_path + 'image_{:04d}_features_record.txt'.format(i),
                       feature_record2=txt_save_path + 'image_{:04d}_features_record.txt'.format(i + 1),
                       image_record1=txt_save_path + 'image_{:04d}_glass_detect_record.txt'.format(i),
                       image_record2=txt_save_path + 'image_{:04d}_glass_detect_record.txt'.format(i + 1),
                       match_save=txt_save_path + 'match_image{:04d}_to_image{:04d}.txt'.format(i, i + 1),
                       distance_threshold=0.6)
    for i in range(num_of_images - 1):
        match_file = txt_save_path + 'match_image{:04d}_to_image{:04d}.txt'.format(i, i + 1)
        plot_match(image_01_file=source_images[i], image_02_file=source_images[i + 1],
                   match_file=match_file, label='match_image{:04d}_to_image{:04d}'.format(i, i + 1),
                   save_path=save_root_path + 'match_map/')

    # ################
    # # Point-level Match
    # ################
    # # # 使用点的特征进行 Patch 之间的匹配
    # # # strategy 01 SIFT detector and descriptor
    # # sift = ff.SIFT(max_features=512)
    # # matcher = cv2.BFMatcher()
    # # images_fp_and_dcs = []  # [[kp, dcs]...]
    # image_path_lists = []
    # for i in range(num_of_images):
    #     glass_record_txt = txt_save_path + 'image_{:04d}_glass_detect_record.txt'.format(i)
    #     with open(glass_record_txt, 'r') as f:
    #         image_path_list = f.readlines()
    #     image_path_lists.append([_[:-1] for _ in image_path_list])
    # #     results = [sift.detect(_) for _ in image_path_list]
    # #     images_fp_and_dcs.append(results)
    # # matches = []
    # # for i in range(num_of_images - 1):
    # #     matches_index = features.find_best_matches_with_geometric_check(matcher,
    # #                                                                     images_fp_and_dcs[i],
    # #                                                                     images_fp_and_dcs[i + 1])
    # #     matches.append(matches_index)
    # # # strategy 02 LoFTR detector and descriptor
    # LoFTR = fm.LoFTR()
    # mat = features.find_best_matches_use_loftr(LoFTR, image_path_lists[0], image_path_lists[1])
    # tv.visualize_array_heatmap(mat, save_path=match_save_path + 'match_heat_map/', label='image0-image1')
    # print('DEBUG...')

    ################
    # Other test
    ################
    # 将 txt 中记录的 mask 集中展示在同一幅图中，生成玻璃区域或非玻璃区域的二值 mask
    for idx, _ in enumerate(source_images):
        masks = functions.txt_to_list(txt_file=txt_save_path + 'image_{:04d}_glass_detect_record.txt'.format(idx))
        collect_masks(masks=masks, image_file=_, label='image_{:04d}'.format(idx),
                      save_path=save_root_path + 'seg_map/')

    ################
    # Alignment-based
    ################
    # 使用 LoFTR 检测规定区域的特征点匹配情况
    loftr_point_match(source_images, resize=0.3, region_label=region_label,
                      save_root_path=save_root_path,
                      txt_save_path=txt_save_path,
                      match_save_path=match_save_path,
                      draw_each_pair=True)

    # # 对平面进行仿射变换，这样是不对的，因为要求齐次矩阵需要知道对应像素点的深度
    # # 那么已知内外参和对应点后是否可以先估计深度
    # matrix = []
    # for i in range(num_of_images - 1):
    #     keypoints_txt = txt_save_path + '{:s}_image{:04d}_image{:04d}'.format(region_label, i,
    #                                                                           i + 1) + '_match_points.txt'
    #     m = functions.cal_warp_mat(keypoints_txt, confidence_filter=0.9)
    #     matrix.append(m)
    #     np.savetxt(txt_save_path + '{:s}_image{:04d}_image{:04d}'.format(region_label, i, i + 1) + '_warp_matrix.txt',
    #                m)
    #     image = cv2.imread(source_images[i + 1])
    #     h, w = image.shape[:2]
    #     align_image = cv2.warpPerspective(image, m, (w, h))
    #     cv2.imwrite(save_root_path + 'seg_map/image{:4d}_warp_use_{:s}.jpg'.format(i + 1, region_label), align_image)


if __name__ == '__main__':
    main(mask_expand=100)
    print('This is alignment.py...')
