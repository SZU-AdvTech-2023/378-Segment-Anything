import numpy as np
import kornia.feature as KF
from FeatureDetection import feature
import torch
import torch.nn.functional as func
import cv2


# strategy 01 patch description
def patch_description(image_record_txt, save_name,
                      patch_size=32, output_dims=256,
                      model_label='SIFT'):
    # SIFT, MKD, HardNet8, MyNet, TFeat, SOSNet
    BaseModel = feature.PatchDescriptorModel()
    model = None
    if model_label == 'SIFT':
        model = BaseModel.patch_SIFT(patch_size=patch_size, num_ang_bins=16, num_spatial_bins=8)
    elif model_label == 'MKD':
        model = BaseModel.patch_MKD(patch_size=patch_size, output_dims=output_dims)
    elif model_label == 'HardNet8':
        model = BaseModel.patch_HardNet8()
        patch_size = 32
    elif model_label == 'MyNet':
        model = BaseModel.patch_HyNet()
        patch_size = 32
    elif model_label == 'TFeat':
        model = BaseModel.patch_TFeat()
        patch_size = 32
    elif model_label == 'SOSNet':
        model = BaseModel.patch_SOSNet()
        patch_size = 32
    else:
        model = None
    with open(image_record_txt, 'r') as f:
        images = f.readlines()
    f.close()
    representations = []
    for idx, image_path in enumerate(images):
        with torch.no_grad():
            image = cv2.imread(image_path[:-1])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_tensor = torch.from_numpy(image).float().to('cuda').unsqueeze(0).unsqueeze(0)
            image_resize = func.interpolate(image_tensor, size=(patch_size, patch_size))
            rep = model(image_resize).squeeze()
            representations.append(rep.cpu().numpy())
    rep = np.array(representations)
    np.savetxt(save_name, representations)


def match(feature_record1, image_record1,
          feature_record2, image_record2,
          match_save,
          distance_threshold=0.5, device='cuda'):
    features1 = np.loadtxt(feature_record1)
    features2 = np.loadtxt(feature_record2)
    with open(image_record1, 'r') as f:
        images1 = f.readlines()
    f.close()
    with open(image_record2, 'r') as f:
        images2 = f.readlines()
    f.close()
    features1_tensor = torch.from_numpy(features1).float().to(device)
    features2_tensor = torch.from_numpy(features2).float().to(device)
    distances_, matches_ = KF.match_mnn(features1_tensor, features2_tensor)
    distances = distances_.cpu().numpy()
    matches = matches_.cpu().numpy()
    indices = np.where(distances < distance_threshold)
    matches = matches[indices[0], :]
    pairs = []
    for _ in matches:
        path1 = images1[_[0]].replace("image_expand", "binary_mask")
        path2 = images2[_[1]].replace("image_expand", "binary_mask")
        pairs.append(path1)
        pairs.append(path2)
    with open(match_save, 'w') as f:
        f.writelines(pairs)
    f.close()


def find_best_matches_use_loftr(model, image0_list, image1_list, conf_filter=0.9):
    rows, cols = len(image0_list), len(image1_list)
    record_mat = np.zeros((rows, cols), dtype=np.uint16)
    for row, image0 in enumerate(image0_list):
        for col, image1 in enumerate(image1_list):
            res = model.do_match(image0_path=image0, image1_path=image1)
            keypoints0, keypoints1, conf = (res['keypoints0'].cpu().numpy(), res['keypoints1'].cpu().numpy(),
                                            res['confidence'].cpu().numpy())
            record_mat[row, col] = len(np.where(conf > conf_filter)[0])
    return record_mat


def find_best_matches_with_geometric_check(matcher, list0_features, list1_features):
    # 用于存储最佳匹配结果
    best_matches = []
    # 对于image_list1中的每一幅图像
    for i, (kp1, des1) in enumerate(list0_features):
        max_matches = 0
        best_match_idx = -1
        best_homography = None
        # 与image_list2中的每一幅图像进行匹配
        for j, (kp2, des2) in enumerate(list1_features):
            matches = matcher.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]

            if len(good_matches) > 4:  # 至少需要4个匹配进行几何检查
                # 获取匹配点的坐标
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                # 使用RANSAC算法估计单应性矩阵
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    num_ransac_matches = np.sum(mask)
                    if num_ransac_matches > max_matches:
                        max_matches = num_ransac_matches
                        best_match_idx = j
                        best_homography = H
        # 存储最佳匹配对的索引、匹配数量及单应性矩阵
        if best_match_idx != -1:
            best_matches.append((i, best_match_idx, max_matches))
    return best_matches
