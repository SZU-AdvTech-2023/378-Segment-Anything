import cv2
import torch
import kornia as K
import kornia.feature as KF


class LoFTR(object):
    def __init__(self, device='cuda'):
        self.device = device
        self.model = self._model_build_()

    def _model_build_(self):
        return KF.LoFTR().to(self.device)

    def do_match(self, image0_path, image1_path, mask0_path=None, mask1_path=None, resize=None):
        image0 = K.io.io.load_image(image0_path, K.io.ImageLoadType.RGB32, device=self.device)[None, ...]
        image1 = K.io.io.load_image(image1_path, K.io.ImageLoadType.RGB32, device=self.device)[None, ...]
        mask0, mask1 = None, None
        if mask0_path is not None:
            mask0 = K.io.io.load_image(mask0_path, K.io.ImageLoadType.GRAY32, device=self.device)[None, ...]
            if resize is not None:
                h0, w0 = image0.shape[-2], image0.shape[-1]
                mask0 = K.geometry.resize(mask0, size=(int(h0 * resize), int(w0 * resize)), interpolation='nearest')
        if mask1_path is not None:
            mask1 = K.io.io.load_image(mask1_path, K.io.ImageLoadType.GRAY32, device=self.device)[None, ...]
            if resize is not None:
                h1, w1 = image1.shape[-2], image0.shape[-1]
                mask1 = K.geometry.resize(mask1, size=(int(h1 * resize), int(w1 * resize)), interpolation='nearest')
        if resize is not None:
            h0, w0 = image0.shape[-2], image0.shape[-1]
            h1, w1 = image1.shape[-2], image0.shape[-1]
            image0 = K.geometry.resize(image0, size=(int(h0 * resize), int(w0 * resize)))
            image1 = K.geometry.resize(image1, size=(int(h1 * resize), int(w1 * resize)))
        input_dict = {
            'image0': K.color.rgb_to_grayscale(image0),
            'image1': K.color.rgb_to_grayscale(image1),
            'mask0': mask0,
            'mask1': mask1
        }
        with torch.inference_mode():
            results = self.model(input_dict)
        return results  # [keypoints0, keypoints1, confidence, batch_indexes]


class SIFT(object):
    def __init__(self, max_features=1024):
        self.sift = cv2.SIFT_create(max_features)

    def detect(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self.sift.detectAndCompute(gray, None)
        return key_points, descriptors

    def match(self, des1, des2):
        # FLANN参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        # 初始化FLANN匹配器
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 使用FLANN进行匹配
        matches = flann.knnMatch(des1, des2, k=2)

        # 仅保留好的匹配 - Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches

    def detect_and_match(self, image0_path, image1_path):
        kp0, des0 = self.detect(image0_path)
        kp1, des1 = self.detect(image1_path)
        good_matches = self.match(des0, des1)
        return kp0, kp1, good_matches


if __name__ == '__main__':
    model = LoFTR()
    resize = 0.2
    image0_path = 'F:/SegAndAlign/SourceImages/DJI_20230517115041_0003_V.JPG'
    image1_path = 'F:/SegAndAlign/SourceImages/DJI_20230517115058_0009_V.JPG'
    res = model.do_match(image0_path, image1_path, resize=resize)
    print('This is feature_match.py...')
