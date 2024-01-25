import os.path

import cv2
import numpy as np
import torch
import kornia.io as kio
import kornia.feature as KF
from kornia.geometry.subpix.spatial_soft_argmax import ConvSoftArgmax3d
from kornia.utils.draw import draw_line


# usually, detector return key points and describe vector
# LAF return key points and confidence
class Response(object):
    def __init__(self, model='gftt', grads_mode='sobel', sigmas=None):
        assert model in ['gtff', 'dog', 'harris', 'hessian']
        self.model = model
        self.grads_mode = grads_mode
        self.sigmas = sigmas

    def _gftt_response_(self, input: torch.Tensor):
        return KF.gftt_response(input, self.grads_mode, self.sigmas)

    def _harris_response_(self, input: torch.Tensor, k=0.04):
        return KF.harris_response(input, k=k, grads_mode=self.grads_mode, sigmas=self.sigmas)

    def _hessian_response_(self, input: torch.Tensor):
        return KF.hessian_response(input, grads_mode=self.grads_mode, sigmas=self.sigmas)

    def _dog_response_single_(self, input: torch.Tensor, sigma1=1.0, sigma2=1.6):
        return KF.dog_response_single(input, sigma1=sigma1, sigma2=sigma2)

    def Response(self, k=0.04, sigma1=1.0, sigma2=1.6):
        if self.model == 'gftt':
            return KF.CornerGFTT(grads_mode=self.grads_mode)
        elif self.model == 'harris':
            return KF.CornerHarris(grads_mode=self.grads_mode, k=k)
        elif self.model == 'hessian':
            return KF.BlobHessian(grads_mode=self.grads_mode)
        elif self.model == 'dog':
            return KF.BlobDoGSingle(sigma1=sigma1, sigma2=sigma2)
        else:
            return -1

    def get_response_map(self, image_path, gray=True, k=0.04, sigma1=1.0, sigma2=1.6):
        if gray:
            image = kio.load_image(image_path, kio.ImageLoadType.GRAY8)
        else:
            image = kio.load_image(image_path, kio.ImageLoadType.RGB8)
        if self.model == 'gftt':
            return self._gftt_response_(image)
        elif self.model == 'harris':
            return self._harris_response_(image, k)
        elif self.model == 'hessian':
            return self._hessian_response_(image)
        elif self.model == 'dog':
            return self._dog_response_single_(image, sigma1, sigma2)
        else:
            return None


class PatchDescriptorModel(object):
    def __init__(self, device='cuda'):
        self.device = device

    def dense_SIFT(self, num_ang_bins=12, num_spatial_bins=4, spatial_bin_size=8,
                   rootsift=True, clipval=0.2, stride=1, padding=1):
        return KF.DenseSIFTDescriptor(num_ang_bins=num_ang_bins,
                                      num_spatial_bins=num_spatial_bins,
                                      spatial_bin_size=spatial_bin_size,
                                      rootsift=rootsift,
                                      clipval=clipval,
                                      stride=stride, padding=padding).to(self.device)

    def patch_SIFT(self, patch_size=41, num_ang_bins=12, num_spatial_bins=8,
                   rootsift=True, clipval=0.2):
        return KF.SIFTDescriptor(patch_size=patch_size,
                                 num_spatial_bins=num_spatial_bins,
                                 num_ang_bins=num_ang_bins,
                                 rootsift=rootsift, clipval=clipval).to(self.device)

    def patch_MKD(self, patch_size=32, output_dims=256,
                  kernel_type='concat', whitening='pcawt', training_set='liberty'):
        return KF.MKDDescriptor(patch_size=patch_size,
                                output_dims=output_dims,
                                kernel_type=kernel_type,
                                whitening=whitening,
                                training_set=training_set).to(self.device)

    def patch_HardNet8(self, pretrained=True):
        return KF.HardNet8(pretrained=pretrained).to(self.device)

    def patch_HyNet(self, pretrained=True, is_bias=True, is_bias_FRN=True,
                    dim_desc=128, drop_rate=0.3, eps_l2_norm=1e-10):
        return KF.HyNet(pretrained=pretrained,
                        is_bias=is_bias,
                        is_bias_FRN=is_bias_FRN,
                        dim_desc=dim_desc,
                        drop_rate=drop_rate,
                        eps_l2_norm=eps_l2_norm).to(self.device)

    def patch_TFeat(self, pretrained=True):
        return KF.TFeat(pretrained=pretrained).to(self.device)

    def patch_SOSNet(self, pretrained=True):
        return KF.SOSNet(pretrained=pretrained).to(self.device)


class LAF(object):
    def __init__(self, device='cuda'):
        self.device = device

    def multi_resolution_detector(self, model, num_features=1024,
                                  ori_module=KF.LAFOrienter,
                                  aff_module=KF.LAFAffineShapeEstimator):
        return KF.MultiResolutionDetector(model=model,
                                          num_features=num_features,
                                          ori_module=ori_module(),
                                          aff_module=aff_module()).to(self.device)

    def scale_space_detector(self, num_features=512, mr_size=6.0,
                             scale_pyr_module=KF.scale_space_detector.ScalePyramid(3, 1.6, 15),
                             resp_module=KF.BlobHessian(),
                             nms_module=ConvSoftArgmax3d((3, 3, 3),
                                                         (1, 1, 1),
                                                         (1, 1, 1),
                                                         normalized_coordinates=False,
                                                         output_value=True),

                             ori_module=KF.LAFOrienter,
                             aff_module=KF.LAFAffineShapeEstimator,
                             minima_are_also_good=False,
                             scale_space_response=False):
        return KF.ScaleSpaceDetector(num_features=num_features,
                                     mr_size=mr_size,
                                     scale_space_response=scale_space_response,
                                     resp_module=resp_module,
                                     nms_module=nms_module,
                                     ori_module=ori_module(),
                                     aff_module=aff_module(),
                                     minima_are_also_good=minima_are_also_good,
                                     scale_pyr_module=scale_pyr_module).to(self.device)

    def key_net_detector(self, pretrained=True, num_features=1024,
                         ori_module=KF.LAFOrienter,
                         aff_module=KF.LAFAffineShapeEstimator):
        return KF.KeyNetDetector(pretrained=pretrained,
                                 num_features=num_features,
                                 ori_module=ori_module(),
                                 aff_module=aff_module()).to(self.device)


class Matcher(object):
    def __init__(self, device='cuda'):
        self.device = device

    def _simple_matcher_(self, match_mode='nn'):
        return KF.DescriptorMatcher(match_mode=match_mode).to(self.device)


class SIFT(object):
    def __init__(self, max_features=1024):
        self.sift = cv2.SIFT_create(max_features)

    def detect(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self.sift.detectAndCompute(gray, None)
        return key_points, descriptors


class DetectorNet(object):
    def __init__(self, label=0, num_of_features=1024, upright=False, device='cuda'):
        assert label < 3
        self.num_of_features = num_of_features
        self.upright = upright
        self.device = device
        self.label = label

    def _GFTTAffNetHardNet_detec_and_describe_(self):
        return KF.GFTTAffNetHardNet(num_features=self.num_of_features,
                                    upright=self.upright,
                                    device=self.device)

    def _KeyNetAffNetHardNet_(self):
        return KF.KeyNetAffNetHardNet(num_features=self.num_of_features,
                                      upright=self.upright,
                                      device=self.device)

    def _KeyNetHardNet_(self):
        return KF.KeyNetHardNet(num_features=self.num_of_features,
                                upright=self.upright,
                                device=self.device)

    def detect(self):
        if self.label == 0:
            model = self._GFTTAffNetHardNet_detec_and_describe_()
        elif self.label == 1:
            model = self._KeyNetAffNetHardNet_()
        elif self.label == 2:
            model = self._KeyNetHardNet_()
        else:
            return None
        return model


class SOLD2LineDetect(object):
    """
    # >>> img = torch.rand(1, 1, 512, 512)
    # >>> sold2_detector = SOLD2_detector()
    # >>> line_segments = sold2_detector(img)["line_segments"]
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.model = self._sold2_model_()

    def _sold2_model_(self):
        return KF.SOLD2_detector().to(self.device)

    def line_detect(self, image_path):
        with torch.inference_mode():
            image0 = kio.io.load_image(image_path, kio.ImageLoadType.GRAY32, device=self.device)[None, ...]
            result = self.model(image0)
        return result


class LSDLineDetect(object):
    def __init__(self, scale=1.0, sigma_scale=0.6, ang_th=22.5, density_th=0.7):
        self.model = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, scale=scale, sigma_scale=sigma_scale,
                                                   ang_th=ang_th, density_th=density_th)

    def detect(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        lines, width, conf, nfa = self.model.detect(gray)
        return lines, width, conf, nfa

    def draw_lines(self, image, lines, label=None, save_path=None, show=False):
        draw_image = self.model.drawSegments(image, lines)
        if show:
            cv2.namedWindow('Line Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Line Detection', draw_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path + label + '_line_seg.jpg', draw_image)


if __name__ == '__main__':
    image_path = 'F:/SegAndAlign/MVS/source_images/DJI_20230516152238_0012_V.JPG'
    image = cv2.imread(image_path)
    black_back = np.zeros_like(image)
    line_detector = LSDLineDetect()
    lines, width, conf, nfa = line_detector.detect(image)
    # line filter [length] [width]
    length_ = (lines[:, 0, 3] - lines[:, 0, 1]) ** 2 + (lines[:, 0, 2] - lines[:, 0, 0]) ** 2
    index = np.intersect1d(np.where(length_ > 900), np.where(width < 10))
    lines = lines[index]
    np.savetxt('F:/SegAndAlign/MVS/line_detect/DJI_20230516152238_0012_V.txt', lines.squeeze())
    line_detector.draw_lines(black_back, lines, show=False, save_path='F:/SegAndAlign/MVS/line_detect/',
                             label='DJI_20230516152238_0012_V_BLACK')
    line_detector.draw_lines(image, lines, show=False, save_path='F:/SegAndAlign/MVS/line_detect/',
                             label='DJI_20230516152238_0012_V')

    lines = np.loadtxt('F:/SegAndAlign/MVS/source_images/out.txt')
    if lines.dtype != np.float32:
        lines = lines.astype(np.float32)
    line_detector.draw_lines(black_back, lines, show=True)

    print('This is feature.py...')
