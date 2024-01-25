from fastSAM.FastSAM.fastsam import FastSAM, FastSAMPrompt, FastSAMPredictor
import os
from visualizer.seg_map import plot_everything, single_instance

this_dir = os.path.abspath(os.path.curdir)


class FSAM:
    def __init__(self, mode='seg_anything', device='cpu', checkpoint_path=None):
        self.mode = mode
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint = self.__check_point__()
        self.model = FastSAM(self.checkpoint)

    def __check_point__(self):
        if self.checkpoint_path is not None:
            checkpoint = self.checkpoint_path
        else:
            checkpoint = os.path.join(this_dir,
                                      'E:/00_Code/PyCharmProjects/UrbanSceneNet/fastSAM/checkpoints/FastSAM-x.pt')
        return checkpoint

    def get_model(self):
        return self.model

    def do_segment(self, image_path, save_path=None, prompt=None, prompt_class='everything',
                   retina_masks=True, image_size=1024, conf=0.6, iou=0.9):
        # tensor [objects, h, w]
        everything_result = self.model(image_path, retina_masks=retina_masks, imgsz=image_size, conf=conf, iou=iou)
        # everything_result 保存了几乎所有的相关信息
        # TODO DEBUG
        # masks = [everything_result[0]['masks']['masks'][_] for _ in range(len(everything_result[0]))]
        # bboxes = [everything_result[0]['boxes']['boxes'][_] for _ in range(len(everything_result[0]))]
        prompt_process = FastSAMPrompt(image_path, everything_result, device=self.device)
        if prompt_class == 'point':
            seg_map = prompt_process.point_prompt(prompt[0], prompt[1])
        elif prompt_class == 'bbox':
            seg_map = prompt_process.box_prompt(prompt)
        elif prompt_class == 'text':
            seg_map = prompt_process.text_prompt(prompt)
        else:
            seg_map = prompt_process.everything_prompt()
        if save_path is not None:
            prompt_process.plot(annotations=seg_map, output_path=save_path)
        return seg_map


if __name__ == '__main__':
    fast_sam_model = FSAM(device='cuda')
    result = fast_sam_model.do_segment('F:/SegAndAlign/SourceImages/DJI_0034.JPG',
                                       prompt_class='everything',
                                       image_size=2048).cpu().numpy()
    plot_everything(result, save='./fseg/')

    print('This is fsam.py...')
