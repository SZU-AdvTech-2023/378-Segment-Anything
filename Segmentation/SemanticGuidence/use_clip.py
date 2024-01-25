import open_clip as oc
import torch
from PIL import Image
import os
import clip

os.environ['CURL_CA_BUNDLE'] = ''


# need requests=2.27.1

class OClip(object):
    def __init__(self, model_idx: int = 0, device='cuda'):
        self.model_idx = model_idx
        self.device = device
        self.model_list = ['hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
                           'ViT-bigG-14-CLIPA-336',
                           'ViT-H-14-CLIPA-336',
                           'ViT-L-14-CLIPA-336',
                           'ViT-H-14-CLIPA-336']
        self.pretrain = ['hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
                         'datacomp1b',
                         'E:/00_Code/PyCharmProjects/UrbanSceneNet/CLIP/checkpoints/vit_h14_i84_224_336_cl32_gap_datacomp1b.pt',
                         'E:/00_Code/PyCharmProjects/UrbanSceneNet/CLIP/checkpoints/vit_l14_i84_224_336_cl32_gap_datacomp1b.pt',
                         'E:/00_Code/PyCharmProjects/UrbanSceneNet/CLIP/checkpoints/vit_h14_i84_224_336_cl32_gap_laion2b.pt']
        self.model, self.preprocess_train, self.preprocess_val = oc.create_model_and_transforms(
            model_name=self.model_list[model_idx],
            pretrained=self.pretrain[model_idx],
            device=self.device)
        # self.model, self.preprocess_train, self.preprocess_val = oc.create_model_and_transforms(
        #     model_name='ViT-B-32',
        #     pretrained='laion2b_s34b_b79k',
        #     device=self.device)
        self.tokenizer = oc.get_tokenizer(model_name='hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')

    def get_image_feature(self, image_path):
        image = self.preprocess_val(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_text_features(self, text):
        tokens = self.tokenizer(text).to(self.device)
        # tokens = oc.tokenize(text, 32).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_similarity_score_from_source(self, image_path, text):
        image_features = self.get_image_feature(image_path)
        text_features = self.get_text_features(text)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs

    @staticmethod
    def get_similarity_score_from_features(image_features, text_features):
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs


class Clip(object):
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=device)

    def get_image_features(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_text_features(self, text):
        tokens = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    @staticmethod
    def get_similarity_from_features(image_features, text_features):
        similarity = image_features @ text_features.T
        return similarity

    def get_similarity_from_source(self, image_path, text):
        image_features = self.get_image_features(image_path)
        text_features = self.get_text_features(text)
        return self.get_similarity_from_features(image_features, text_features)


if __name__ == '__main__':
    image_path = 'F:/SAM/002_masks/17th_image_mask_crop.jpg'
    text = ["glazing window", "outer wall", "green plant"]
    # Alignment = OClip(model_idx=0)
    # scores = Alignment.get_similarity_score_from_source(image_path, text)
    Alignment = Clip()
    scores = Alignment.get_similarity_from_source(image_path, text)
    print('This is use_clip.py...')
