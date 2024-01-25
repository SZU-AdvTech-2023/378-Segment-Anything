from segment_anything_hq import sam_model_registry




if __name__ == '__main__':
    model_type = "vit_h"  # "vit_l/vit_b/vit_h/vit_tiny"
    sam_checkpoint = "E:/00_Code/PyCharmProjects/UrbanSceneNet/sam-hq/checkpoints/sam_hq_vit_h.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to('cuda')


