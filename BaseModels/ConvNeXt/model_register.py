from convnext_model import *


def convnext_tiny(num_classes: int = 1000, need_head=False):
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     need_head=need_head)
    return model


def convnext_small(num_classes: int = 1000, need_head=True):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     need_head=need_head)
    return model


def convnext_base(num_classes: int = 1000, need_head=True):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes,
                     need_head=need_head)
    return model


def convnext_large(num_classes: int = 1000, need_head=True):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes,
                     need_head=need_head)
    return model


def convnext_xlarge(num_classes: int = 22000, need_head=True):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes,
                     need_head=need_head)
    return model


x = torch.randn(4, 3, 512, 512).to('cuda')
net = nn.Linear(512, 1024).to('cuda')
x = net(x)
print(x.shape)
