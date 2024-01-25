from BaseModels.ResNet.resnet_model import *


def resnet34(num_classes=1000, include_top=True, multi_features=False):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  multi_features=multi_features)


def resnet50(num_classes=1000, include_top=True, multi_features=False):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  multi_features=multi_features)


def resnet101(num_classes=1000, include_top=True, multi_features=False):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  multi_features=multi_features)


def resnext50_32x4d(num_classes=1000, include_top=True, multi_features=False):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group,
                  multi_features=multi_features)


def resnext101_32x8d(num_classes=1000, include_top=True, multi_features=False):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group,
                  multi_features=multi_features)
