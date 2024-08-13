import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class VGG(nn.Module):

    def __init__(self, features, output_dim):
        """
        Args:
            - features: convolutional layers of VGG networks
            - output_dim: the quantity of output layer in this classification problem
        """

        super().__init__()

        self.feature = features
        self.avgpool = nn.AdaptiveAvgPool2d(2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            #
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            #
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h


def get_vgg_layers(config, batch_norm):

    layers, in_channels = [], 3

    for c in config:
        assert c == "M" or isinstance(c, int)
        if c == "M":
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)


if __name__ == "main":
    vgg_layers, models = [], {}

    vgg11_config = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    vgg13_config = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ]
    vgg16_config = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ]
    vgg19_config = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ]

    for vgg, key, pre_trained_model in zip(
        [vgg11_config, vgg13_config, vgg16_config, vgg19_config],
        ["VGG" + str(i) for i in [11, 13, 16, 19]],
        [
            vgg11_bn(pretrained=True),
            vgg13_bn(pretrained=True),
            vgg16_bn(pretrained=True),
            vgg19_bn(pretrained=True),
        ],
    ):
        in_features = pre_trained_model.classifier[-1].in_features
        final_fc = nn.Linear(in_features=in_features, out_features=10)
        pre_trained_model.classifier[-1] = final_fc

        vgg_layers.append(get_vgg_layers(vgg, batch_norm=True))
        model = VGG(vgg_layers[-1], 10)
