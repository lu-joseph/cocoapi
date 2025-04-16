import numpy as np
import torch.nn as nn
import torch


class AlexNet_GAP(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet_GAP, self).__init__()
        self.features = nn.Sequential(
            # L1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding="valid"),
            nn.ReLU(inplace=True),
            # L1_MP
            nn.MaxPool2d(kernel_size=3, stride=2),
            # L2
            nn.Conv2d(96, 256, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            # L2_MP
            nn.MaxPool2d(kernel_size=3, stride=2),
            # L3
            nn.Conv2d(256, 384, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            # L4
            nn.Conv2d(384, 384, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            # L5
            nn.Conv2d(384, 256, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            # Extra Layer
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        self.feature_maps = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
