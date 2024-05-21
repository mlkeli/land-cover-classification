import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(FCN, self).__init__()
        # 编码器（通常使用预训练的卷积神经网络）
        self.encoder = nn.Sequential(
            # 卷积层1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸：原始尺寸的一半
            # 卷积层2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸：原始尺寸的1/4
            # 卷积层3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸：原始尺寸的1/8
        )
        # 解码器
        self.decoder = nn.Sequential(
            # 上采样
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)  # 输出尺寸：原始尺寸的1/8
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
