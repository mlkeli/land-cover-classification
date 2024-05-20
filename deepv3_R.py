import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# 定义带有残差连接的DeepLabv3-R模型
class DeepLabv3R(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabv3R, self).__init__()
        # 加载ResNet-50作为预训练模型
        self.resnet = models.resnet50(pretrained=True)
        
        # 删除最后一层全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # 定义ASPP模块
        self.aspp = ASPP(2048, 256)  # 假设ResNet-50的输出通道数为2048

        # 定义解码器模块
        self.decoder = Decoder(256, num_classes)  # ASPP模块的输出通道数为256

    def forward(self, x):
        # 编码器部分
        x = self.resnet(x)
        
        # ASPP模块
        x = self.aspp(x)
        
        # 解码器部分
        x = self.decoder(x)
        
        return x

# 定义ASPP模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # 定义ASPP的各个分支
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.atrous1(x)
        x3 = self.atrous2(x)
        x4 = self.atrous3(x)
        x5 = F.interpolate(self.pooling(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        return torch.cat((x1, x2, x3, x4, x5), dim=1)

# 定义解码器模块
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)  # 上采样
        x = self.conv1(x)
        x = self.conv2(x)
        return x


