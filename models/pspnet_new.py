import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# 加载预训练的ResNet50模型
resnet50 = torchvision.models.resnet50(pretrained=True)

# 修改ResNet50的最后一层，使其适用于PSPNet
resnet50.fc = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)

# 定义PSPNet模型
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(3)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        size = x.size()[2:]
        out1 = F.interpolate(self.conv1(self.avg_pool1(x)), size=size, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.conv2(self.avg_pool2(x)), size=size, mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.conv3(self.avg_pool3(x)), size=size, mode='bilinear', align_corners=True)
        out4 = F.interpolate(self.conv4(self.avg_pool4(x)), size=size, mode='bilinear', align_corners=True)
        return torch.cat([x, out1, out2, out3, out4], dim=1)

class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        self.resnet50 = resnet50
        self.pyramid_pooling = PyramidPooling(2048, 512)  # 使用512作为输入和输出的通道数
        self.final_conv = nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.pyramid_pooling(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)  # 对结果进行上采样
        return x