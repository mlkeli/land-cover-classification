import torch
import torch.nn as nn
import torch.nn.functional as F
from hrnet import hrnet18  # 导入HRNet-18模型

class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in pool_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels * (len(pool_sizes) + 1), in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def _make_stage(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [x]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class PSPNetWithHRNet18(nn.Module):
    def __init__(self, num_classes):
        super(PSPNetWithHRNet18, self).__init__()
        self.hrnet = hrnet18(pretrained=True)  # 加载预训练的HRNet-18模型
        self.pyramid_pooling = PSPModule(in_channels=512)  # 假设HRNet-18的输出通道数为512
        self.decode = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.hrnet(x)  # 经过HRNet-18的特征提取部分
        x = self.pyramid_pooling(x)
        x = self.decode(x)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True) 
        return x