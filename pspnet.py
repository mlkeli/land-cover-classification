import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        resnet = models.resnet50()
        # resnet.load_state_dict((torch.load("../input/resnet/resnet50/resnet50-19c8e357.pth")))
        resnet.load_state_dict((torch.load(r"D:\算法汇总\遥感\地物分类\resnet50\resnet50-19c8e357.pth")))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.pyramid_pooling = PSPModule(in_channels=2048)

        self.decode = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pyramid_pooling(x)
        x = self.decode(x)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True) 
        return x
