import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50(nn.Module):
    """ ResNet50 Multi-Scale Hough Transform """
    def __init__(self, args):
        super(ResNet50, self).__init__()
        pretrained = args.pretrained
        train_backbone = args.train_backbone

        resnet = models.resnet50(pretrained=pretrained)
        resnet = nn.Sequential(*list(resnet.children())[:-2])

        if not train_backbone:
            for name, parameter in resnet.named_parameters():
                parameter.requires_grad_(False)

        # self.feature2 = nn.Sequential(*list(resnet.children())[:5])
        self.feature3 = nn.Sequential(*list(resnet.children())[:6])
        self.feature4 = nn.Sequential(*list(resnet.children())[6:7])
        self.feature5 = nn.Sequential(*list(resnet.children())[7:8])

        # self.num_channels = [256, 512, 1024]
        # self.scale_factor = [4, 8, 16]

        self.num_channels = [512, 1024, 2048]
        self.scale_factor = [8, 16, 32]

    def forward(self, x):
        # f2 = self.feature2(x)       # B, 256, H//4,  W//4
        f3 = self.feature3(x)       # B, 512, H//8,  W//8
        f4 = self.feature4(f3)      # B, 1024, H//16, W//16
        f5 = self.feature5(f4)      # B, 2048, H//32, W//32

        return f3, f4, f5
