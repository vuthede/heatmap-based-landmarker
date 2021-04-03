
import torch
import torch.nn as nn
import math
import sys
import cv2
import torch.nn.functional as F

# sys.path.insert(0, "./models")



def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        # expand_ratio=1
        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self, alpha=1.0):
        super(PFLDInference, self).__init__()

        self.inplane = 64 #1x
        self.alpha  = alpha
        self.base_channel = int(self.inplane*self.alpha)

        self.conv1 = nn.Conv2d(
            3,  self.base_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.base_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
             self.base_channel,  self.base_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d( self.base_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual( self.base_channel,  self.base_channel, 2, False, 2)
        self.block3_2 = InvertedResidual( self.base_channel,  self.base_channel, 1, True, 2)
        self.block3_3 = InvertedResidual( self.base_channel,  self.base_channel, 1, False, 2)
        self.block3_4 = InvertedResidual( self.base_channel,  self.base_channel, 1, True, 2)
        self.block3_5 = InvertedResidual( self.base_channel,  self.base_channel, 1, False, 2)


        self.conv4_1 = InvertedResidual(self.base_channel, self.base_channel*2, 2, False, 2)

        self.conv5_1 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, False, 4)
        self.block5_2 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, True, 4)
        # self.block5_3 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, True, 4)
        # self.block5_4 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, True, 4)
        # self.block5_5 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, True, 4)
        self.block5_6 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, False, 4)
        self.conv6_1 = InvertedResidual(self.base_channel*2, self.base_channel*3, 1, False, 2)  # [16, 14, 14]

        # self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        # self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        # self.bn8 = nn.BatchNorm2d(128)

       



    def forward(self, x):  
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        print(f"0 shape: {x.shape}")


        c0 = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        print(f"0.1 shape: {c0.shape}")

        c1 = self.conv3_1(c0)
        print(f"1 shape: {c1.shape}")

        x = self.block3_2(c1)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)

        c2 = self.conv4_1(x)
        print(f"2 shape: {c2.shape}")

        x = self.conv5_1(c2)
        x = self.block5_2(x)
        # x = self.block5_3(x)
        # x = self.block5_4(x)

        # x = self.block5_5(x)

        x = self.block5_6(x)
        c3 = self.conv6_1(x)

        print(f"3 shape: {c3.shape}")

        # x1 = self.avg_pool1(x)
        # x1 = x1.view(x1.size(0), -1)

        # x = self.conv7(x)

        # x3 = self.conv8(x)
        kwargs = {'size': c0.shape[-2:],'mode': 'bilinear','align_corners': False}
        features =  torch.cat([F.interpolate(xx,**kwargs) for xx in [c0, c1, c3]], 1)


        


        return features


if __name__ == "__main__":
    # x = torch.rand(1, 3, 256, 256)
    x = torch.rand(1, 3, 192, 192)


    model = PFLDInference(alpha=0.5)
    features  = model(x)
    print(features.shape)

