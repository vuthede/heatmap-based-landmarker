import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['MobileNetV2', 'mobilenetv2']

import time

DEBUG=False

class Block(nn.Module):
    """ 
    Bottleneck Residual Block
    
    """
    def __init__(self, in_channels, out_channels, expansion=1, stride=1):
        super(Block, self).__init__()
        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            channels = expansion * in_channels
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.residual = (stride == 1) and (in_channels == out_channels)


    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out = out + x
        return out


class MobileNetV2(nn.Module):
    def __init__(self, config):
        super(MobileNetV2, self).__init__()
        in_channels = config[0][1]
        features = [nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )]
        for expansion, out_channels, blocks, stride in config[1:]:
            for i in range(blocks):
                features.append(Block(in_channels, out_channels, expansion, stride if i == 0 else 1))
                in_channels = out_channels
        self.features = nn.Sequential(*features)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # print(type(m.weight[0][0][0][0].item()))   
        #         print("before: ", m.weight[0][0][0][0])  

        # weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.normal_(m.bias)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.normal_(m.weight)
        #         nn.init.normal_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.normal_(m.bias)

    def forward(self, x):
        t1 = time.time()
        c2 = self.features[:4](x)
        c3 = self.features[4:7](c2)
        c4 = self.features[7:14](c3)
        print("Features c1, c2, c3 shape: ", c2.shape, c3.shape, c4.shape)
        print("Time inference mobile featuresss: ", (time.time()-t1)*1000 )
        kwargs = {'size': c2.shape[-2:],'mode': 'bilinear','align_corners': False}
        # features =  torch.cat([F.interpolate(xx,**kwargs) for xx in [c2[:,0:2,...]]], 1)

        if DEBUG:
            print(f'------------------------- \nFeatures shape mobilev2: {features.shape}\n---------------------------------')

        return c2

def mobilenetv2(pretrained=False, **kwargs):
    """Constructs a MobileNetv2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    config = [
        (1,  32, 1, 1),
        (1,  16, 1, 1),
        (6,  24, 2, 2), 
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
    ]
    model = MobileNetV2(config=config)
    if pretrained:
        assert kwargs["model_url"] is not None, f'Model url should not be  None'
        print("Loading weight image net-----------------------------------")
        model.load_state_dict(model_zoo.load_url(kwargs["model_url"]), strict=False)
    return model


# if __name__ == "__main__":
#     x = torch.rand((1, 3, 256,256))
#     model = mobilenetv2()
#     a = model(x)
#     print(a.shape)

