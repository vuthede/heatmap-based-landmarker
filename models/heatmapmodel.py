import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0,'..')
from models.mobilenet import mobilenetv2
from models.mobilenetv2faster import PFLDInference
from torchvision import transforms
import random
import numpy as np
import cv2
import time



DEBUG = False

"""
\ Heatmap BxCxHxW to BxCx2
  Used when inference time
"""
def heatmap2coord(heatmap, topk=7):
    N, C, H, W = heatmap.shape
    score, index = heatmap.view(N,C,1,-1).topk(topk, dim=-1)
    coord = torch.cat([index%W, index//W], dim=2)
    return (coord*F.softmax(score, dim=-1)).sum(-1)

"""
\ Predicted heatmap to topk softmax heatmap
 Used when training model. After the decode step, we ave the heatmap 
 then we get only topk points in that and get softmax of those
"""
def heatmap2topkheatmap(heatmap, topk=7):
    """
    \ Find topk value in each heatmap and calculate softmax for them.
    \ Another non topk points will be zero.
    \Based on that https://discuss.pytorch.org/t/how-to-keep-only-top-k-percent-values/83706
    """
    N, C, H, W = heatmap.shape
   
    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N,C,1,-1)
    
    score, index = heatmap.topk(topk, dim=-1)
    score = F.softmax(score, dim=-1)
    heatmap = F.softmax(heatmap, dim=-1)


    # Assign non-topk zero values
    # Assign topk with calculated softmax value
    res = torch.zeros(heatmap.shape)
    res = res.scatter(-1, index, score)

    # Reshape to the original size
    heatmap = res.view(N, C, H, W)
    # heatmap = heatmap.view(N, C, H, W)


    return heatmap

def mean_topk_activation(heatmap, topk=7):
    """
    \ Find topk value in each heatmap and calculate softmax for them.
    \ Another non topk points will be zero.
    \Based on that https://discuss.pytorch.org/t/how-to-keep-only-top-k-percent-values/83706
    """
    N, C, H, W = heatmap.shape
   
    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N,C,1,-1)
    
    score, index = heatmap.topk(topk, dim=-1)
    score = F.sigmoid(score)

    return score


def heatmap2softmaxheatmap(heatmap):
    N, C, H, W = heatmap.shape
   
    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N,C,1,-1)
    heatmap = F.softmax(heatmap, dim=-1)


    # Reshape to the original size
    heatmap = heatmap.view(N, C, H, W)

    return heatmap

def heatmap2sigmoidheatmap(heatmap):
    heatmap = F.sigmoid(heatmap)

    return heatmap

def generate_gaussian(t, x, y, sigma=10):
    """
    Generates a 2D Gaussian point at location x,y in tensor t.
    
    x should be in range (-1, 1) to match the output of fastai's PointScaler.
    
    sigma is the standard deviation of the generated 2D Gaussian.
    """
    _gaussians = {}


    h,w = t.shape
    
    # Heatmap pixel per output pixel
    mu_x = int(0.5 * (x + 1.) * w)
    mu_y = int(0.5 * (y + 1.) * h)
    
    tmp_size = sigma * 3
    
    # Top-left
    x1,y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)
    
    # Bottom right
    x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)
    if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
        return t
    
    size = 2 * tmp_size + 1
    tx = np.arange(0, size, 1, np.float32)
    ty = tx[:, np.newaxis]
    x0 = y0 = size // 2
    
    # The gaussian is not normalized, we want the center value to equal 1
    g = _gaussians[sigma] if sigma in _gaussians \
                else torch.Tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))
    _gaussians[sigma] = g
    
    # Determine the bounds of the source gaussian
    g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
    g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1
    
    # Image range
    img_x_min, img_x_max = max(0, x1), min(x2, w)
    img_y_min, img_y_max = max(0, y1), min(y2, h)
    
    t[img_y_min:img_y_max, img_x_min:img_x_max] = \
      g[g_y_min:g_y_max, g_x_min:g_x_max]
    
    return t

def coord2heatmap(w, h, ow, oh, x, y, sigma=1.5, random_round=False, random_round_with_gaussian=False):
    """
    Inserts a coordinate (x,y) from a picture with 
    original size (w x h) into a heatmap, by randomly assigning 
    it to one of its nearest neighbor coordinates, with a probability
    proportional to the coordinate error.
    
    Arguments:
    x: x coordinate
    y: y coordinate
    w: original width of picture with x coordinate
    h: original height of picture with y coordinate
    """
    # Get scale
    sx = ow / w
    sy = oh / h
    
    # Unrounded target points
    px = x * sx
    py = y * sy
    
    # Truncated coordinates
    nx,ny = int(px), int(py)
    
    # Coordinate error
    ex,ey = px - nx, py - ny

    # Heatmap    
    heatmap = torch.zeros(ow, oh)

    if random_round_with_gaussian:
        xyr = torch.rand(2)
        xx = (ex >= xyr[0]).long()
        yy = (ey >= xyr[1]).long()
        row = min(ny + yy, heatmap.shape[0] - 1)
        col = min(nx+xx, heatmap.shape[1] - 1)

        # Normalize into - 1, 2
        col = (col/float(ow)) * (2) + (-1)
        row = (row/float(oh)) * (2) + (-1)
        heatmap = generate_gaussian(heatmap, col, row, sigma=1.5)


    elif random_round:
        xyr = torch.rand(2)
        xx = (ex >= xyr[0]).long()
        yy = (ey >= xyr[1]).long()
        heatmap[min(ny + yy, heatmap.shape[0] - 1), 
                min(nx+xx, heatmap.shape[1] - 1)] = 1
    else:
        nx = min(nx, ow-1)
        ny = min(ny, oh-1)
        heatmap[ny][nx] = (1-ex) * (1-ey)
        if (ny+1<oh-1):
            heatmap[ny+1][nx] = (1-ex) * ey
        
        if (nx+1<ow-1):
            heatmap[ny][nx+1] = ex * (1-ey)
        
        if (nx+1<ow-1 and ny+1<oh-1):
            heatmap[ny+1][nx+1] = ex * ey
    
    return heatmap

"""
\ Generate GT lmks to heatmap
"""
def lmks2heatmap(w, h, ow, oh, sigma, lmks,random_round=False, random_round_with_gaussian=False):
    # w,h,ow,oh=256,256,64,64
    heatmap = torch.rand((lmks.shape[0],lmks.shape[1], ow, oh))
    for i in range(lmks.shape[0]):  # num_lmks
        for j in range(lmks.shape[1]):
            heatmap[i][j] = coord2heatmap(w, h, ow, oh, sigma, lmks[i][j][0], lmks[i][j][1], random_round=random_round, random_round_with_gaussian=random_round_with_gaussian)
    
    return heatmap

import math
class BinaryHeadBlock(nn.Module):
    """BinaryHeadBlock
    """
    def __init__(self, in_channels, proj_channels, out_channels, **kwargs):
        super(BinaryHeadBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels, out_channels*2, 1,bias=False),
        )   

        
    def forward(self, input):
        N, C, H, W = input.shape
        binary_heats = self.layers(input).view(N, 2, -1, H, W)

        return binary_heats

class BinaryHeatmap2Coordinate(nn.Module):
    """BinaryHeatmap2Coordinate
    """
    def __init__(self, stride=4.0, topk=5, **kwargs):
        super(BinaryHeatmap2Coordinate, self).__init__()
        self.topk = topk
        self.stride = stride
        
    def forward(self, input):
        return self.stride * heatmap2coord(input[:,1,...], self.topk)
        
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'topk={}, '.format(self.topk)
        format_string += 'stride={}'.format(self.stride)
        format_string += ')'
        return format_string
        
class HeatmapHead(nn.Module):
    """HeatmapHead
    """
    def __init__(self,in_channels=152, proj_channels=152, out_channels=106):
        super(HeatmapHead, self).__init__()

        self.decoder = BinaryHeatmap2Coordinate(topk=18, stride=4)

        self.head = BinaryHeadBlock(in_channels=in_channels, proj_channels=proj_channels, out_channels=out_channels)

    def forward(self, input):
        binary_heats = self.head(input)
        lmks = self.decoder(binary_heats)

        if DEBUG:
            print(f'----------------\nBinary heats shape: {binary_heats.shape}\n----------------------------')
            print(f'----------------\nDecoded lmks shape: {lmks.shape}\n----------------------------')

        return binary_heats, lmks
        
class HeatMapLandmarker(nn.Module):
    def __init__(self, pretrained=False, model_url=None):
        super(HeatMapLandmarker, self).__init__()
        # self.backbone = mobilenetv2(pretrained=pretrained, model_url=model_url)
        self.backbone = PFLDInference(alpha=0.25)
        self.heatmap_head = HeatmapHead(in_channels=96, proj_channels=96, out_channels=106)
        # self.transform = transforms.Compose([
        #     transforms.Resize(256, 256),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        # ])
    
    
    def forward(self, x):
        heatmaps, landmark = self.heatmap_head(self.backbone(x))

        # Note that the 0 channel indicate background
        return heatmaps[:,1,...], landmark



class HeatmapHeadInference(nn.Module):
    """HeatmapHead
    """
    def __init__(self):
        super(HeatmapHeadInference, self).__init__()

        self.decoder = BinaryHeatmap2Coordinate(topk=18, stride=4)

        self.head = BinaryHeadBlock(in_channels=96, proj_channels=96, out_channels=68)

    def forward(self, input):
        binary_heats = self.head(input)
        return binary_heats

class HeatMapLandmarkerInference(nn.Module):
    """
        Use when convert from torch --> onnx  --> ncnn model in inference time
    """
    def __init__(self, pretrained=False, model_url=None):
        super(HeatMapLandmarkerInference, self).__init__()
        # self.backbone = mobilenetv2(pretrained=pretrained, model_url=model_url)
        self.backbone = PFLDInference(alpha=0.25)
        self.heatmap_head = HeatmapHeadInference()
        # self.transform = transforms.Compose([
        #     transforms.Resize(256, 256),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        # ])
    
    
    def forward(self, x):
        t1 = time.time()
        fea = self.backbone(x)
        # print("Time mobile v2: ", (time.time()-t1)*1000)

        t2 = time.time()
        heatmaps = self.heatmap_head(fea)
        # print("heatmap head v2: ", (time.time()-t2)*1000)


        # # # Note that the 0 channel indicate background
        B, Two, C, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, C*Two, H, W)
        # heatmaps = heatmaps[:,1,...]

        # return heatmaps  # 1x212x56x56
        return heatmaps

def loss_heatmap(gt, pre):
    """
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Fast_Human_Pose_Estimation_CVPR_2019_paper.pdf
    \gt BxCx64x64
    \pre BxCx64x64
    """
    # nn.MSELoss()
    B, C, H, W = gt.shape
    gt = gt.view(B, C, -1)
    pre = pre.view(B, C, -1)
    loss  = torch.sum((pre-gt)*(pre-gt), axis=-1)  # Sum square error in each heatmap
    loss = torch.mean(loss, axis=-1)  # MSE in 1 sample / batch over all heatmaps
    loss = torch.mean(loss, axis=-1)  # Avarage MSE in 1 batch (.i.e many sample)
    return loss

def cross_loss_entropy_heatmap(p, g, pos_weight=torch.Tensor([1])):
    """\ Bx 106x 256x256
    """
    BinaryCrossEntropyLoss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    B, C, W, H = p.shape

    loss = BinaryCrossEntropyLoss(p, g)
    
    return loss


def adaptive_wing_loss(y_pred, y_true, w=14, epsilon=1.0, theta = 0.5, alpha=2.1):
    """
    \ref https://arxiv.org/pdf/1904.07399.pdf
    """
    # Calculate A and C
    p1 = (1/ (1+(theta/epsilon)**(alpha-y_true)))
    p2 = (alpha-y_true) * ((theta/epsilon)**(alpha-y_true-1)) * (1/epsilon)
    A = w * p1 * p2
    C = theta*A - w*torch.log(1+(theta/epsilon)**(alpha-y_true))

    # Asolute value
    absolute_x = torch.abs( y_true - y_pred)

    # Adaptive wingloss
    losses = torch.where(theta > absolute_x, w * torch.log(1.0 + (absolute_x/epsilon)**(alpha-y_true) ), A*absolute_x-C)
    losses = torch.sum(losses, axis=[2,3])
    losses = torch.mean(losses)

    return losses # Mean wingloss for each sample in batch





if __name__ == "__main__":

    # Inference model
    # x = torch.rand((1, 3, 256, 256))
    x = torch.rand((1, 3, 164, 164))
    model = HeatMapLandmarker(pretrained=False)
    model = HeatMapLandmarkerInference(pretrained=False)
    # model = HeatMapLandmarkerInference(pretrained=True, model_url="https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1")
    model = model.to("cpu")

    # checkpoint_path = "../ckpt/epoch_80.pth.tar"   # Aungment lighting/ fix translation/no histequal  --more epoch
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(checkpoint['plfd_backbone'])

    # params = list(model.parameters())
    # for i in range(len(params)):
    #     params[i].data = torch.round(params[i].data*10**4) / 10**4

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # print("uo: ", type(m.weight[0][0][0][0].item()))  
            print("after: ", m.weight[0][0][0][0])  



    model.eval()
    times = []
    for i in range(10):
        t1 = time.time()
        heatmaps = model(x)
        times.append((time.time() - t1)*1000)
    print("time:", np.mean(times), " (ms)")

    # print(f'heat size :{heatmaps.shape}. lmks shape :{lmks.shape}')
    # topkheatmap = heatmap2topkheatmap(heatmaps, topk=4)

    # print(f'heat topk heatmap: ', topkheatmap.shape)

    # # Lmks:
    # lm = torch.rand(lmks.shape)

    # heatGT = lmks2heatmap(lm)

    # print(heatGT.shape)


    # # Loss
    # rme = loss_heatmap(topkheatmap, heatGT)
    # print("Loss:", rme)


