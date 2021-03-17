import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0,'..')
from models.mobilenet import mobilenetv2
from torchvision import transforms

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
    # heatmap = F.softmax(heatmap, dim=-1)


    # Assign non-topk zero values
    # Assign topk with calculated softmax value
    res = torch.zeros(heatmap.shape)
    res = res.scatter(-1, index, score)

    # Reshape to the original size
    heatmap = res.view(N, C, H, W)
    # heatmap = heatmap.view(N, C, H, W)


    return heatmap

def heatmap2softmaxheatmap(heatmap):
    N, C, H, W = heatmap.shape
   
    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N,C,1,-1)
    heatmap = F.softmax(heatmap, dim=-1)


    # Reshape to the original size
    heatmap = heatmap.view(N, C, H, W)

    return heatmap


def coord2heatmap(w, h, ow, oh, x, y):
    """
    Turns an (x,y) coordinate into a lossless heatmap. Arguments:
    x: x coordinate
    y: y coordinate
    s: stride ==4
    """
    # Get scale
    sx = ow/w
    sy = oh/h
    
    # Unrounded target points
    px = x * sx
    py = y * sy
    
    # Truncated coordinates
    nx,ny = int(px), int(py)
    
    # Coordinate error
    ex,ey = px - nx, py - ny    
    
    heatmap = torch.zeros(ow, oh)
   
    
    heatmap[nx][ny] = (1-ex) * (1-ey)
    if (ny+1<oh-1):
        heatmap[nx][ny+1] = (1-ex) * ey
    
    if (nx+1<ow-1):
        heatmap[nx+1][ny] = ex * (1-ey)
    
    if (nx+1<ow-1 and ny+1<oh-1):
        heatmap[nx+1][ny+1] = ex * ey


    return heatmap

"""
\ Generate GT lmks to heatmap
"""
def lmks2heatmap(lmks):
    w,h,ow,oh=256,256,64,64
    heatmap = torch.rand((lmks.shape[0],lmks.shape[1], ow, oh))
    for i in range(lmks.shape[0]):  # num_lmks
        for j in range(lmks.shape[1]):
            heatmap[i][j] = coord2heatmap(w, h, ow, oh, lmks[i][j][0], lmks[i][j][1])
    
    return heatmap

class BinaryHeadBlock(nn.Module):
    """BinaryHeadBlock
    """
    def __init__(self, in_channels, proj_channels, out_channels, **kwargs):
        super(BinaryHeadBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels, out_channels*2, 1, bias=False),
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
    def __init__(self):
        super(HeatmapHead, self).__init__()

        self.decoder = BinaryHeatmap2Coordinate(topk=9, stride=4)

        self.head = BinaryHeadBlock(in_channels=152, proj_channels=152, out_channels=106)

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
        self.backbone = mobilenetv2(pretrained=pretrained, model_url=model_url)
        self.heatmap_head = HeatmapHead()
        self.transform = transforms.Compose([
            transforms.Resize(256, 256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    
    
    def forward(self, x):
        heatmaps, landmark = self.heatmap_head(self.backbone(x))

        # Note that the 0 channel indicate background
        return heatmaps[:,1,...], landmark



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




if __name__ == "__main__":
    import time

    # Inference model
    x = torch.rand((16, 3, 256, 256))
    model = HeatMapLandmarker(pretrained=False)
    heatmaps, lmks = model(x)
    print(f'heat size :{heatmaps.shape}. lmks shape :{lmks.shape}')
    topkheatmap = heatmap2topkheatmap(heatmaps, topk=4)

    print(f'heat topk heatmap: ', topkheatmap.shape)

    # Lmks:
    lm = torch.rand(lmks.shape)
    t1 = time.time()

    heatGT = lmks2heatmap(lm)
    print("time:", time.time()-t1)

    print(heatGT.shape)


    # Loss
    rme = loss_heatmap(topkheatmap, heatGT)
    print("Loss:", rme)


