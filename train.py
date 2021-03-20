#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import time
import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import cv2
import sys
from models.heatmapmodel import HeatMapLandmarker,\
     heatmap2coord, heatmap2topkheatmap, lmks2heatmap, loss_heatmap, cross_loss_entropy_heatmap,\
                     heatmap2softmaxheatmap
from datasets.dataLAPA106 import LAPA106DataSet
from torchvision import  transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')])


transform_valid = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f'Save checkpoint to {filename}')


def compute_nme(preds, target, typeerr='inter-ocular'):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """

    assert typeerr in ['inter-ocular', 'inter-pupil'], f'Typeerr should be in { ["inter-ocular", "inter-pupil"]}'

    preds = preds.reshape(preds.shape[0], -1, 2).detach().cpu().numpy() # landmark 
    target = target.reshape(target.shape[0], -1, 2).detach().cpu().numpy() # landmark_gt

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    if typeerr=='inter-ocular':
        l, r = 66, 79
    else:
        l, r = 104, 105

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]

        eye_distant = np.linalg.norm(pts_gt[l ] - pts_gt[r])
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (eye_distant)

    return rmse


def train_one_epoch(traindataloader, model, optimizer, epoch, args=None):
    model.train()
    losses = AverageMeter()
    num_batch = len(traindataloader)
    i = 0

    for img, lmksGT in traindataloader:
        i += 1
       
        # img shape: B x 3 x 256 x 256
        # NORMALZIED lmks shape: B x 106 x 256 x 256
        img = img.to(device)

        # Denormalize lmks
        lmksGT = lmksGT.view(lmksGT.shape[0],-1, 2)
        lmksGT = lmksGT * 256  
        
        # Generate GT heatmap by randomized rounding
        heatGT = lmks2heatmap(lmksGT, args.random_round)  

        # Inference model to generate heatmap
        heatPRED, lmksPRED = model(img.to(device))


        if args.random_round: #Using cross loss entropy
            heatPRED = heatPRED.to('cpu')

            loss = cross_loss_entropy_heatmap(heatPRED, heatGT, pos_weight=torch.Tensor([args.pos_weight]))
        else:
            # MSE loss
            if (args.get_topk_in_pred_heats_training):
                heatPRED = heatmap2topkheatmap(heatPRED.to('cpu'))
            else:
                heatPRED = heatmap2softmaxheatmap(heatPRED.to('cpu'))
            
            # Loss
            loss = loss_heatmap(heatPRED, heatGT)
    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        print(f"Epoch:{epoch}. Lr:{optimizer.param_groups[0]['lr']} Batch {i} / {num_batch} batches. Loss: {loss.item()}")

    return losses.avg

    

def validate(valdataloader, model, optimizer, epoch, args):
    if not os.path.isdir(args.snapshot):
        os.makedirs(args.snapshot)

    logFilepath  = os.path.join(args.snapshot, args.log_file)

    logFile  = open(logFilepath, 'a')

    model.eval()
    losses = AverageMeter()
    num_batch = len(valdataloader)


    num_vis_batch = 250
    batch = 0
    nme_interocular = []
    nme_interpupil = []

    for img, lmksGT in valdataloader:
        img = np.array(img)
        batch += 1
        # img shape: B x  256 x 256 x3
        # NORMALZIED lmks shape: B x 106 x 256 x 256
        img_ori = img.copy()
        new_img = []
        for i in range(len(img)):
            new_img.append(transform_valid(img[i]).numpy())  #B x 3 x 256 x 256
        img = torch.Tensor(np.array(new_img))

        img = img.to(device)

        # Denormalize lmks
        lmksGT = lmksGT.view(lmksGT.shape[0],-1, 2)
        lmksGT = lmksGT * 256  
        
        # Generate GT heatmap by randomized rounding
        # print(lmksGT.shape)
        heatGT = lmks2heatmap(lmksGT, args.random_round)  

        # Inference model to generate heatmap
        heatPRED, lmksPRED = model(img.to(device))

        if args.random_round: #Using cross loss entropy
            heatPRED =heatPRED.to('cpu')
            loss = cross_loss_entropy_heatmap(heatPRED, heatGT, pos_weight=torch.Tensor([args.pos_weight]))
        
        else:
            # MSE loss
            if (args.get_topk_in_pred_heats_training):
                heatPRED = heatmap2topkheatmap(heatPRED.to('cpu'))
            else:
                heatPRED = heatmap2softmaxheatmap(heatPRED.to('cpu'))
            
            # Loss
            loss = loss_heatmap(heatPRED, heatGT)

        if batch < num_vis_batch:
            vis_prediction_batch(batch, img_ori, lmksPRED)


        # Loss
        nme_interocular_batch = list(compute_nme(lmksPRED, lmksGT, typeerr='inter-ocular'))
        nme_interpupil_batch = list(compute_nme(lmksPRED, lmksGT, typeerr='inter-pupil'))


        nme_interocular += nme_interocular_batch
        nme_interpupil += nme_interpupil_batch


        losses.update(loss.item())
        message = f"VAldiation Epoch:{epoch}. Lr:{optimizer.param_groups[0]['lr']} Batch {batch} / {num_batch} batches. Loss: {loss.item()}.  NME_ocular :{np.mean(nme_interocular_batch)}. NME_pupil :{np.mean(nme_interpupil_batch)}"
        print(message)
    
    message = f" Epoch:{epoch}. Lr:{optimizer.param_groups[0]['lr']}. Loss :{losses.avg}. NME_ocular :{np.mean(nme_interocular)}. NME_pupil :{np.mean(nme_interpupil)}"
    logFile.write(message + "\n")

    return losses.avg


## Visualization
def _put_text(img, text, point, color, thickness):
    img = cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, thickness, cv2.LINE_AA)
    return img

def draw_landmarks(img, lmks):
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, (255,0,0), -1, lineType=cv2.LINE_AA)

    return img

def vis_prediction_batch(batch, img, lmk, output="./vis"):
    """
    \eye_imgs Bx256x256x3
    \lmks Bx106x2
    """
    if not os.path.isdir(output):
        os.makedirs(output)
    
    for i in range(len(img)):
        image = draw_landmarks(img[i], lmk.cpu().detach().numpy()[i])
        cv2.imwrite(f'{output}/batch_{batch}_image_{i}.png', image)
    


def main(args):
    # Init model
    model = HeatMapLandmarker(pretrained=True, model_url="https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1")
    
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['plfd_backbone'])
        model.to(device)

    
    
    model.to(device)

  

    # Train dataset, valid dataset
    train_dataset = LAPA106DataSet(img_dir=f'{args.dataroot}/images', anno_dir=f'{args.dataroot}/landmarks', augment=True,
    transforms=transform)
    val_dataset = LAPA106DataSet(img_dir=f'{args.val_dataroot}/images', anno_dir=f'{args.val_dataroot}/landmarks')

    # Dataloader
    traindataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=2,
        drop_last=True)

    
    validdataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        [{
            'params': model.parameters()
        }],
        lr=args.lr,
        weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size ,gamma=args.gamma)
    
    # for im, lm in train_dataset:
    #     print(type(im), lm.shape)

    if args.mode == 'train':
        for epoch in range(100000):
            train_one_epoch(traindataloader, model, optimizer, epoch, args)
            validate(validdataloader, model, optimizer, epoch, args)
            save_checkpoint({
                'epoch': epoch,
                'plfd_backbone': model.state_dict()
            }, filename=f'{args.snapshot}/epoch_{epoch}.pth.tar')
            scheduler.step()
    else:  # inference mode
        validate(validdataloader, model, optimizer, -1, args)




def parse_args():
    parser = argparse.ArgumentParser(description='pfld')

    parser.add_argument(
        '--snapshot',
        default='./checkpoint/',
        type=str,
        metavar='PATH')

    parser.add_argument(
        '--log_file', default="log.txt", type=str)

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default='/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=16, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    parser.add_argument('--get_topk_in_pred_heats_training', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--step_size', default=60, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--resume', default="", type=str)
    parser.add_argument('--random_round', default=1, type=int)
    parser.add_argument('--pos_weight', default=64*64, type=int)
    parser.add_argument('--mode', default='train', type=str)





    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

          