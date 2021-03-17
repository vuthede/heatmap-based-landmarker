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
from models.heatmapmodel import HeatMapLandmarker
from datasets.dataLAPA106 import LAPA106DataSet

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




def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f'Save checkpoint to {filename}')


def train_one_epoch(traindataloader, model, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    num_batch = len(traindataloader)
    i = 0

    return 0
    

def validate(valdataloader, model, epoch,optimizer, args):
    if not os.path.isdir(args.snapshot):
        os.makedirs(args.snapshot)

    logFilepath  = os.path.join(args.snapshot, args.log_file)

    logFile  = open(logFilepath, 'a')

    model.eval()
    losses = AverageMeter()

    num_vis_batch = 40
    batch = 0

    return 0


## Visualization
def _put_text(img, text, point, color, thickness):
    img = cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, thickness, cv2.LINE_AA)
    return img

def draw_landmarks(img, lmks):
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, (255,0,0), -1, lineType=cv2.LINE_AA)

    return img

def vis_prediction_batch(batch, imgs, lmks, output="./vis"):
    """
    \eye_imgs batchx1x64x64
    \gaze_ears batchx3
    """
    if not os.path.isdir(output):
        os.makedirs(output)
    
   




def main(args):
    # Init model
    model = HeatMapLandmarker(pretrained=False)
    model.to(device)

  

    # Train dataset, valid dataset
    train_dataset = LAPA106DataSet(img_dir=f'{args.dataroot}/images', anno_dir=f'{args.dataroot}/landmarks')
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
        lr=0.001,
        weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60 ,gamma=0.1)
    
    for epoch in range(100000):
        train_one_epoch(traindataloader, model, optimizer, epoch)
        validate(validdataloader, model, epoch, optimizer,args)
        save_checkpoint({
            'epoch': epoch,
            'plfd_backbone': model.state_dict()
        }, filename=f'{args.snapshot}/epoch_{epoch}.pth.tar')
        scheduler.step()



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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

