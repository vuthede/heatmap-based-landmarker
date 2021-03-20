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
     heatmap2coord, heatmap2topkheatmap, lmks2heatmap, loss_heatmap, heatmap2softmaxheatmap
from datasets.dataLAPA106 import LAPA106DataSet
from torchvision import  transforms

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:0"

# Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')])

def square_box(box, ori_shape):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)//2, (y1+y2)//2
    w = max(x2-x1, y2-y1)*1.2
    x1 = cx - w//2
    y1 = cy - w//2
    x2 = cx + w//2
    y2 = cy + w//2

    x1 = max(x1, 0)
    y1 = max(y1+(y2-y1)*0, 0)
    x2 = min(x2-(x1-x1)*0, ori_shape[1]-1)
    y2 = min(y2, ori_shape[0]-1)

    return [x1, y1, x2, y2]

def draw_landmarks(img, lmks, color =(0,255,0)):
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, color, -1, lineType=cv2.LINE_AA)

    return img

def concat_gt_heatmap(heat):
    """
    \ Heat size : 106 x 64 x 64
    """
    # print(f'Shape Gt heatmap: {heat.shape}')
    heat = heat.numpy()
    heat = np.max(heat, axis=0)
    heat = heat*255.0/(np.max(heat))
    # heat = cv2.merge([heat, heat, heat])

    return heat



if __name__ == "__main__":
    from retinaface import RetinaFace
    detector = RetinaFace(quality="normal")


    model = HeatMapLandmarker()
    model_path = "/home/vuthede/heatmap-based-landmarker/checkpoint/epoch_30.pth.tar"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['plfd_backbone'])
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture("/home/vuthede/Desktop/hardcases/highlight1.mp4")
    # cap = cv2.VideoCapture("/home/vuthede/Downloads/out0.mp4")

    cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter('demo_hightlight.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280, 720))

    train_dataset = LAPA106DataSet(img_dir='/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train/images',
     anno_dir=f'/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train/landmarks', augment=True,
    transforms=None)

    for img, lmksGT in train_dataset:
        # img, lmksGT = train_dataset[8]
        print(img.shape, lmksGT.shape)
        img = np.array(img)
        lmksGT = lmksGT.reshape(-1, 2)
        print(lmksGT.shape)
        lmksGT = lmksGT * 256  
        
        img_tensor = transform(img)

        img_tensor = torch.unsqueeze(img_tensor, 0)  # 1x3x256x256

        pred_heatmap, lmks = model(img_tensor.to('cuda:0'))

        lmks = lmks.cpu().detach().numpy()[0] # 106x2
       
        img = draw_landmarks(img, lmks)
        img = draw_landmarks(img, lmksGT[[104 ,105]], color= (0,0,255))

        heatGT = lmks2heatmap(np.array([lmksGT]), True)[0]
        heatGT_vis = concat_gt_heatmap(heatGT)

        pred_heatmap = heatmap2softmaxheatmap(pred_heatmap)
        pred_heatmap_vis = pred_heatmap.cpu().detach()[0]  # 106x64x64
        pred_heatmap_vis = concat_gt_heatmap(pred_heatmap_vis)


        img = cv2.resize(img, None, fx=2, fy=2)
        cv2.imshow("Image", img)
        heatGT_vis = cv2.resize(heatGT_vis, None, fx=2, fy=2)
        pred_heatmap_vis = cv2.resize(pred_heatmap_vis, None, fx=2, fy=2)

        cv2.imshow("GT heatmap", heatGT_vis)
        cv2.imshow("pred heatmap", pred_heatmap_vis)



        k = cv2.waitKey(0)
        
        if k==27:
            break

    cv2.destroyAllWindows()


    # while 1:
    #     ret, img = cap.read()
    #     img = cv2.resize(img, (1280, 720))

    #     if not ret:
    #         break

    #     # Get box detector and then make it square
    #     faces = detector.predict(img)
    #     if len(faces) ==0 :
    #         continue

    #     box = [faces[0]['x1'], faces[0]['y1'], faces[0]['x2'], faces[0]['y2']]
    #     box = square_box(box, img.shape)
    #     box = list(map(int, box))
    #     x1, y1, x2, y2 = box

    #     # Inference lmks
    #     crop_face = img[y1:y2, x1:x2]
    #     crop_face = cv2.resize(crop_face, (256, 256))
    #     img_tensor = transform(crop_face)
    #     img_tensor = torch.unsqueeze(img_tensor, 0)  # 1x3x256x256

    #     _, lmks = model(img_tensor.to(device))

    #     lmks = lmks.cpu().detach().numpy()[0] # 106x2
    #     lmks = lmks/256.0  # Scale into 0-1 coordination
    #     lmks[:,0], lmks[:,1] = lmks[: ,0] * (x2-x1) + x1 ,\
    #                            lmks[:, 1] * (y2-y1) + y1

    #     img = draw_landmarks(img, lmks)
    #     img =  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1) 

    #     cv2.imshow("Image", img)
    #     out.write(img)

    #     k = cv2.waitKey(1)
        
    #     if k==27:
    #         break

    # cv2.destroyAllWindows()


