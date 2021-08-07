import numpy as np
import cv2
import sys
from torch.utils import data
import glob
import os
import albumentations as A
import imgaug.augmenters as iaa
from torchvision import  transforms
import ast
import random
import math
sys.path.insert(0, "../datasets")
from data300VW import VW300
from data300WLP import W300LargePose
from data300WStyle import W300Style




if __name__=="__main__":
    import torch
    vw300 = VW300(img_dir="/home/ubuntu/vuthede/landmarkV2/300VW_frames",
                anno_dir="/home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14",
                augment=False,
                transforms=None, set_type="train")

    style = W300Style(img_dir="/home/ubuntu/vuthede/landmarkV2/300W-Convert",
            anno_dir="/home/ubuntu/vuthede/landmarkV2/300W-Convert",
            augment=False,
            transforms=None, set_type="train")

    
    lp = W300LargePose(img_dir="/home/ubuntu/vuthede/landmarkV2/300W-LP",
                anno_dir="/home/ubuntu/vuthede/landmarkV2/300W-LP",
                augment=False,
                transforms=None, set_type="train")
    
    print(len(vw300), len(style), len(lp))
    concat_dataset =  torch.utils.data.ConcatDataset([vw300, style, lp])
    print(len(concat_dataset))

    i = 0
    total_sample = 20000
    while True:
        rand_ind = random.randint(0, len(concat_dataset)-1)
        # print(rand_ind)
        i += 1
        if i > total_sample:
            break
        
        print(i)
        
        img, landmarks = concat_dataset[rand_ind]

        png = f'random_samples_20k/{i}.png'
        cv2.imwrite(png, img)
        
        txt = png.replace(".png", ".txt")
        with open(txt, 'w') as f :
            landmarks = landmarks*256.0
            landmarks = landmarks.reshape(-1,)
            landmarks = [str(i) for i in landmarks]
            landmarks_str = ",".join(landmarks)
            f.write(landmarks_str)



        


