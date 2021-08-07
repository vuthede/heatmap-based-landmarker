import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
import glob
import os
#import mxnet as mx

import albumentations as A
import imgaug.augmenters as iaa
from torchvision import  transforms
import ast
import random
import math

# Declare an augmentation pipeline
category_ids = [0]
transformerr = A.Compose(
    [
        A.ColorJitter (brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.7),
        A.ShiftScaleRotate (shift_limit_x=0.0625, shift_limit_y=(-0.3, 0.4), scale_limit=0.25, rotate_limit=30, interpolation=1, border_mode=4, always_apply=False, p=0.5)
        # A.ShiftScaleRotate (shift_limit_y=(0.1, 0.4), scale_limit=0.25, rotate_limit=30, interpolation=1, border_mode=4, always_apply=False, p=0.5)
       
    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)



def square_box(box, ori_shape, lmks, expand=1.25):
    """
    \Square the box with expandsion
    Note: Square as long as it cover at least the "bigger" lmks points
    """

    # Box of detector
    x1, y1, x2, y2 = box

    # min and max of lmks
    minx_lmk = np.min(lmks[:,0])
    miny_lmk = np.min(lmks[:,1])
    maxx_lmk = np.max(lmks[:,0])
    maxy_lmk = np.max(lmks[:,1])

    # Get the min and the max 
    x1 = min(x1, minx_lmk)
    y1 = min(y1, miny_lmk)
    x2 = max(x2, maxx_lmk)
    y2 = max(y2, maxy_lmk)


    cx, cy = (x1+x2)//2, (y1+y2)//2
    w = max(x2-x1, y2-y1)*expand
    
    x1 = cx - w//2 - 2
    y1 = cy - w//2 - 2
    x2 = cx + w//2 + 2
    y2 = cy + w//2 + 2

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, ori_shape[1]-1)
    y2 = min(y2, ori_shape[0]-1)

    return [x1, y1, x2, y2]

def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                             M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_



class VW300(data.Dataset):
    TARGET_IMAGE_SIZE = (256, 256)


    def __init__(self, img_dir, anno_dir, augment=False, transforms=None, imgsize=256, set_type="train"):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        self.augment = augment


        assert set_type in ["train", "val"], "set_type have to be train or val"


        folders = os.listdir(img_dir)
        self.img_path_list = []
        if not os.path.isfile("trainlist300vw.npy") or not os.path.isfile("vallist300vw.npy"):
            for folder in folders:
                self.img_path_list += glob.glob(img_dir + f"/{folder}/*.jpg")

            random.Random(777).shuffle(self.img_path_list)
            if set_type=="train":
                self.img_path_list = self.img_path_list[0:int(len(self.img_path_list)*0.9)]
                np.save("trainlist300vw.npy", self.img_path_list)

            elif set_type=="val":
                self.img_path_list = self.img_path_list[int(len(self.img_path_list)*0.9):]
                np.save("vallist300vw.npy", self.img_path_list)
            else:
                raise NotImplementedError
        else:
            if set_type=="train":
                self.img_path_list = np.load("trainlist300vw.npy")
            elif set_type=="val":
                self.img_path_list = np.load("vallist300vw.npy")
            else:
                raise NotImplementedError
        
        self.TARGET_IMAGE_SIZE = (imgsize, imgsize)


    
    def _get_landmarks(self, gt):
        with open(gt, "r") as f:
            lines = f.readlines()
        
        lmk = [-1] * 68*2
        for i, line in enumerate(lines[3:-1]):
            splitted = line.strip().split()
            lmk[i*2] = float(splitted[0])
            lmk[i*2 + 1] = float(splitted[1])
        lmk = np.array(lmk)
        lmk = np.reshape(lmk,(-1,2))
        assert len(lmk)==68, f"There should be 106 landmarks. Get {len(lmk)}"
        return lmk

    
    def lmks2box(self, lmks, expand_forehead=0.2):
        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)

        # Expand forehead
        expand = (zz[1] - xy[1]) * expand_forehead
        xy[1] -= expand
        xy[1] = max(xy[1], 0) 

        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        boxsize = int(np.max(wh)*1.1)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = self.img.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        return [x1, y1, x2, y2]

    def __get_default_item(self):
        return self.__getitem__(random.randint(0, self.__len__()-1))

    def  __getitem__(self, index):

        f = self.img_path_list[index]
        self.img = cv2.imread(f)

        replacing_extension = ".pts"
        anno  = f.replace(self.img_dir, self.anno_dir)
        pre, pos = os.path.dirname(anno), os.path.basename(anno).replace(".jpg", replacing_extension)
        anno = pre + "/annot/" + pos

        self.landmark = self._get_landmarks(anno)


        self.box = None
        if self.box is None:
            expand_random = random.uniform(0.1, 0.17)
            self.box = self.lmks2box(self.landmark, expand_forehead=expand_random)

        # If fail then get the default item
        if np.min(self.landmark[:,0]) < 0 or \
           np.min(self.landmark[:,1]) < 0 or \
           np.max(self.landmark[:,0]) >= self.img.shape[1]  or \
           np.max(self.landmark[:,1]) >= self.img.shape[0] :
        #    print("Get default itemmmmmmmmmmmmmmmmm!")
           return self.__get_default_item()


        # Round box in case box out of range
        x1, y1, x2, y2 = self.box
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, self.img.shape[1]-1)
        y2 = min(y2, self.img.shape[0]-1)
        self.box = [x1, y1, x2, y2]


        if self.augment:
            transformed = transformerr(image=self.img, bboxes=[self.box], category_ids=category_ids,keypoints= self.landmark )
            imgT = np.array(transformed["image"])
            boxes = np.array(transformed["bboxes"])
            lmks = np.array(transformed["keypoints"])

            lmks_ok = (lmks.shape == self.landmark.shape)
            box_ok = len(boxes) >0
            augment_sucess = lmks_ok and box_ok
            if (augment_sucess):
                imgT = imgT
                box = boxes[0]
                lmks = lmks
            else:
                # print("Augment not success!!!!!!!!")
                imgT = self.img
                box = self.box
                lmks = self.landmark
        else:
            imgT = self.img
            box = self.box
            lmks = self.landmark

       
        assert  (lmks.shape == self.landmark.shape), f'Lmks Should have shape {self.landmark.shape}'

        expand_random = random.uniform(1.0, 1.1)
        box = square_box(box, imgT.shape, lmks, expand=expand_random)
        x1, y1, x2, y2 = list(map(math.ceil, box))
        imgT = imgT[y1:y2, x1:x2]
       
  

        lmks[:,0], lmks[:,1] = (lmks[: ,0] - x1)/imgT.shape[1] ,\
                               (lmks[:, 1] - y1)/imgT.shape[0]

        imgT = cv2.resize(imgT, self.TARGET_IMAGE_SIZE)
        augment_sucess = (lmks.shape == self.landmark.shape) and ((lmks >= 0).all())\
                             and ((lmks < 1).all())

        if self.transforms is not None:
            imgT = self.transforms(imgT)  # Normalize, et
        
        return imgT, lmks

        # return None, None

    

    def __len__(self):
        return len(self.img_path_list)


# if __name__ == "__main__":
#     import torch

#     transform = transforms.Compose([transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])


#     lapa = VW300(img_dir="/vinai/devt/landmarkV2/300VW_frames",
#                 anno_dir="/vinai/devt/landmarkV2/300VW_Dataset_2015_12_14",
#                 augment=True,
#                 transforms=None, set_type="train")
    

#     lapaval = VW300(img_dir="/vinai/devt/landmarkV2/300VW_frames",
#                 anno_dir="/vinai/devt/landmarkV2/300VW_Dataset_2015_12_14",
#                 augment=True,
#                 transforms=None, set_type="val")
    
#     print(len(lapa), len(lapaval))
#     img, landmarks = lapa[0]
#     print(img.shape, landmarks.shape)
#     for p in landmarks:
#         p = p*256.0
#         p = p.astype(int)

#         img = cv2.circle(img, tuple(p), 1, (255, 0, 0), 1)
    
#     cv2.imwrite("300vw.png", img)

    

        
