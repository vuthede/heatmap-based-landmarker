import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
import glob
import os
import mxnet as mx

import albumentations as A
import imgaug.augmenters as iaa
from torchvision import  transforms


# Declare an augmentation pipeline
transformerr = A.Compose([
    A.Blur(blur_limit=5),
    A.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.15, always_apply=False, p=0.5)
])


seq = iaa.Sequential([
    iaa.CoarseDropout((0.1, 0.15), size_percent=(0.02, 0.03)),
    iaa.JpegCompression(compression=(50, 90)),
    iaa.CoarsePepper(0.05, size_percent=(0.02, 0.03)),  
])


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

class LAPA106DataSet(data.Dataset):
    TARGET_IMAGE_SIZE = (256, 256)

    def __init__(self, img_dir, anno_dir, augment=False, transforms=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        self.augment = augment
        self.img_path_list = glob.glob(img_dir + "/*.jpg")

        

    def _get_106_landmarks(self, path):
        file1 = open(path, 'r') 
        ls = file1.readlines() 
        ls = ls[1:] # Get only lines that contain landmarks. 68 lines

        lm = []
        for l in ls:
            l = l.replace("\n","")
            a = l.split(" ")
            a = [float(i) for i in a]
            lm.append(a)
        
        lm = np.array(lm)
        assert len(lm)==106, "There should be 106 landmarks. Get {len(lm)}"
        return lm
    
    def __getitem__(self, index):
        f = self.img_path_list[index]
        self.img = cv2.imread(f)

        replacing_extension = ".txt"
       
        self.landmark = self._get_106_landmarks(f.replace(self.img_dir, self.anno_dir).replace(".jpg", replacing_extension))


        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        boxsize = int(np.max(wh)*1.25)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = self.img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = self.img[y1:y2, x1:x2]

        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        # Original cut face and lamks
        imgT_original = cv2.resize(imgT, self.TARGET_IMAGE_SIZE)
        landmark_original = (self.landmark - xy)/boxsize
        landmark_original = np.reshape(landmark_original, (-1)).astype(np.float32)
        assert (landmark_original >= 0).all(), str(landmark_original) + str([dx, dy])
        assert (landmark_original <= 1).all(), str(landmark_original) + str([dx, dy])
        if self.transforms:
            imgT_original = self.transforms(imgT_original)
        else:
            imgT_original = np.array(imgT_original)

        if not self.augment:
            fail_augment = True

        # Random augmeentation rotate and scale
        fail_augment = False
        angle = np.random.randint(-30, 30)
        cx, cy = center
        cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
        cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
        M, landmark = rotate(angle, (cx,cy), self.landmark)
        imgT = cv2.warpAffine(self.img, M, (int(self.img.shape[1]*1.25), int(self.img.shape[0]*1.25)))
        wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
        size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
        xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
        landmark = (landmark - xy) / size
        landmark = np.reshape(landmark, (-1)).astype(np.float32)

        if (landmark < 0).any() or (landmark >= 1).any():
            fail_augment=True

        if fail_augment:
            # print("yooooooooooooooooo")
            return imgT_original, landmark_original

        x1, y1 = xy
        x2, y2 = xy + size
        height, width, _ = imgT.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = imgT[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx >0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        imgT = cv2.resize(imgT, self.TARGET_IMAGE_SIZE)

        # Augment more  lighting, erase, noise
        transformed = transformerr(image=imgT)

        imgT = transformed["image"]
        imgT = seq(image=imgT)



        if self.transforms:
            imgT = self.transforms(imgT)
        else:
            imgT = np.array(imgT)

        
        return imgT, landmark

    def __getitem1__(self, index):
        f = self.img_path_list[index]
        self.img = cv2.imread(f)
        h, w, _ = self.img.shape
        self.img = cv2.resize(self.img, self.TARGET_IMAGE_SIZE)
        self.landmark = self._get_106_landmarks(f.replace(self.img_dir, self.anno_dir).replace(".jpg", ".txt"))
        self.landmark[:,0] = self.landmark[:,0] * (self.TARGET_IMAGE_SIZE[1]/w) 
        self.landmark[:,1] = self.landmark[:,1] * (self.TARGET_IMAGE_SIZE[0]/h) 
        self.landmark = np.reshape(self.landmark, (-1))

        self.landmark = self.landmark/self.TARGET_IMAGE_SIZE[0]


        if self.transforms:
            self.img = self.transforms(self.img)

        
        return (self.img, self.landmark.astype(np.float32))
        

    def __len__(self):
        return len(self.img_path_list)


if __name__ == "__main__":
    import torch

    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    lapa = LAPA106DataSet(img_dir="/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train/images",
                            anno_dir="/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train/landmarks",
                            transforms=transform)
    
    print(len(lapa))
    for img, landmarks in lapa:
        print(img.shape, torch.max(img))
        break
    #     landmarks = np.reshape(landmarks, (-1,2))
    #     for p in landmarks:
    #         img = cv2.circle(img, tuple(p*256), 1, (255, 0, 0), 1)
        
    #     cv2.imshow("Imgae", img)

    #     k = cv2.waitKey(0)

    #     if k==27:
    #         break
    
    # cv2.destroyAllWindows()

        