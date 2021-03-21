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
import ast
import random
import math

# Declare an augmentation pipeline
category_ids = [0]
transformerr = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),  ## Becareful when using that, because the keypoint is flipped but the index is flipped too
        A.ColorJitter (brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.7),
        A.ShiftScaleRotate (shift_limit=0.0625, scale_limit=0.25, rotate_limit=30, interpolation=1, border_mode=4, always_apply=False, p=0.5)
       
    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
)

# transformerr = A.Compose(
#     [
#         # A.HorizontalFlip(p=0.5),
#         A.ColorJitter (brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.7),
#         # A.ShiftScaleRotate (shift_limit=0.0625, scale_limit=0.25, rotate_limit=30, interpolation=1, border_mode=4, always_apply=False, p=0.5)
       
#     ],
# )


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
    
    def _get_detector_box(self, path):
        if not os.path.isfile(path):
            return None

        with open(path, 'r') as f:
            l = f.readline()
            box = ast.literal_eval(l)
        return box
    
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
        return self.__getitem__(0)

    def  __getitem__(self, index):

        f = self.img_path_list[index]
        # f="/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train/images/LFPW_image_test_0116_0.jpg"
        # print("File: ", f)
        self.img = cv2.imread(f)
        # self.img = cv2.imread(f)

        replacing_extension = ".txt"
        self.landmark = self._get_106_landmarks(f.replace(self.img_dir, self.anno_dir).replace(".jpg", replacing_extension))
        self.box = self._get_detector_box(f.replace("images", "bboxes")+"_bbox.txt")

        self.box = None
        if self.box is None:
            expand_random = random.uniform(0.1, 0.17)
            self.box = self.lmks2box(self.landmark, expand_forehead=expand_random)

        # If fail then get the default item
        if np.min(self.landmark[:,0]) < 0 or \
           np.min(self.landmark[:,1]) < 0 or \
           np.max(self.landmark[:,0]) >= self.img.shape[1]  or \
           np.max(self.landmark[:,1]) >= self.img.shape[0] :
           print("Get default itemmmmmmmmmmmmmmmmm!")
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
                print("Augment not success!!!!!!!!")
                imgT = self.img
                box = self.box
                lmks = self.landmark
        else:
            imgT = self.img
            box = self.box
            lmks = self.landmark

       
        assert  (lmks.shape == self.landmark.shape), f'Lmks Should have shape {self.landmark.shape}'

        expand_random = random.uniform(1.0, 1.1)
        # print("Expand: ", expand_random)
        box = square_box(box, imgT.shape, lmks, expand=expand_random)
        x1, y1, x2, y2 = list(map(math.ceil, box))
        imgT = imgT[y1:y2, x1:x2]
       
        # print("Image size: ", self.img.shape)
        # print(f"Before max /lmks :{np.max(lmks)}. Min lmks : {np.min(lmks)}, .Box :{[x1,y1,x2,y2]}")

        lmks[:,0], lmks[:,1] = (lmks[: ,0] - x1)/imgT.shape[1] ,\
                               (lmks[:, 1] - y1)/imgT.shape[0]

        imgT = cv2.resize(imgT, self.TARGET_IMAGE_SIZE)
        augment_sucess = (lmks.shape == self.landmark.shape) and ((lmks >= 0).all())\
                             and ((lmks < 1).all())
        
        # print(f"max lmks :{np.max(lmks)}. Min lmks : {np.min(lmks)} .Box :{[x1,y1,x2,y2]}")

        # print("Success: ", augment_sucess)
        
        if self.transforms is not None:
            imgT = self.transforms(imgT)  # Normalize, et
        
        return imgT, lmks

    
    def __getitem1__(self, index):
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

        # cv2.imshow("origin", self.img)

        imgT = self.img[y1:y2, x1:x2]

        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        # cv2.imshow("border", imgT)


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
        angle = np.random.randint(-10, 10)
        cx, cy = center
        cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
        cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
        M, landmark = rotate(angle, (cx,cy), self.landmark)
        # print("Origin Min lmks, max lmks:", np.min(self.landmark), ", ", np.max(self.landmark))

        # print("Min lmks, max lmks:", np.min(landmark), ", ", np.max(landmark))

        imgT = cv2.warpAffine(self.img, M, (int(self.img.shape[1]), int(self.img.shape[0])))
        # cv2.imshow("rotate", imgT)

        wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
        size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
        xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
        landmark = (landmark - xy) / size
        landmark = np.reshape(landmark, (-1)).astype(np.float32)

        if (landmark < 0).any() or (landmark >= 1).any():
            fail_augment=True

        if fail_augment:
            # print("Fail augment")
            return imgT_original, landmark_original
        else:
            # print("sucess")
            pass

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
        # imgT = seq(image=imgT)



        if self.transforms:
            imgT = self.transforms(imgT)
        else:
            imgT = np.array(imgT)

        
        return imgT, landmark
    

    def __len__(self):
        return len(self.img_path_list)


if __name__ == "__main__":
    import torch

    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    lapa = LAPA106DataSet(img_dir="/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train/images",
                            anno_dir="/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train/landmarks",
                            augment=True,
                            transforms=None)
    
    print(len(lapa))
    for img, landmarks in lapa:
        print(img.shape, landmarks.shape)
        # landmarks = np.reshape(landmarks, (-1,2))
        for p in landmarks:
            p = p*256.0
            p = p.astype(int)

            img = cv2.circle(img, tuple(p), 1, (255, 0, 0), 1)
        
        cv2.imshow("Imgae", img)

        k = cv2.waitKey(0)

        if k==27:
            break
    
    cv2.destroyAllWindows()

        