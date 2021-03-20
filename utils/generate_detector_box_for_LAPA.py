from retinaface import RetinaFace
import glob
from tqdm import tqdm
import cv2
import os
import multiprocessing
from joblib import Parallel, delayed


detector = RetinaFace(quality="normal")

def task(jpg):
    img = cv2.imread(jpg)
    base = os.path.basename(jpg) + "_bbox.txt"
    
    faces = detector.predict(img)
    if len(faces) ==0 :
        return

    box = [faces[0]['x1'], faces[0]['y1'], faces[0]['x2'], faces[0]['y2']]

    ROOT_OUT = os.path.dirname(jpg).replace("images", "bboxes")
    if not os.path.isdir(ROOT_OUT):
        os.makedirs(ROOT_OUT)

    with open(f'{ROOT_OUT}/{base}', 'w') as f:
        f.write(",".join([str(el) for el in box]))




if __name__ == "__main__":
    ROOT_DATA = "/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa"
    jpgs = glob.glob(ROOT_DATA + "/*/images/*")

    num_cores = 2
    Parallel(n_jobs=num_cores,  require='sharedmem')(delayed(task)(file) for file in jpgs)
