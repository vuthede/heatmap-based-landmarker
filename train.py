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
                     heatmap2softmaxheatmap, heatmap2sigmoidheatmap, adaptive_wing_loss
# from datasets.dataLAPA106 import LAPA106DataSet
from torchvision import  transforms


from datasets.data300VW import VW300 
from datasets.data300WStyle import W300Style
from datasets.data300WLP import W300LargePose
from datasets.datasetMask import DatasetMask

from logger.TensorboardLogger import TensorBoardLogger
from tqdm import tqdm


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

    if L==106:
        if typeerr=='inter-ocular':
            l, r = 66, 79
        else:
            l, r = 104, 105
    else:
        if typeerr=='inter-ocular':
            l, r = 36, 45
        else:
            l, r = 36, 45


    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]

        eye_distant = np.linalg.norm(pts_gt[l ] - pts_gt[r])
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (eye_distant)

    return rmse


def data_generator(dataloaders, tags=["300VW", "Style", "LP", "Mask"], args=None):
    # Total batch
    total_batch = 0

    # Balance data in each dataset    
    if args.sampling_data:
        min_len_dataloaders = np.min([len(dl) for dl in dataloaders])
        total_batch = min_len_dataloaders * len(tags)
    else:
        for dl in dataloaders:
            total_batch += len(dl)

    print("Total batch need to be trained :", total_batch)
    
    # Iterators
    iters = []
    for i in range(0,len(dataloaders)):
        iters.append(iter(dataloaders[i]))


    if args.curriculum:
        min_len_dataloaders = np.min([len(dl) for dl in dataloaders])
        for dataiter, tag in zip(iters, tags):
            b = 0
            for batch in dataiter:
                if args.sampling_data:
                    b +=1
                    if b>min_len_dataloaders:
                        break
                yield batch, tag
    else:
        for i in range(0, total_batch):
            # random_ind = random.randint(0,len(tags)-1)
            random_ind = i%(len(tags))
            # print(f"Choose batch of dataset :{tags[random_ind]}")
            chosen_data_loader = iters[random_ind] 
            batch = next(iters[random_ind], None)
            
            
            # If the end of dataloader --> reset dataloader
            if not batch:  
                iters[random_ind] = iter(dataloaders[random_ind]) 
                batch = next(iters[random_ind], None)
                assert batch is not None, f'Batch should not be None here!!!!!. Something wrong with dataloader'

            yield batch, tags[random_ind]


def train_shuffle_cross_data(dataloaders, model, optimizer, epoch, args, tensorboardLogger=None,tags=["Synthesis", "MLR", "LaPa", "Mask"]):
    model.train()
    losses300VW = AverageMeter()
    lossesStyle = AverageMeter()
    lossesLP = AverageMeter()
    lossesMask = AverageMeter()



    num_batch = 0

    if args.sampling_data:
        min_len_dataloaders = np.min([len(dl) for dl in dataloaders])
        num_batch = min_len_dataloaders * len(tags)
    else:
        for dl in dataloaders:
            num_batch += len(dl)
    i = 0

    with tqdm(data_generator(dataloaders, tags=tags, args=args), unit="batch") as tepoch:
        for (img, lmksGT), tag in tepoch:
            i +=1
            # if i ==40:
            #     break
            tepoch.set_description(f"Epoch {epoch}")
            img = img.to(device)

            # Groundtruth heatmaps
            lmksGT = lmksGT.view(lmksGT.shape[0],-1, 2)
            x_ok = torch.logical_and(lmksGT[:,:,0] >= 0,  lmksGT[:,:,0] <= 1)
            y_ok = torch.logical_and(lmksGT[:,:,1] >= 0,  lmksGT[:,:,1] <= 1)
            y_true_visible_mask = torch.logical_and(x_ok, y_ok)
            y_true_visible_mask = torch.unsqueeze(y_true_visible_mask, -1)
            y_true_visible_mask = torch.unsqueeze(y_true_visible_mask, -1)
            # print(y_true_visible_mask.shape)

            lmksGT = lmksGT * 256  
            heatGT = lmks2heatmap(lmksGT, args.random_round, args.random_round_with_gaussian)  

            # Predicted heatmaps
            heatPRED, lmksPRED, lmks_regression, lmks_regression_end = model(img.to(device))
            heatPRED = heatmap2sigmoidheatmap(heatPRED.to('cpu'))


            # Loss Adaptive wingloss
            if args.use_visible_mask:
                loss = adaptive_wing_loss(heatPRED, heatGT, y_true_visible_mask=y_true_visible_mask)
            else:
                loss = adaptive_wing_loss(heatPRED, heatGT, y_true_visible_mask=None)

             
            # If use regression loss to force model keep boundary
            if args.include_regression:
                lmks_regression = lmks_regression.view(lmks_regression.shape[0],-1, 2)
                l2_distant = torch.sum((lmksGT.to('cpu')/256.0 - lmks_regression.to('cpu')) * (lmksGT.to('cpu')/256.0 - lmks_regression.to('cpu')), axis=1)
                l2_distant = torch.mean(l2_distant)
                loss = loss + l2_distant * 10
            
            if args.include_regression_end:
                lmks_regression_end = lmks_regression_end.view(lmks_regression_end.shape[0],-1, 2)
                l2_distant = torch.sum((lmksGT.to('cpu')/256.0 - lmks_regression_end.to('cpu')) * (lmksGT.to('cpu')/256.0 - lmks_regression_end.to('cpu')), axis=1)
                l2_distant = torch.mean(l2_distant)
                loss = loss + l2_distant * 10



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if tag=="300VW":
                losses300VW.update(loss.item())

            elif tag=="Style":
                lossesStyle.update(loss.item())
            elif tag=="LP":
                lossesLP.update(loss.item())
            elif tag=="Mask":
                lossesMask.update(loss.item())
            else:
                raise NotImplementedError
            
            # LOG
            tepoch.set_postfix(l2loss=loss.item(), dataset=tag)
        
        # Log train averge loss in each dataset
        tensorboardLogger.log(f"train/loss/300VW", losses300VW.avg, epoch)
        tensorboardLogger.log(f"train/loss/Style", lossesStyle.avg, epoch)
        tensorboardLogger.log(f"train/loss/Lp", lossesLP.avg, epoch)
        tensorboardLogger.log(f"train/loss/Mask", lossesMask.avg, epoch)


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
        heatGT = lmks2heatmap(lmksGT, args.random_round, args.random_round_with_gaussian)  

        # Inference model to generate heatmap
        heatPRED, lmksPRED = model(img.to(device))


        if args.random_round_with_gaussian:
            heatPRED = heatmap2sigmoidheatmap(heatPRED.to('cpu'))
            loss = adaptive_wing_loss(heatPRED, heatGT)

        elif args.random_round: #Using cross loss entropy
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

        
        # If use regression loss to force model keep boundary
        if args.include_regression:
            l2_distant = torch.sum((lmksGT/256.0 - lmksPRED/256.0) * (lmksGT/256.0 - lmksPRED/256.0), axis=1)
            loss = loss + l2_distant


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        print(f"Epoch:{epoch}. Lr:{optimizer.param_groups[0]['lr']} Batch {i} / {num_batch} batches. Loss: {loss.item()}")

    return losses.avg

    

def validate(valdataloader, model, optimizer, epoch, args, tensorboardLogger=None,tag="300VW"):
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

        if batch>=num_vis_batch:
            break

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
        x_ok = torch.logical_and(lmksGT[:,:,0] >= 0,  lmksGT[:,:,0] <= 1)
        y_ok = torch.logical_and(lmksGT[:,:,1] >= 0,  lmksGT[:,:,1] <= 1)
        y_true_visible_mask = torch.logical_and(x_ok, y_ok)
        y_true_visible_mask = torch.unsqueeze(y_true_visible_mask, -1)
        y_true_visible_mask = torch.unsqueeze(y_true_visible_mask, -1)
        lmksGT = lmksGT * 256  
        
        # Generate GT heatmap by randomized rounding
        # print(lmksGT.shape)
        heatGT = lmks2heatmap(lmksGT, args.random_round, args.random_round_with_gaussian)  

        # Inference model to generate heatmap
        heatPRED, lmksPRED, lmks_regression, lmks_regression_end = model(img.to(device))

    
        heatPRED = heatmap2sigmoidheatmap(heatPRED.to('cpu'))
        loss = adaptive_wing_loss(heatPRED, heatGT, y_true_visible_mask=y_true_visible_mask)

    
        
        # Loss
        loss = loss_heatmap(heatPRED, heatGT)

        if batch < num_vis_batch:
            vis_prediction_batch(batch, img_ori, lmksPRED, output=args.vis_dir)


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

    tensorboardLogger.log(f"val/loss/{tag}", losses.avg, epoch)
    tensorboardLogger.log(f"val/nme_ocular/{tag}", np.mean(nme_interocular), epoch)

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
    tensorboardLogger = TensorBoardLogger(root="runs", experiment_name=args.snapshot)

    # Init model
    model = HeatMapLandmarker(pretrained=True, model_url="https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1", usehrnet18=args.use_hrnet18)
    
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['plfd_backbone'], strict=False)
        model.to(device)
    
    model.to(device)

  

    # # Train dataset, valid dataset
    # train_dataset = LAPA106DataSet(img_dir=f'{args.dataroot}/images', anno_dir=f'{args.dataroot}/landmarks', augment=True,
    # transforms=transform)
    # val_dataset = LAPA106DataSet(img_dir=f'{args.val_dataroot}/images', anno_dir=f'{args.val_dataroot}/landmarks')

    # # Dataloader
    # traindataloader = DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batchsize,
    #     shuffle=True,
    #     num_workers=2,
    #     drop_last=True)

    
    # validdataloader = DataLoader(
    #     val_dataset,
    #     batch_size=args.val_batchsize,
    #     shuffle=False,
    #     num_workers=2,
    #     drop_last=True)

    train_300VW = VW300(img_dir=args.vw300_datadir,
                        anno_dir=args.vw300_annotdir,
                        augment=args.do_train_augment,
                        imgsize=args.imgsize,
                        transforms=transform, set_type="train")
    
    val_300VW = VW300(img_dir=args.vw300_datadir,
                        anno_dir=args.vw300_annotdir,
                        augment=False,
                        imgsize=args.imgsize,
                        transforms=None, set_type="val")

    train_style = W300Style(img_dir=args.style_datadir,
                    anno_dir=args.style_datadir,
                    augment=args.do_train_augment,
                        imgsize=args.imgsize,
                    transforms=transform, set_type="train")
    
    val_style = W300Style(img_dir=args.style_datadir,
                    anno_dir=args.style_datadir,
                    augment=False,
                        imgsize=args.imgsize,
                    transforms=None, set_type="val")
    
    
    train_lp = W300LargePose(img_dir=args.lp_datadir,
                    anno_dir=args.lp_datadir,
                    augment=args.do_train_augment,
                        imgsize=args.imgsize,
                    transforms=transform, set_type="train")
    
    val_lp = W300LargePose(img_dir=args.lp_datadir,
                    anno_dir=args.lp_datadir,
                    augment=False,
                        imgsize=args.imgsize,
                    transforms=None, set_type="val")

    train_mask = DatasetMask(img_dir=args.mask_datadir,
            anno_dir=args.mask_datadir,
            augment=True,
            transforms=transform, set_type="train")

    val_mask = DatasetMask(img_dir=args.mask_datadir,
            anno_dir=args.mask_datadir,
            augment=False,
            transforms=None, set_type="val")

    print(f'Len train LP :{len(train_lp)}. len val LP :{len(val_lp)}')

    # Dataloader
    traindataloader_300VW = DataLoader(
        train_300VW,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=8,
        drop_last=True)
    
    validdataloader_300VW = DataLoader(
        val_300VW,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=2,
        drop_last=True)
    
    traindataloader_style = DataLoader(
        train_style,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=8,
        drop_last=True)
    
    validdataloader_style = DataLoader(
        val_style,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    
    traindataloader_lp = DataLoader(
        train_lp,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=8,
        drop_last=True)
    
    validdataloader_lp = DataLoader(
        val_lp,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    
    traindataloader_mask = DataLoader(
        train_mask,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=8,
        drop_last=True)
    
    validdataloader_mask = DataLoader(
        val_mask,
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
        loaders = []
        tags = []
        if args.include_300vw:
            loaders.append(traindataloader_300VW)
            tags.append("300VW")
        if args.include_style:
            loaders.append(traindataloader_style)
            tags.append("Style")
        if args.include_lp:
            loaders.append(traindataloader_lp)
            tags.append("LP")
        
        if args.include_mask:
            loaders.append(traindataloader_mask)
            tags.append("Mask")

        for epoch in range(300):
            # train_one_epoch(traindataloader, model, optimizer, epoch, args)
            # validate(validdataloader, model, optimizer, epoch, args)
            train_shuffle_cross_data(loaders, model, optimizer, epoch, args, tensorboardLogger, tags=tags)


            validate(validdataloader_300VW, model, optimizer, epoch, args, tensorboardLogger, tag="300VW")
            validate(validdataloader_style, model, optimizer, epoch, args, tensorboardLogger, tag="Style")
            validate(validdataloader_lp, model, optimizer, epoch, args, tensorboardLogger, tag="LP")
            validate(validdataloader_mask, model, optimizer, epoch, args, tensorboardLogger, tag="Mask")

            

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
    parser.add_argument('--random_round_with_gaussian', default=1, type=int)
    parser.add_argument('--mode', default='train', type=str)

    # 
    parser.add_argument('--vis_dir', default="./vis", type=str)
    parser.add_argument('--vw300_datadir', default="", type=str)
    parser.add_argument('--vw300_annotdir', default="", type=str)
    parser.add_argument('--lp_datadir', default="", type=str)
    parser.add_argument('--mask_datadir', default="", type=str)
    parser.add_argument('--style_datadir', default="", type=str)
    parser.add_argument('--include_300vw', default=1, type=int)
    parser.add_argument('--include_style', default=0, type=int)
    parser.add_argument('--include_lp', default=0, type=int)
    parser.add_argument('--include_mask', default=0, type=int)
    parser.add_argument('--sampling_data', default=1, type=int)
    parser.add_argument('--do_train_augment', default=1, type=int)
    parser.add_argument('--curriculum', default=0, type=int)
    parser.add_argument('--imgsize', default=256, type=int)
    parser.add_argument('--use_visible_mask', default=1, type=int)
    parser.add_argument('--use_hrnet18', default=0, type=int)
    parser.add_argument('--include_regression', default=0, type=int)
    parser.add_argument('--include_regression_end', default=0, type=int)



    






    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

          