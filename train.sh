# CUDA_VISIBLE_DEVICES=0  taskset --cpu-list 0-20  python3 train.py --snapshot ckpt_entropy_weight_gaussian\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.0005\
#                                                 --step_size 10\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --resume ckpt_entropy_weight_gaussian/epoch_31.pth.tar



# CUDA_VISIBLE_DEVICES=0  taskset --cpu-list 21-40  python3 train.py --snapshot ckpt_entropy_weight_gaussian_more_augment\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.0005\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --resume ckpt_entropy_weight_gaussian/epoch_31.pth.tar


# CUDA_VISIBLE_DEVICES=0  taskset --cpu-list 41-60  python3 train.py --snapshot ckpt_entropy_weight_gaussian_more_augment_noLP\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.0005\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --resume ckpt_entropy_weight_gaussian/epoch_31.pth.tar



############################################### Experiments 17/07/2021 #####################################################3
### Old setup. Truncated points ---> Force to boundary --> Use loss on that
# CUDA_VISIBLE_DEVICES=0  taskset --cpu-list 0-20  python3 train.py --snapshot 17072021_force_truncated_to_boundary\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --resume ckpt_entropy_weight_gaussian/epoch_31.pth.tar\
#                                                 --use_visible_mask 0

### Old setup. Using HRNET18 instead 
# CUDA_VISIBLE_DEVICES=1  taskset --cpu-list 21-40  python3 train.py --snapshot 17072021_force_truncated_to_boundary_hrnet18\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 40\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --use_visible_mask 0\
#                                                 --use_hrnet18 1



### Old setup. With boundary information
# CUDA_VISIBLE_DEVICES=2  taskset --cpu-list 41-60  python3 train.py --snapshot 17072021_force_truncated_to_boundary_include_regression\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 1 \
#                                                 --use_hrnet18 0 \
#                                                 --resume ckpt_entropy_weight_gaussian/epoch_31.pth.tar\


####################################3 Experiments on 19/07/2021 ###############################################
## Force to boundary and use only regression loss
## This experiment want to test if the loss matter (regresison loss vs adaptive wingloss). 

########################## -------> Does not work --> end!!

# CUDA_VISIBLE_DEVICES=0  taskset --cpu-list 0-20  python3 train.py --snapshot 19072021_force_truncated_to_boundary_onlyregressionloss\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 1 \
#                                                 --use_hrnet18 0 \
#                                                 --resume ckpt_entropy_weight_gaussian/epoch_31.pth.tar\


## Train with mask augmentation
# CUDA_VISIBLE_DEVICES=0  taskset --cpu-list 21-40  python3 train.py --snapshot 19072021_force_truncated_to_boundary_train_with_mask\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --mask_datadir /home/ubuntu/vuthede/random_samples_20k_masked\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --include_mask 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 0 \
#                                                 --use_hrnet18 0 \
#                                                 --resume 17072021_force_truncated_to_boundary/epoch_112.pth.tar\



# CUDA_VISIBLE_DEVICES=1  taskset --cpu-list 41-60  python3 train.py --snapshot 29072021_force_truncated_to_boundary_train_with_mask_hrnet18\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --mask_datadir /home/ubuntu/vuthede/random_samples_20k_masked\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --include_mask 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 0 \
#                                                 --use_hrnet18 1 \
#                                                 --resume 17072021_force_truncated_to_boundary_hrnet18/epoch_45.pth.tar\

######################## Experiemnt 21/07/2021
# CUDA_VISIBLE_DEVICES=0  taskset --cpu-list 21-40  python3 train.py --snapshot 21072021_force_truncated_to_boundary_train_with_mask_hrnet18_with_regression\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --mask_datadir /home/ubuntu/vuthede/random_samples_20k_masked\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --include_mask 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 1 \
#                                                 --use_hrnet18 1 \
#                                                 --resume 17072021_force_truncated_to_boundary_hrnet18/epoch_45.pth.tar\

# CUDA_VISIBLE_DEVICES=6  taskset --cpu-list 0-20  python3 train.py --snapshot 21072021_force_truncated_to_boundary_train_with_mask_hrnet18_with_regressionend\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --mask_datadir /home/ubuntu/vuthede/random_samples_20k_masked\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --include_mask 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 0 \
#                                                 --include_regression_end 1 \
#                                                 --use_hrnet18 1 \
#                                                 --resume 17072021_force_truncated_to_boundary_hrnet18/epoch_45.pth.tar\


################ experiment 29072021#####################

# CUDA_VISIBLE_DEVICES=1  taskset --cpu-list 21-40  python3 train.py --snapshot 29072021_force_truncated_to_boundary_train_with_mask_hrnet18_fixmask\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --mask_datadir /home/ubuntu/vuthede/random_samples_20k_masked\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --include_mask 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 0 \
#                                                 --use_hrnet18 1 \
#                                                 --resume 17072021_force_truncated_to_boundary_hrnet18/epoch_45.pth.tar\

# CUDA_VISIBLE_DEVICES=2  taskset --cpu-list 41-60  python3 train.py --snapshot 29072021_force_truncated_to_boundary_train_with_mask_hrnet18_with_regression_fixmask\
#                                                 --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
#                                                 --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkV2/300W-Convert\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkV2/300W-LP\
#                                                 --mask_datadir /home/ubuntu/vuthede/random_samples_20k_masked\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --include_mask 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 1 \
#                                                 --use_hrnet18 1 \
#                                                 --resume 17072021_force_truncated_to_boundary_hrnet18/epoch_45.pth.tar\

##################### Experiments on 31/11/2021 ###########################
# rm -rf runs/ckpt_31112021_heatmap_mobile_onlyeyeregion
# CUDA_VISIBLE_DEVICES=2  taskset --cpu-list 30-40  python3 train.py --snapshot ckpt_31112021_heatmap_mobile_onlyeyeregion\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --train_batchsize 16\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment_noLP\
#                                                 --imgsize 256\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkerData/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkerData/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkerData/cropstyle\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkerData/landmarkV2/300W-LP\
#                                                 --dataset_mask /home/ubuntu/vuthede/landmarkerData/random_samples_20k_masked\
#                                                 --dataset_vinai /home/ubuntu/vuthede/landmarkerData/delivery_17082021\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --include_mask 1\
#                                                 --include_vinai 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 0 \
#                                                 --use_hrnet18 0 \
#                                                 --consin_lr_scheduler 1\
#                                                 --max_num_epoch 80\
#                                                 --num_batch_per_dataset 615\
#                                                 --resume 17072021_force_truncated_to_boundary/epoch_112.pth.tar\
#                                                 --sampling_data 1\
#                                                 --only_eyeregion 1\

# rm -rf runs/ckpt_13122021_heatmap_mobile_onlyeyeregion
# CUDA_VISIBLE_DEVICES=3  taskset --cpu-list 60-70  python3 train.py --snapshot ckpt_13122021_heatmap_mobile_onlyeyeregion\
#                                                 --get_topk_in_pred_heats_training 0\
#                                                 --train_batchsize 16\
#                                                 --lr 0.001\
#                                                 --step_size 20\
#                                                 --gamma 0.5\
#                                                 --random_round_with_gaussian 1\
#                                                 --mode train\
#                                                 --vis_dir vis_68lmks_moreaugment\
#                                                 --imgsize 256\
#                                                 --vw300_datadir /home/ubuntu/vuthede/landmarkerData/landmarkV2/300VW_frames\
#                                                 --vw300_annotdir /home/ubuntu/vuthede/landmarkerData/landmarkV2/300VW_Dataset_2015_12_14\
#                                                 --style_datadir /home/ubuntu/vuthede/landmarkerData/cropstyle\
#                                                 --lp_datadir /home/ubuntu/vuthede/landmarkerData/landmarkV2/300W-LP\
#                                                 --dataset_mask /home/ubuntu/vuthede/landmarkerData/random_samples_20k_masked\
#                                                 --dataset_vinai /home/ubuntu/vuthede/landmarkerData/delivery_17082021\
#                                                 --include_300vw 1\
#                                                 --include_style 1\
#                                                 --include_lp 1\
#                                                 --include_mask 1\
#                                                 --include_vinai 1\
#                                                 --use_visible_mask 0\
#                                                 --include_regression 0 \
#                                                 --use_hrnet18 0 \
#                                                 --consin_lr_scheduler 1\
#                                                 --max_num_epoch 80\
#                                                 --num_batch_per_dataset 615\
#                                                 --sampling_data 1\
#                                                 --only_eyeregion 1\

##############Base code for anh Toan to start #########
rm -rf runs/ckpt_29122021_heatmap_mobile
CUDA_VISIBLE_DEVICES=3  taskset --cpu-list 60-70  python3 train.py --snapshot ckpt_29122021_heatmap_mobile\
                                                --get_topk_in_pred_heats_training 0\
                                                --train_batchsize 16\
                                                --lr 0.001\
                                                --step_size 20\
                                                --gamma 0.5\
                                                --random_round_with_gaussian 1\
                                                --mode train\
                                                --vis_dir vis_68lmks_moreaugment\
                                                --imgsize 256\
                                                --vw300_datadir /home/ubuntu/vuthede/landmarkerData/landmarkV2/300VW_frames\
                                                --vw300_annotdir /home/ubuntu/vuthede/landmarkerData/landmarkV2/300VW_Dataset_2015_12_14\
                                                --style_datadir /home/ubuntu/vuthede/landmarkerData/cropstyle\
                                                --lp_datadir /home/ubuntu/vuthede/landmarkerData/landmarkV2/300W-LP\
                                                --dataset_mask /home/ubuntu/vuthede/landmarkerData/random_samples_20k_masked\
                                                --dataset_vinai /home/ubuntu/vuthede/landmarkerData/delivery_17082021\
                                                --include_300vw 1\
                                                --include_style 1\
                                                --include_lp 1\
                                                --include_mask 1\
                                                --include_vinai 1\
                                                --use_visible_mask 0\
                                                --include_regression 0 \
                                                --use_hrnet18 0 \
                                                --consin_lr_scheduler 1\
                                                --max_num_epoch 80\
                                                --num_batch_per_dataset 615\
                                                --sampling_data 1\
                                                --only_eyeregion 0\
                                                # --resume ckpt_68lmks/epoch_17.pth.tar