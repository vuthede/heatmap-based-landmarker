CUDA_VISIBLE_DEVICES=0 python3 train.py --snapshot ckpt_first_version\
                                        --dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/train\
                                        --val_dataroot /media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/LaPa/val\
                                        --get_topk_in_pred_heats_training 0\
                                        --lr 0.001\
                                        --step_size 15\
                                        --gamma 0.5\
                                        --random_round 1\
                                        --pos_weight 1\
                                        --mode train
