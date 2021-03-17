CUDA_VISIBLE_DEVICES=0 python3 train.py --snapshot ckpt_first_version\
                                        --dataroot \
                                        --val_dataroot \
                                        --get_topk_in_pred_heats_training 0
