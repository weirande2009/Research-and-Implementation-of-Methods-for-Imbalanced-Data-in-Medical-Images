import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import wandb
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.metrics import IoU_and_AP, mIoU_and_mAP
from utils.utils import *


def evaluate(net, dataloader, device, val_count, save_file):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    dice_score_list = torch.zeros([net.n_classes])
    mIoU_score = 0
    mIoU_score_list = torch.zeros([net.n_classes])
    mAP_score = 0
    mAP_score_list = torch.zeros([net.n_classes])
    Recall_score = 0
    Recall_score_list = torch.zeros([net.n_classes])
    # iterate over the validation set
    for i, batch in enumerate(dataloader):
        images = batch['image']
        mask_true = batch['mask']
        names = batch['name']
        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            masks_pred = net(images)

            masks_pred = torch.softmax(masks_pred, dim=1).float()

            # compute the Dice score, ignoring background
            dice, dice_list = multiclass_dice_coeff(F.one_hot(torch.argmax(masks_pred, dim=1), net.n_classes).permute(0, 3, 1, 2).float(),
                                                    F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(), 
                                                    reduce_batch_first=False)
            dice_score += dice
            dice_score_list += dice_list

            # compute mIoU and mAP
            mIoU, mIoU_list, mAP, mAP_list, Recall, Recall_list = mIoU_and_mAP(masks_pred, mask_true)
            mIoU_score += mIoU
            mIoU_score_list += mIoU_list
            mAP_score += mAP
            mAP_score_list += mAP_list
            Recall_score += Recall
            Recall_score_list += Recall_list

    net.train()
    return dice_score / num_val_batches, \
           dice_score_list / num_val_batches, \
           mIoU_score / num_val_batches, \
           mIoU_score_list / num_val_batches, \
           mAP_score / num_val_batches, \
           mAP_score_list / num_val_batches, \
           Recall_score / num_val_batches, \
           Recall_score_list / num_val_batches

def evaluate_binary(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    mIoU_score = 0
    mAP_score = 0
    precision_list = torch.zeros(2, device=device)
    recall_list = torch.zeros(2, device=device)
    # iterate over the validation set
    for i, batch in enumerate(dataloader):
    # for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        images = batch['image']
        mask_true = batch['binary_mask']
        names = batch['name']
        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            masks_pred = net(images)

            masks_pred = torch.sigmoid(masks_pred).float().reshape([masks_pred.shape[0], masks_pred.shape[2], masks_pred.shape[3]])

            masks_pred[masks_pred > 0.5] = 1
            masks_pred[masks_pred <= 0.5] = 0

            # compute the Dice score, ignoring background
            dice = dice_coeff(masks_pred, mask_true.float())
            dice_score += dice

            # compute mIoU and mAP
            mIoU, mAP, precision, recall = IoU_and_AP(masks_pred, mask_true)
            mIoU_score += mIoU
            mAP_score += mAP
            precision_list += precision
            recall_list += recall

    net.train()
    precision_list = (precision_list / num_val_batches).cpu().numpy()
    recall_list = (recall_list / num_val_batches).cpu().numpy()
    return dice_score / num_val_batches, \
           mIoU_score / num_val_batches, \
           mAP_score / num_val_batches, \
           precision_list, \
           recall_list



