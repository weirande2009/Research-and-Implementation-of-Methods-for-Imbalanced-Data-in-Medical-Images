import argparse
import datetime
import logging
import random
import sys
import os
import shutil
from pathlib import Path
from PIL import Image


import cv2
import numpy as np
from sklearn.metrics import recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import *
# from utils.dice_score import dice_loss
from utils.utils import *
from evaluate import *
from unet import *
from model.deeplabv3 import *
from utils.loss import *

start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def predict(net, dataset, device, batch_size, save_path):
    random.seed(1)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, pin_memory=True)
    # Begin predicting
    net.eval()
    pbar = tqdm(total=len(dataloader), desc="Predicting")
    for i, batch in enumerate(dataloader):
        images = batch['image']
        true_masks = batch['mask']
        names = batch['name']
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        with torch.no_grad():
            masks_pred = net(images)
            for i in range(batch_size):
                masks_result = torch.softmax(masks_pred, dim=1)[i].float().cpu()
                masks_result = torch.argmax(masks_result, dim=0).float()
                # Set Background
                colored_mask = mask_to_color(masks_result.numpy())
                colored_mask = colored_mask.astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, "predictions/color", names[i]+".png"), colored_mask)
                cv2.imwrite(os.path.join(save_path, "predictions/raw", names[i]+".png"), masks_result.numpy())
        pbar.update()
    pbar.close()
                

def predict_binary(net, dataset, device, batch_size, save_path):
    random.seed(1)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, pin_memory=True)
    # Begin predicting
    net.eval()
    pbar = tqdm(total=len(dataloader), desc="Predicting")
    for i, batch in enumerate(dataloader):
        images = batch['image']
        mask_true = batch['binary_mask']
        names = batch['name']
        images = images.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            masks_pred = net(images)
            for i in range(batch_size):
                masks_result = torch.sigmoid(masks_pred)[i].float().cpu()
                masks_result = masks_result.reshape([*masks_result.shape[1:]]).float()
                masks_result[masks_result > 0.5] = 255
                masks_result[masks_result <= 0.5] = 0
                cv2.imwrite(os.path.join(save_path, "predictions", names[i]+".png"), masks_result.numpy())
        pbar.update()
    pbar.close()

def predict_from_both(args, predict_mask_folder, predict_binary_mask_folder, save_file):
    # Set dataset
    dataset = args.dataset
    if dataset == "MoNuSAC":
        n_classes = 5
    elif dataset == "CoNIC":
        n_classes = 7
    image_folder = os.path.join("../Datasets", dataset, "data", "images")
    color_mask_folder = os.path.join("../Datasets", dataset, "data", "color_masks")
    mask_folder = os.path.join("../Datasets", dataset, "data", "masks")
    # Predicting
    dice_score = 0
    dice_score_list = torch.zeros([n_classes])
    mIoU_score = 0
    mIoU_score_list = torch.zeros([n_classes])
    mAP_score = 0
    mAP_score_list = torch.zeros([n_classes])
    Recall_score = 0
    Recall_score_list = torch.zeros([n_classes])
    num_val_batches = 0
    for filepath, dirnames, filenames in os.walk(os.path.join(predict_mask_folder, "raw")):
        pbar = tqdm(total=len(filenames))
        num_val_batches = len(filenames)
        for filename in filenames:
            predict_color_mask = np.asarray(Image.open(os.path.join(predict_mask_folder, "color", filename)))
            predict_mask = torch.as_tensor(np.asarray(Image.open(os.path.join(predict_mask_folder, "raw", filename))).copy())
            predict_binary_mask = torch.as_tensor(np.asarray(Image.open(os.path.join(predict_binary_mask_folder, filename))).copy())
            combined_mask = predict_mask.clone()
            combined_mask[predict_binary_mask == 0] = 0
            image = np.asarray(Image.open(os.path.join(image_folder, filename)))
            color_mask = np.asarray(Image.open(os.path.join(color_mask_folder, filename)))
            mask = torch.as_tensor(np.asarray(Image.open(os.path.join(mask_folder, filename))).copy())
            resized_mask = torch.as_tensor(np.asarray(Image.open(os.path.join(mask_folder, filename)).resize(predict_mask.shape)).copy())
            # evaluate
            dice, dice_list = multiclass_dice_coeff(F.one_hot(combined_mask.reshape([1, *combined_mask.shape]).to(torch.int64), n_classes).permute(0, 3, 1, 2),
                                                    F.one_hot(resized_mask.reshape([1, *resized_mask.shape]).to(torch.int64), n_classes).permute(0, 3, 1, 2), 
                                                    reduce_batch_first=False)
            dice_score += dice
            dice_score_list += dice_list

            # compute mIoU and mAP
            mIoU, mIoU_list, mAP, mAP_list, Recall, Recall_list = mIoU_and_mAP(F.one_hot(predict_mask.reshape([1, *predict_mask.shape]).to(torch.int64), n_classes).permute(0, 3, 1, 2), 
                                                          resized_mask.reshape([1, *resized_mask.shape]))
            mIoU_score += mIoU
            mIoU_score_list += mIoU_list
            mAP_score += mAP
            mAP_score_list += mAP_list
            Recall_score += Recall
            Recall_score_list += Recall_list

            combined_mask = cv2.resize(combined_mask.numpy(), (image.shape[1], image.shape[0]))

            cv2.imwrite(os.path.join(save_file, "predictions", filename), np.array(mask_to_color(combined_mask), dtype=np.uint8))
            pbar.update()
        pbar.close()
        break

    dice_score = dice_score / num_val_batches
    mIoU_score = mIoU_score / num_val_batches
    mAP_score = mAP_score / num_val_batches
    Recall_score = Recall_score / num_val_batches
    dice_score_list = dice_score_list / num_val_batches
    mIoU_score_list = mIoU_score_list / num_val_batches
    mAP_score_list = mAP_score_list / num_val_batches
    Recall_score_list = Recall_score_list / num_val_batches

    f = open(os.path.join(save_file, 'result.txt'), "w")
    f.write("Dice score: %.4f\n" % dice_score)
    f.write("mIoU score: %.4f\n" % mIoU_score_list.mean().item())
    f.write("mAP score: %.4f\n" % mAP_score_list.mean().item())
    f.write("Recall score: %.4f\n" % Recall_score_list.mean().item())
    f.close()

    print("Save predictions in " + save_file)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--dataset', '-d', type=str, default="MoNuSAC", help='Dataset Kind: BCSS, CoNIC, MoNuSAC')
    parser.add_argument('--name', '-n', type=str, default="myname", help='')
    parser.add_argument('--binary_mask', '-b', type=str, default="results/MoNuSAC/binary/u focal dice Adam fw_5 hn_512/predictions/", help='Dataset Kind: BCSS, CoNIC, MoNuSAC')
    parser.add_argument('--mask', '-m', type=str, default="results/MoNuSAC/normal/attunet dice/predictions", help='Dataset Kind: BCSS, CoNIC, MoNuSAC')
    parser.add_argument('--save-path', '-m', type=str, default="results/MoNuSAC/normal/attunet dice/predictions", help='Dataset Kind: BCSS, CoNIC, MoNuSAC')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    predict_from_both(args)




                
            




