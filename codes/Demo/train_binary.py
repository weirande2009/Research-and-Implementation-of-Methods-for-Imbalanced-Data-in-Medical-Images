import argparse
import datetime
import logging
import sys
import os
import shutil
from pathlib import Path
from tkinter import E

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import *
from utils.utils import *
from utils.loss import *
from evaluate import *
from unet import *
from model.deeplabv3 import *
from predict import *

start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def train_net(args, net, dataset, optimizer, device, save_file):
    # Split dataset into train / validation partitions
    n_val = int(len(dataset) * args.val)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    my_loss = MyBinaryLoss(args, dataset, device)
    global_step = 0
    best_loss = 10000000
    validation_times = 0

    # Begin training
    pbar = tqdm(total=args.epochs)
    for epoch in range(args.epochs):
        net.train()
        epoch_loss = 0
        for batch in train_loader:
            images = batch['image']
            binary_masks = batch['binary_mask']
            true_masks = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            # true_masks = true_masks.to(device=device, dtype=torch.long)
            binary_masks = binary_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=args.amp):
                masks_pred = net(images)
                masks_pred = masks_pred.reshape([masks_pred.shape[0], masks_pred.shape[2], masks_pred.shape[3]])

                loss, focal_loss, dice_loss, ob_loss = my_loss(masks_pred, binary_masks)
            
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            # pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()

            pbar.set_postfix(**{'batch loss': loss.item()})
        # Evaluation round
        dice_score, miou_score, map_score, precision_list, recall_list = evaluate_binary(net, val_loader, device)
        validation_times += 1
        scheduler.step(dice_score)
            
        if args.save:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(net.state_dict(), os.path.join(save_file, 'best.pth'))
                f = open(os.path.join(save_file, 'result.txt'), "w")
                f.write("Loss: %.4f\n" % best_loss)
                f.write("Dice score: %.4f\n" % dice_score)
                f.write("mIoU score: %.4f\n" % miou_score)
                f.write("mAP score: %.4f\n" % precision_list.mean().item())
                f.write("Recall score: %.4f\n" % recall_list.mean().item())
                f.close()

        pbar.update()
    pbar.close()

    # Predicting
    if args.predict:
        predict_binary(net, dataset, device, args.batch_size, save_file)
        print("Finish predicting")
        print("Save predictions in " + save_file)


def train_binary(args, save_path):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset
    n_classes = 1
    if args.dataset == "CoNIC":
        dataset = CoNICDataset(focal_weights_kind=args.focal_weight)
    elif args.dataset == "MoNuSAC":
        dataset = MoNuSACDataset(focal_weights_kind=args.focal_weight)

    # Model
    if args.b_net == "unet":
        net = UNet(n_channels=3, n_classes=n_classes, bilinear=False)
    elif args.b_net == "unet++":
        net = NestedUNet(in_ch=3, out_ch=n_classes)
    elif args.b_net == "unet+++":
        net = UNet_3Plus(in_channels=3, n_classes=n_classes)
    elif args.b_net == "attunet":
        net = AttU_Net(img_ch=3, output_ch=n_classes)
    elif args.b_net == "deeplabv3":
        net = DeepLabV3(n_channels=3, n_classes=n_classes)
    net = net.to(device)

    # Optimizer
    if args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=1e-8)
        
    # Training    
    try:
        train_net(args, net, dataset, optimizer, device, save_path)
    except KeyboardInterrupt:
        sys.exit(0)

                
            




