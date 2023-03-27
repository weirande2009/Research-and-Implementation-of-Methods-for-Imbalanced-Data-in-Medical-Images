import argparse
import datetime
import logging
import sys
import os
import shutil
from pathlib import Path

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
from evaluate import evaluate
from unet import *
from model.deeplabv3 import *
from predict import predict

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
    my_loss = MyLoss(args, dataset, device, net.n_classes)
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
            true_masks = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=args.amp):
                masks_pred = net(images)
                loss, basic_loss, tversky_loss, focal_loss, dice_loss, ob_loss = my_loss(masks_pred, true_masks)
            
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()

            pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation
        dice_score, dice_score_list, miou_score, miou_score_list, map_score, map_score_list, recall_score, recall_score_list = evaluate(net, val_loader, device, validation_times, save_file)
        validation_times += 1
        scheduler.step(dice_score)
        
        if args.save:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(net.state_dict(), os.path.join(save_file, 'best.pth'))
                f = open(os.path.join(save_file, 'result.txt'), "w")
                f.write("Loss: %.4f\n" % best_loss)
                f.write("Dice score: %.4f\n" % dice_score)
                f.write("mIoU score: %.4f\n" % miou_score_list.mean().item())
                f.write("mAP score: %.4f\n" % map_score_list.mean().item())
                f.write("Recall score: %.4f\n" % recall_score_list.mean().item())
                f.close()

        pbar.set_description("mIoU: %.4f" % miou_score_list.mean().item())

        pbar.update()
    pbar.close()

    # Predicting
    if args.predict:
        predict(net, val_set, device, args.batch_size, save_file)
        print("Finish predicting")
        print("Save predictions in " + save_file)


def train_multi(args, save_path):
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset
    if args.dataset == "CoNIC":
        dataset = CoNICDataset(focal_weights_kind=args.focal_weight)
        n_classes = 7
    elif args.dataset == "MoNuSAC":
        dataset = MoNuSACDataset(focal_weights_kind=args.focal_weight)
        n_classes = 5
    
    # Model
    if args.m_net == "unet":
        net = UNet(n_channels=3, n_classes=n_classes, bilinear=False)
    elif args.m_net == "unet++":
        net = NestedUNet(in_ch=3, out_ch=n_classes)
    elif args.m_net == "unet+++":
        net = UNet_3Plus(in_channels=3, n_classes=n_classes)
    elif args.m_net == "attunet":
        net = AttU_Net(img_ch=3, output_ch=n_classes)
    elif args.m_net == "deeplabv3":
        net = DeepLabV3(n_channels=3, n_classes=n_classes)

    # Optimizer
    if args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=1e-8)

    net = net.to(device)

    # Training    
    try:
        train_net(args, net, dataset, optimizer, device, save_path)
    except KeyboardInterrupt:
        sys.exit(0)




                
            




