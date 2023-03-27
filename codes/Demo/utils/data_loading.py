import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.utils import class_weight
from utils.boundary_loss_utils import *


class BasicDataset(Dataset):
    def __init__(self, dataset_dir: str, scale: int = 512, n_classes:int = 0, boundary: bool = False, binary_masks: bool = False, focal_weights_kind=1):
        self.image_dir = os.path.join(dataset_dir, "images")
        self.mask_dir = os.path.join(dataset_dir, "masks")
        self.color_mask_dir = os.path.join(dataset_dir, "color_masks")
        self.binary_mask_dir = os.path.join(dataset_dir, "binary_masks")
        self.dataset_dir = dataset_dir
        self.scale = scale
        self.boundary = boundary
        self.n_classes = n_classes
        self.binary_masks = binary_masks
        self.focal_weights_kind = focal_weights_kind

        self.ids = [splitext(file)[0] for file in listdir(self.mask_dir) if not file.startswith('.')]
        # self.mask_ids = [splitext(file)[0] for file in listdir(self.mask_dir) if not file.startswith('.')]

        # assert len(self.ids) == len(self.mask_ids) , 'The number of images should match the number of masks'

        if not self.ids:
            raise RuntimeError(f'No input file found in {self.image_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.weights = self.set_label_weights()

    def set_label_weights(self):
        f = open(os.path.join(self.dataset_dir, "count.txt"), "r")
        weights = f.read().split(" ")[:-1]
        for i in range(len(weights)):
            weights[i] = float(weights[i])
        weights = torch.tensor(weights, dtype=torch.float32)

        # sklearn.utils.class_weight
        if self.focal_weights_kind == 0:
            weights = torch.ones([weights.shape[0]], dtype=torch.float32)
        elif self.focal_weights_kind == 1:
            weights = weights*100000
            weights = weights.to(dtype=torch.int)
            labels = []
            for i in range(weights.shape[0]):
                labels += [i] * weights[i].item()
            classes = [i for i in range(weights.shape[0])]
            weights = class_weight.compute_class_weight("balanced", classes=classes, y=labels)
            weights = torch.from_numpy(weights)
        elif self.focal_weights_kind == 2:
            weights = torch.exp(weights*torch.exp(torch.tensor([1], dtype=torch.float)))
        elif self.focal_weights_kind == 3:
            weights_fg = weights[1:] / torch.sum(weights[1:])
            weights_fg = torch.exp(-weights_fg*torch.exp(torch.tensor(1, dtype=torch.float)))
            weights = torch.exp(-weights*torch.exp(torch.tensor(1, dtype=torch.float)))
            weights[1:] = weights_fg
        elif self.focal_weights_kind == 4:
            weights_fg = weights[1:] / torch.sum(weights[1:])
            computed_weight = torch.exp(-weights_fg*torch.exp(torch.tensor(1, dtype=torch.float)))
            weights[1:] = computed_weight / torch.sum(computed_weight)
            weights[0] = 0
        elif self.focal_weights_kind == 5:
            weights_bg = 1 - weights[0]
            weights_fg = 1 - weights_bg
            weights = torch.tensor([weights_bg, weights_fg])

        
        # version 1
        # weights = torch.exp(weights*torch.exp(torch.tensor([1], dtype=torch.float)))
        # version 2
        # weights = torch.exp(-weights*torch.exp(torch.tensor([1], dtype=torch.float)))

        # version 3(without bg)
        # weights_fg = weights[1:] / torch.sum(weights[1:])
        # computed_weight = torch.exp(-weights_fg*torch.exp(torch.tensor(1, dtype=torch.float)))
        # weights[1:] = computed_weight / torch.sum(computed_weight)
        # weights[0] = 0

        # version 4
        # weights_fg = weights[1:] / torch.sum(weights[1:])
        # weights_fg = torch.exp(-weights_fg*torch.exp(torch.tensor(1, dtype=torch.float)))
        # weights = torch.exp(-weights*torch.exp(torch.tensor(1, dtype=torch.float)))
        # weights[1:] = weights_fg
        # weights = weights / torch.sum(weights)

        # version for binary mask
        # weights_bg = 1 - weights[0]
        # weights_fg = 1 - weights_bg
        # weights = torch.tensor([weights_bg, weights_fg])

        return weights / torch.sum(weights)


    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img, is_mask, is_binary=False):
        if self.scale != 0:
            pil_img = pil_img.resize((self.scale, self.scale))
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255
        
        if is_binary and is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = os.path.join(self.mask_dir, name+".png")
        img_file = os.path.join(self.image_dir, name+".png")
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        # boundary 计算
        if self.boundary:
            dist = torch.from_numpy(mask.reshape([1, *mask.shape]).copy())
            dist = class2one_hot(dist, self.n_classes)
            # print(data2)
            dist = dist[0].numpy()
            dist = one_hot2dist(dist)   #bcwh
            # dist = dist / np.max(dist)

        if self.binary_masks:
            binary_mask_file = os.path.join(self.binary_mask_dir, name+".png")
            binary_mask = Image.open(binary_mask_file)
            binary_mask = self.preprocess(binary_mask, is_mask=True, is_binary=self.binary_masks)


        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'binary_mask': torch.as_tensor(binary_mask.copy()).long().contiguous() if self.binary_masks else torch.zeros([1]),
            'name': name
        }

class BCSSDDataset(BasicDataset):
    DIR = "../Datasets/BCSS/data"
    SCALE = 0
    CLASSES = 22
    def __init__(self):
        super().__init__(self.DIR, self.SCALE, self.CLASSES)

class CoNICDataset(BasicDataset):
    DIR = "../Datasets/CoNIC/data"
    SCALE = 256
    CLASSES = 7
    def __init__(self, focal_weights_kind=1, dir=None):
        if dir is not None:
            self.DIR = dir
        super().__init__(self.DIR, self.SCALE, self.CLASSES, binary_masks=True, focal_weights_kind=focal_weights_kind)

class MoNuSACDataset(BasicDataset):
    DIR = "../Datasets/MoNuSAC/data"
    SCALE = 512
    CLASSES = 5
    def __init__(self, focal_weights_kind=1, dir=None):
        if dir is not None:
            self.DIR = dir
        super().__init__(self.DIR, self.SCALE, self.CLASSES, binary_masks=True, focal_weights_kind=focal_weights_kind)

if __name__ == "__main__":
    dataset = CoNICDataset(focal_weights_kind=2)
    a = dataset[0]
    a = 1
