import torch
import glob
import numpy as np
import random
import os
import pdb

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_from_disk, load_dataset

from logger.logger import info


class DataExtractor(Dataset):
    def __init__(self, mode, lr_path, hr_path, transform=None, eval=False):
        self.mode = mode
        self.lr_path = os.path.join(lr_path, 'arrow')
        self.hr_path = os.path.join(hr_path, 'arrow')
        self.transform = transform
        self.eval = eval

        self.lr_img, self.hr_img = self.load_datasets()

    def save_to_disk(self, arrow_path):
        os.makedirs(arrow_path)
        db_path = os.path.dirname(arrow_path)
        db_img = load_dataset(db_path, data_files={self.mode: '*.png'}, split=self.mode).with_format('torch')
        db_img.save_to_disk(arrow_path)
        return db_img

    def load_datasets(self):
        if os.path.exists(self.lr_path):
            lr_img = load_from_disk(self.lr_path).with_format("torch")
            info(f'{len(lr_img)} LR images have been loaded')
        else:
            lr_img = self.save_to_disk(self.lr_path)
            info(f'{len(lr_img)} LR images have been loaded and saved to disk')

        if os.path.exists(self.hr_path):
            hr_img = load_from_disk(self.hr_path).with_format("torch")
            info(f'{len(hr_img)} HR images have been loaded')
        else:
            hr_img = self.save_to_disk(self.hr_path)
            info(f'{len(hr_img)} HR images have been loaded and saved to disk')

        return lr_img, hr_img

    def __len__(self):
        return len(self.lr_img)

    def __getitem__(self, i):
        img_item = {}
        lr_img = self.lr_img[i]['image']
        hr_img = self.hr_img[i]['image']

        if self.eval:
            img_item['lr'] = lr_img
            img_item['hr'] = hr_img
            return img_item

        img_item['lr'] = (lr_img / 127.5) - 1.0
        img_item['hr'] = (hr_img / 127.5) - 1.0

        img_item['lr'] = np.array(img_item['lr']).transpose(1, 2, 0)
        img_item['hr'] = np.array(img_item['hr']).transpose(1, 2, 0)

        if self.transform is not None:
            img_item = self.transform(img_item)

        img_item['lr'] = img_item['lr'].transpose(2, 0, 1)
        img_item['hr'] = img_item['hr'].transpose(2, 0, 1)
        return img_item


class crop(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, sample):
        lr_img, hr_img = sample['lr'], sample['hr']
        ih, iw = lr_img.shape[:2]

        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)

        tx = ix * self.scale
        ty = iy * self.scale

        lr_patch = lr_img[iy: iy + self.patch_size, ix: ix + self.patch_size]
        hr_patch = hr_img[ty: ty + (self.scale * self.patch_size), tx: tx + (self.scale * self.patch_size)]

        return {'lr': lr_patch, 'hr': hr_patch}


class augmentation(object):
    def __call__(self, sample):
        lr_img, hr_img = sample['lr'], sample['hr']

        hor_flip = random.randrange(0, 2)
        ver_flip = random.randrange(0, 2)
        rot = random.randrange(0, 2)

        if hor_flip:
            temp_lr = np.fliplr(lr_img)
            lr_img = temp_lr.copy()
            temp_hr = np.fliplr(hr_img)
            hr_img = temp_hr.copy()

            del temp_lr, temp_hr

        if ver_flip:
            temp_lr = np.flipud(lr_img)
            lr_img = temp_lr.copy()
            temp_hr = np.flipud(hr_img)
            hr_img = temp_hr.copy()

            del temp_lr, temp_hr

        if rot:
            lr_img = lr_img.transpose(1, 0, 2)
            hr_img = hr_img.transpose(1, 0, 2)

        return {'lr': lr_img, 'hr': hr_img}

