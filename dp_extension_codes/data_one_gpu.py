#!/usr/bin/python
#
# Copyright 2022 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as tf
import torchvision.transforms as transforms
import os
from typing import Callable
from torch.utils.data.distributed import DistributedSampler


def get_files(path, ext):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(ext)]


"""class TransformOCTBilinear(object):
    def __new__(cls, img_size=(128, 128), *args, **kwargs):
        return tf.Compose([
            tf.Resize(img_size)
        ])"""


class TransformOCTBilinear(object):
    def __init__(self, img_size=(128, 128), n_channels=None):
        self.img_size = img_size
        self.n_channels = n_channels
        self.resize_transform = transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, img):
        # Check and handle the number of channels
        if img.mode == 'RGBA':  # Assuming img is a PIL.Image
            # Convert RGBA to RGB by discarding the alpha channel
            img = img.convert('RGB')

        # Apply resizing
        img = self.resize_transform(img)

        # Optionally adjust the number of channels if specifically required
        if self.n_channels == 1:
            # Convert to grayscale (this would now operate on RGB images)
            img = transforms.Grayscale(num_output_channels=1)(img)
        elif self.n_channels == 3 and img.mode != 'RGB':
            # Ensure it is in RGB format
            img = img.convert('RGB')

        return img
def get_data(data_path, img_size, batch_size, val_batch_size=10):
    train_dataset_path = os.path.join(data_path, "train")
    val_dataset_path = os.path.join(data_path, "val")
    test_dataset_path = os.path.join(data_path, "test")

    #size_transform = TransformOCTBilinear(img_size=(img_size, img_size))
    size_transform = TransformOCTBilinear(img_size=(img_size, img_size), n_channels=1)

    img_transform = None

    train_dataset = DatasetOct(train_dataset_path, size_transform=size_transform, normalized=True,
                               image_transform=img_transform,is_train=True,is_eval=False)
    val_dataset = DatasetOct(val_dataset_path, size_transform=size_transform, normalized=True,is_eval=True)
    test_dataset = DatasetOct(test_dataset_path, size_transform=size_transform, normalized=True,is_eval=True)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)


    return trainloader, valloader, testloader, train_dataset, val_dataset, test_dataset


class TransformStandardization(object):
    """
    Standardizaton / z-score: (x-mean)/std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + f": mean {self.mean}, std {self.std}"


class TransformOCTMaskAdjustment(object):
    """
    Adjust OCT 2015 Mask
    from: classes [0,1,2,3,4,5,6,7,8,9], where 9 is fluid, 0 is empty space above and 8 empty space below
    to: class 0: not class, classes 1-7: are retinal layers, class 8: fluid
    """

    def __call__(self, mask):
        mask[mask == 8] = 0
        mask[mask == 9] = 8
        return mask


class DatasetOct(Dataset):
    """
    Map style dataset object for oct 2015 data
    - expects .npy files
    - assumes sliced images, masks - produced by our project: dataloading/preprocessing.py
        (valid dims of images,masks and encoding: pixel label e [0..9], for every pixel)

    Parameters:
        dataset_path: path to the dataset path/{images,masks}
        size_transform: deterministic transformation for resizing applied to image and mask separately
        joint_transform: random transformations applied to image and mask jointly after size_transform
        image_transform: transformation applied only to the image and after joint_transform
    _getitem__(): returns image and corresponding mask
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, size_transform: Callable = None,
                 image_transform: Callable = None, normalized=True, is_train=False,is_eval=True) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = get_files(self.input_path, ".npy")
        self.is_train = is_train
        self.is_eval = is_eval
        # size transform
        self.size_transform = size_transform

        self.joint_transform = joint_transform if self.is_train else None  # Apply joint transforms only if is_train is True

        self.mask_adjust = TransformOCTMaskAdjustment()

        self.image_transform = image_transform

        self.normalized = normalized
        # gray scale oct 2015: calculated with full tensor in memory {'mean': tensor([46.3758]), 'std': tensor([53.9434])}
        # calculated with batched method {'mean': tensor([46.3756]), 'std': tensor([53.9204])}
        self.normalize = TransformStandardization((46.3758),
                                                  (53.9434))  # torchvision.transforms.Normalize((46.3758), (53.9434))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        """if self.is_eval:
            print(f"Processing file: {image_filename}")"""
        img = np.load(os.path.join(self.input_path, image_filename))
        mask = np.load(os.path.join(self.output_path, image_filename))

        # img_size 128 works - general transforms require (N,C,H,W) dims
        img = img.squeeze()
        mask = mask.squeeze()

        img = torch.Tensor(img).reshape(1, 1, *img.shape)
        mask = torch.Tensor(mask).reshape(1, 1, *mask.shape).int()

        # adjust mask classes
        mask = self.mask_adjust(mask)

        # note for some reason some masks differ in size from the actual image (dims)
        if self.size_transform:
            img = self.size_transform(img)
            mask = self.size_transform(mask)

        # normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        if self.joint_transform:
            img, mask = self.joint_transform([img, mask])

        # img = img.reshape(1, img.shape[2], img.shape[3])
        if self.image_transform:
            img = self.image_transform(img)

        # img = img.reshape(1, *img.shape)

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (H,W)  where value at (h,w) maps to class of corresponding pixel
        mask = mask.squeeze(dim=1).long()

        return img, mask

