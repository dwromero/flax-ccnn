import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
from .datamodule import DataModule

import numpy as np
from functools import partial

# config
from hydra import utils


class CIFAR10DataModule(DataModule):
    def __init__(
            self,
            data_dir,
            batch_size,
            test_batch_size,
            data_type,
            num_workers,
            noise_padded,
            grayscale,
            augment,
            **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
        )

        # Dataset specifc attributes
        self.noise_padded = noise_padded
        self.grayscale = grayscale  # Used for LRA
        self.augment = augment

        # Determine data_type
        if data_type == "default":
            self.data_type = "image"
            self.data_dim = 2
        elif data_type == "sequence":
            self.data_type = data_type
            self.data_dim = 1
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        # Noise for noise-padded sCIFAR10
        if self.data_type == "sequence" and self.noise_padded:
            self.rands = torch.randn(1, 1000 - 32, 96)

        # Determine sizes of dataset
        if self.data_type == "sequence" and self.noise_padded:
            self.input_channels = 96
        elif self.grayscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
        self.output_channels = 10

        # Define data mean and std
        if self.grayscale:
            DATA_MEAN = (122.6 / 255.0,)
            DATA_STD = (61.0 / 255.0,)
        else:
            DATA_MEAN = (0.4914, 0.4822, 0.4465)
            DATA_STD = (0.247, 0.243, 0.261)

        # Create transforms
        train_transform = []
        if self.grayscale:
            train_transform = train_transform + [
                transforms.Grayscale(),
            ]
        train_transform = train_transform + [
            partial(self.image_to_numpy, data_mean=DATA_MEAN, data_std=DATA_STD),
        ]

        val_test_transform = train_transform
        # Augmentation before normalization, taken from:
        # https://github.com/dipuk0506/SpinalNet/blob/master/CIFAR-10/ResNet_default_and_SpinalFC_CIFAR10.py#L39
        if self.augment:
            train_transform = [
                                  transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
                                  transforms.RandomHorizontalFlip(),
                              ] + train_transform

        self.train_transform = transforms.Compose(train_transform)
        self.val_test_transform = transforms.Compose(val_test_transform)

        # define collate_function
        self.construct_collate_fn()

    def prepare_data(self):
        # download data, train then test
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self):
        # set up datamodules
        cifar10 = datasets.CIFAR10(
            self.data_dir,
            train=True,
            transform=self.train_transform,
        )
        self.train_dataset, self.val_dataset = random_split(
            cifar10,
            [45000, 5000],
            generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
        )
        self.test_dataset = datasets.CIFAR10(
            self.data_dir,
            train=False,
            transform=self.val_test_transform,
        )
        # Construct dataloaders based on the datasets
        self.construct_dataloaders()

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return test_dataloader

    def construct_collate_fn(self):
        if self.data_type == 'sequence':
            self.collate_fn = self.sequential_collate_fn
        else:
            self.collate_fn = self.default_collate_fn

    @staticmethod
    def sequential_collate_fn(batch):
        x, y = DataModule.default_collate_fn(batch)
        # If sequential, flatten the input [B, Y, X, C] -> [B, -1, C]
        x_shape = x.shape
        x = np.reshape(x, (x_shape[0], -1, x_shape[-1]))
        batch = x, y
        return batch






