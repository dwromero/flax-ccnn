import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
from .datamodule import DataModule

import numpy as np
from functools import partial

# config
from hydra import utils


class MNISTDataModule(DataModule):
    def __init__(
            self,
            data_dir,
            batch_size,
            test_batch_size,
            data_type,
            num_workers,
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

        # Determine sizes of dataset
        self.input_channels = 1
        self.output_channels = 10

        # Define data mean and std
        DATA_MEAN = (0.1307,)
        DATA_STD = (0.3081,)

        # Create transforms
        train_transform = [
            partial(self.image_to_numpy, data_mean=DATA_MEAN, data_std=DATA_STD),
            ]

        val_test_transform = train_transform
        # Augmentation before normalization, taken from:
        # https://github.com/dipuk0506/SpinalNet/blob/master/CIFAR-10/ResNet_default_and_SpinalFC_CIFAR10.py#L39
        if self.augment:
            raise NotImplementedError
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
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self):
        # set up datamodules
        mnist = datasets.MNIST(
            self.data_dir,
            train=True,
            transform=self.train_transform,
        )
        self.train_dataset, self.val_dataset = random_split(
            mnist,
            [55000, 5000],
            generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
        )
        self.test_dataset = datasets.MNIST(
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
    def default_collate_fn(batch) -> tuple[np.array, np.array]:
        x, y = DataModule.default_collate_fn(batch)
        x = np.reshape(x, (*x.shape, 1))
        batch = x, y
        return batch

    @staticmethod
    def sequential_collate_fn(batch):
        x, y = MNISTDataModule.default_collate_fn(batch)
        # If sequential, flatten the input [B, Y, X, C] -> [B, -1, C]
        x_shape = x.shape
        x = np.reshape(x, (x_shape[0], -1, x_shape[-1]))
        batch = x, y
        return batch






