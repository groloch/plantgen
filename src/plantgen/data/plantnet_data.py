import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from datasets import load_dataset

from ..config.data import PlantNetDataConfig, PlantNetTTIDataConfig


class PlantNetDataset(Dataset):
    def __init__(self, dataset, image_size, augmentation_mode):
        super().__init__()
        self.dataset = dataset
        self.augmentation_mode = augmentation_mode

        if augmentation_mode == 'valid':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif augmentation_mode == 'classifier':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.AutoAugment(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif augmentation_mode == 'vae':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif augmentation_mode == 'vae_grayscale':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
            ])

        elif augmentation_mode == 'vlm':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToImage(),
                # transforms.ToDtype(torch.float32, scale=True),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self._len = len(self.dataset)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


class PlantNetTTIDataset(Dataset):
    def __init__(self, dataset, annotations, image_size, threshold):
        super().__init__()

        self.dataset = dataset
        self.annotations = annotations
        self.image_size = image_size

        self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.siglip_scores = np.load('data/plantnet_captions_siglip_scores.npy')
        self.indices = np.where(self.siglip_scores >= threshold)[0]

        self._len = len(self.indices)

        assert len(self.dataset) == len(self.annotations) == len(self.siglip_scores)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        item = self.dataset[self.indices[index]]
        image = item['image']
        caption = self.annotations.iloc[self.indices[index]]['caption'][1:-1]
        if self.transform:
            image = self.transform(image)
        return image, caption


class PlantNetPackedTTIDataset(Dataset):
    def __init__(
            self,
            dataset,
            latent_dataset,
            annotations,
            latent_dim: int,
            latent_size: int,
            threshold: float
        ):
        super().__init__()
        self.dataset = dataset
        self.latent_dataset = latent_dataset
        self.annotations = annotations

        self.latent_dim = latent_dim
        self.latent_size = latent_size

        self.siglip_scores = np.load('data/plantnet_captions_siglip_scores.npy')
        self.indices = np.where(self.siglip_scores >= threshold)[0]
        self._len = len(self.indices)

        assert len(self.dataset) == len(self.annotations) == len(self.siglip_scores)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        latent = self.latent_dataset[self.indices[index]]['latents']
        caption = self.annotations.iloc[self.indices[index]]['caption'][1:-1]

        latent = torch.as_tensor(latent)
        latent = latent.reshape(self.latent_dim, self.latent_size, self.latent_size)

        return latent, caption


def get_plantnet_dataloaders(
        data_config: PlantNetDataConfig,
        drop_last: bool = True,
        shuffle_train: bool = True):
    dataset = load_dataset('mikehemberger/plantnet300K')
    # TODO do something with test data

    train_dataset = PlantNetDataset(
        dataset['train'],
        image_size=data_config.image_size,
        augmentation_mode=data_config.augmentation_mode
    )
    val_dataset = PlantNetDataset(
        dataset['validation'],
        image_size=data_config.image_size,
        augmentation_mode='valid'
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=shuffle_train,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=drop_last
    )

    return train_dataloader, valid_dataloader

def get_plantnet_tti_dataloaders(data_config: PlantNetTTIDataConfig):
    dataset = load_dataset('mikehemberger/plantnet300K')
    annotations = pd.read_csv(data_config.annotations_path, sep='\r')
    # TODO generate captions for valid / test

    if data_config.precomputed_latents:
        latent_dataset = load_dataset(data_config.latents_path)
        train_dataset = PlantNetPackedTTIDataset(
            dataset['train'],
            latent_dataset['train'],
            annotations=annotations,
            latent_dim=data_config.latent_dim,
            latent_size=data_config.latent_size,
            threshold=data_config.similarity_threshold
        )
    else:
        train_dataset = PlantNetTTIDataset(
            dataset['train'],
            annotations=annotations,
            image_size=data_config.image_size,
            threshold=data_config.similarity_threshold
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, None
