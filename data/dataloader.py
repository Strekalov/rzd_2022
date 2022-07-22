import numpy as np
import torch

from data import dataset_config
from schemas import Config

from . import dataset, transforms


def get_train_val_idx(config: Config, full_lenght: int):
    indices = list(range(full_lenght))
    split = int(np.floor(config.dataset.val_size * full_lenght))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    return train_idx, val_idx


def get_test_dataloader(config: Config, feature_extractor):
    test_dataset = dataset.RzdPublicDataset(
        root_dir=config.dataset.test_dir,
        id2color=dataset_config.ID_TO_COLOR,
        feature_extractor=feature_extractor,
        transform=transforms.val_transform,
        config=config,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return test_loader


def get_scale_dataloader(config: Config, scale: float):
    val_dataset = dataset.RzdDinamicScaleDataset(
        root_dir=config.dataset.root_dir,
        id2color=dataset_config.ID_TO_COLOR,
        scale=scale,
        transform=transforms.val_transform,
        config=config,
    )
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_dataloaders(config: Config, feature_extractor):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """

    train_dataset = dataset.RzdDataset(
        root_dir=config.dataset.root_dir,
        id2color=dataset_config.ID_TO_COLOR,
        feature_extractor=feature_extractor,
        transform=transforms.get_train_transform(config),
        config=config,
    )
    val_dataset = dataset.RzdDataset(
        root_dir=config.dataset.root_dir,
        id2color=dataset_config.ID_TO_COLOR,
        feature_extractor=feature_extractor,
        transform=transforms.val_transform,
        config=config,
    )
    full_lenght = len(train_dataset)
    train_idx, valid_idx = get_train_val_idx(config, full_lenght)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        # shuffle=True,
        # shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        # shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader
