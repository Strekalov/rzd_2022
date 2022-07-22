from dataclasses import dataclass
from typing import Optional


@dataclass
class CLAHE:
    clip_limit: int
    grid_height: int
    grid_width: int


@dataclass
class Gamma:
    gamma_limit: int


@dataclass
class Augmentation:
    clahe: CLAHE
    gamma: Gamma


@dataclass
class Dataset:
    root_dir: str
    test_dir: str
    val_size: float
    with_augs: bool
    seed: int
    input_size: int
    batch_size: int
    num_workers: 16
    augmentation: Augmentation
    part: str = "full"


@dataclass
class Model:
    pretrain_name: str


@dataclass
class LrScheduler:
    name: str
    step_size: int
    gamma: float
    last_epoch: float


@dataclass
class Train:
    optimizer: str
    grad_accum_steps: int
    learning_rate: Optional[float]
    min_learning_rate: Optional[float]
    momentum: float
    weight_decay: float
    lr_schedule: LrScheduler
    n_epoch: int


@dataclass
class Config:
    exp_name: str
    seed: int
    out_dir: str
    dataset: Dataset
    model: Model
    train: Train
