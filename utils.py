import pathlib
import random
from collections import OrderedDict

import numpy as np
import torch
import transformers
from omegaconf import OmegaConf
from transformers.optimization import Adafactor, AdafactorSchedule

from schemas import Config


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self):
        return self.val, self.avg


def get_scheduler(config: Config, optimizer):
    if config.train.lr_schedule.name == "Adafactor":
        return AdafactorSchedule(optimizer)
    elif config.train.lr_schedule.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.n_epoch,
            eta_min=config.train.min_learning_rate,
            last_epoch=-1,
        )
    elif config.train.lr_schedule.name == "cosine_warmup":
        num_training_steps = 4100 / 2 * config.train.n_epoch
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2,
            num_training_steps=num_training_steps,
            num_cycles=4,
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.n_epoch,
            eta_min=0.0000005,
            last_epoch=-1,
        )
    elif config.train.lr_schedule.name == "CyclicLR":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.train.min_learning_rate,
            max_lr=config.train.learning_rate,
            cycle_momentum=False,
        )
    else:
        # torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=- 1, verbose=False)
        return getattr(torch.optim.lr_scheduler, config.train.lr_schedule.name)(
            optimizer,
            step_size=config.train.lr_schedule.step_size,
            gamma=config.train.lr_schedule.gamma,
            last_epoch=config.train.lr_schedule.last_epoch,
        )


def get_optimizer(config: Config, net):

    if config.train.optimizer == "Adafactor":
        optimizer = Adafactor(
            net.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=config.train.learning_rate,
        )
    else:
        optimizer = getattr(torch.optim, config.train.optimizer)(
            net.parameters(),
            weight_decay=config.train.weight_decay,
            lr=config.train.learning_rate,
        )

    return optimizer


def get_training_parameters(config: Config, net):
    optimizer = get_optimizer(config, net)
    scheduler = get_scheduler(config, optimizer)
    return optimizer, scheduler


def set_global_seed(config: Config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def prepare_exp_dir(config: Config):
    exp_name = config.exp_name
    exp_dir = pathlib.Path().absolute().joinpath("experiments").joinpath(exp_name)
    checkpoints_dir = exp_dir.joinpath("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    yml_config = exp_dir.joinpath(f"{exp_name}.yml")
    with yml_config.open("w") as f:
        OmegaConf.save(config, f)


def prepare_predictions_dir(config: Config):
    exp_name = config.exp_name
    exp_dir = pathlib.Path().absolute().joinpath("predictions").joinpath(exp_name)
    predictions_dir = exp_dir.joinpath("predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    yml_config = exp_dir.joinpath(f"{exp_name}.yml")
    with yml_config.open("w") as f:
        OmegaConf.save(config, f)


def save_checkpoint(
    config: Config, model, optimizer, scheduler, epoch, train_miou, val_miou
):
    """Saves checkpoint to disk"""
    exp_name = config.exp_name

    checkpoints_dir = (
        pathlib.Path()
        .absolute()
        .joinpath("experiments")
        .joinpath(exp_name)
        .joinpath("checkpoints")
    )
    filename = checkpoints_dir.joinpath(
        f"model_{epoch:03d}_miou_{train_miou:.4f}_{val_miou:.4f}.pth"
    )
    weights = model.state_dict()
    state = OrderedDict(
        [
            ("state_dict", weights),
            ("optimizer", optimizer.state_dict()),
            ("scheduler", scheduler.state_dict()),
            ("epoch", epoch),
        ]
    )

    torch.save(state, str(filename))


# SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (
        (outputs & labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + 1e-6) / (union + 1e-6)  # We smooth our devision to avoid 0/0

    thresholded = (
        torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds

    return thresholded.mean()
