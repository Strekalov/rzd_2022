import argparse
import logging
import sys
from pathlib import Path

import optuna
import torch
import typer
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm

import utils
from data import dataloader
from models import segformer
from schemas import Config
from train import train, validation


def run_train(config: Config):

    print("Load Feature Extractor...")
    feature_extractor = segformer.get_feature_extractor(config)
    print("Done.")

    print("Preparing train and val dataloaders...")
    train_dataloader, val_dataloader = dataloader.get_dataloaders(
        config, feature_extractor
    )
    print("Done.")
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    print("Load model...")
    model = segformer.get_model(config)
    model.to(device)
    print("Done.")

    print("Prepare training params...")
    optimizer, scheduler = utils.get_training_parameters(config, model)

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    print("Done.")
    n_epoch = config.train.n_epoch

    train_epoch = tqdm(range(n_epoch), dynamic_ncols=True, desc="Epochs", position=0)

    for epoch in train_epoch:
        train(
            config=config,
            model=model,
            dataloader=train_dataloader,
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )
        return validation(model=model, dataloader=val_dataloader, device=device)
        # scheduler.step()
        # utils.save_checkpoint(
        #     config=config,
        #     model=model,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     epoch=epoch,
        #     val_miou=val_miou,
        # )


def objective(trial) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    args = parse_arguments(sys.argv[1:])
    config: Config = OmegaConf.load(args.cfg)
    utils.set_global_seed(config)
    # utils.prepare_exp_dir(config)

    config.train.learning_rate = trial.suggest_float("lr", 1e-5, 5e-4)
    # config.train.optimizer = trial.suggest_categorical(
    #     "optimizer", ["AdamW"]
    # )
    # config.model.pretrain_name = trial.suggest_categorical(
    #     "pretrain",
    #     [
    #         "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    #         # "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    #         # "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    #     ],
    # )
    config.train.grad_accum_steps = trial.suggest_int("grad_accum_steps", 4, 32, step=4)
    config.train.weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.3)
    config.dataset.input_size = trial.suggest_int("input_size", 512, 1024, step=32)
    print(
        f"Start with pretrain: {config.model.pretrain_name}\n image_size: {config.dataset.input_size}\n optimizer: {config.train.optimizer}\n lr: {config.train.learning_rate:.6f}\n weight_decay: {config.train.weight_decay}\n grad_accum_steps: {config.train.grad_accum_steps}"
    )
    return run_train(config)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="Path to config file.")
    return parser.parse_args(argv)


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}\n Best params: {study.best_trial.params}")


def main():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "find_b4_best_hyps"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=100000, callbacks=[print_best_callback])


if __name__ == "__main__":
    main()
