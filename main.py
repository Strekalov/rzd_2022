from pathlib import Path

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
from utils import AverageMeter

miou_stat = AverageMeter("MIOU")


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
            miou_stat=miou_stat,
        )
        # val_miou = validation(model=model, dataloader=val_dataloader, device=device)
        scheduler.step()
        print(f"Train MIOU on {epoch} epoch: {miou_stat.avg}")
        train_miou = miou_stat.avg
        miou_stat.reset()
        val_miou = validation(
            model=model, dataloader=val_dataloader, device=device, miou_stat=miou_stat
        )
        print(f"Val MIOU on {epoch} epoch: {miou_stat.avg}")
        utils.save_checkpoint(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_miou=train_miou,
            val_miou=val_miou,
        )
        miou_stat.reset()


def main(
    cfg: Path = typer.Option("configs/baseline.yml", help="Путь до конфига"),
):
    config: Config = OmegaConf.load(cfg)
    utils.set_global_seed(config)
    utils.prepare_exp_dir(config)
    run_train(config)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
