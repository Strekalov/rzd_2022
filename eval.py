from pathlib import Path

import albumentations as A
import numpy as np
import torch
import transformers
import typer
from accelerate import Accelerator
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import utils
from data import dataloader
from models import segformer
from schemas import Config
from train import validation
from utils import AverageMeter

miou_stat = AverageMeter("MIOU")


root_dir = Path("/home/artem/datasets/rzd_public/")
id2label = {0: "background", 1: "secondary_rails", 2: "main_rails", 3: "train_car"}
label2id = {"background": 0, "secondary_rails": 1, "main_rails": 2, "train_car": 3}
id2color = {0: [0, 0, 0], 1: [6, 6, 6], 2: [7, 7, 7], 3: [10, 10, 10]}
id2color_rgb = {0: [255, 255, 255], 1: [0, 0, 255], 2: [255, 0, 0], 3: [0, 255, 0]}
device = "cuda"


def run_inference(config: Config, checkpoint_path: str, scale: float):
    print("Load Feature Extractor...")
    config.dataset.batch_size = 1

    # feature_extractor = segformer.get_feature_extractor(config)
    print("Done.")

    print("Preparing val dataloaders...")
    test_dataloader = dataloader.get_scale_dataloader(config, scale=scale)
    print("Done.")
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    print("Load model...")
    model = segformer.get_model(config)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cuda")["state_dict"]
    model.load_state_dict(checkpoint)
    print("Done.")

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    print("Done.")

    val_miou = validation(
        model=model, dataloader=test_dataloader, device=device, miou_stat=miou_stat
    )
    print(f"Val MIOU on scale {scale}: {val_miou}")


def main(
    cfg: Path = typer.Option("configs/baseline.yml", help="Путь до конфига"),
    checkpoint_path: Path = typer.Option("", help="Путь до чекпоинта модели"),
    scale: float = typer.Option("", help="Scale image"),
):
    config: Config = OmegaConf.load(cfg)
    utils.set_global_seed(config)
    utils.prepare_predictions_dir(config)
    run_inference(config, checkpoint_path, scale)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
