from pathlib import Path
from typing import Optional

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
from models.segformer import get_feature_extractor
from schemas import Config


def get_clahe(config: Config):
    clip_limit = config.dataset.augmentation.clahe.clip_limit
    height = config.dataset.augmentation.clahe.grid_height
    width = config.dataset.augmentation.clahe.grid_width
    return A.CLAHE(
        always_apply=False,
        p=1.0,
        clip_limit=(clip_limit, clip_limit),
        tile_grid_size=(height, width),
    )


def is_dark(image: np.ndarray, threshold=127):
    return True if np.mean(image) < threshold else False


root_dir = Path("/home/artem/datasets/rzd_public/")
id2label = {0: "background", 1: "secondary_rails", 2: "main_rails", 3: "train_car"}
label2id = {"background": 0, "secondary_rails": 1, "main_rails": 2, "train_car": 3}
id2color = {0: [0, 0, 0], 1: [6, 6, 6], 2: [7, 7, 7], 3: [10, 10, 10]}
id2color_rgb = {0: [255, 255, 255], 1: [0, 0, 255], 2: [255, 0, 0], 3: [0, 255, 0]}
device = "cuda"


def save_logits(config: Config, logits, filename: str, prefix: float):
    logits = logits.numpy()
    exp_name = config.exp_name
    save_dir = (
        Path()
        .absolute()
        .joinpath("predictions")
        .joinpath(exp_name)
        .joinpath(f"predictions_{prefix}")
    )
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir.joinpath(f"{filename}.npy")
    np.save(str(save_path), logits)


def simple_inference(config: Config, model, scale):
    test_dir = Path(config.dataset.test_dir)
    # test_dir=root_dir
    for image_path in tqdm(list(test_dir.glob("*"))):

        with torch.no_grad():
            image = Image.open(str(image_path))

            if scale is not None:
                # print(min(int(image.size[0]*scale), int(image.size[1]*scale)))
                config.dataset.input_size = min(
                    int(image.size[0] * scale), int(image.size[1] * scale)
                )

            np_image = np.array(image)
            feature_extractor = get_feature_extractor(config=config)
            if config.dataset.part == "dark":
                condition = is_dark(image=np_image, threshold=75)
            elif config.dataset.part == "light":
                condition = not is_dark(image=np_image, threshold=75)
            else:
                condition = True
            if condition:
                # if is_dark(image=image, threshold=75):
                #     clahe = get_clahe(config=config)
                #     # gamma = get_gamma_correction(config=self.config)
                #     image = clahe(image=image)['image']
                #     # image = gamma(image=image)["image"]

                # prepare the image for the model
                encoding = feature_extractor(image, return_tensors="pt")
                pixel_values = encoding.pixel_values.to(device)
                # forward pass
                outputs = model(pixel_values=pixel_values)
                # logits are of shape (batch_size, num_labels, height/4, width/4)
                logits = outputs.logits.cpu()

                # First, rescale logits to original image size
                # upsampled_logits = torch.nn.functional.interpolate(logits,
                #                 size=image.size[::-1], # (height, width)
                #                 mode='bilinear',
                #                 align_corners=False)
                prefix = scale if scale is not None else config.dataset.input_size
                save_logits(
                    config=config,
                    logits=logits,
                    filename=image_path.name,
                    prefix=prefix,
                )

                # seg = upsampled_logits.argmax(dim=1)[0]

                # color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
                # # color_seg_rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
                # for label, color in id2color.items():
                #     color_seg[seg == label, :] = color
                #     # color_seg_rgb[seg == label, :] = id2color_rgb[label]
                # mask = color_seg.astype(np.uint8)
                # mask = Image.fromarray(mask, mode='RGB')
                # mask.save(f"ensemble_b4_b5/{image_path.name}")
                # img = np.array(image) * 0.5 + color_seg_rgb * 0.5
                # img = img.astype(np.uint8)
                # mask_and_image = Image.fromarray(img, mode='RGB')
                # mask_and_image.save(f"/home/artem/datasets/rzd_pseudo_rgb/{image_path.name}")
                # import cv2
                # # cv2.imshow("img", img)
                # # cv2.waitKey(0)
                # cv2.imwrite(f"new_scam/{i}.png", img)


def inference(model, dataloader, config: Config):
    test_iter = tqdm(dataloader, desc="Public", dynamic_ncols=True, position=2)
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(test_iter, start=1):

            outputs = model(**batch["image"])

            # outputs = model(pixel_values=pixel_values)
            # logits are of shape (batch_size, num_labels, height/4, width/4)
            batch_logits = outputs.logits.cpu()

            for image_size, image_filename, logits in zip(
                batch["image_size"], batch["filename"], batch_logits
            ):
                # print(image_size, image_filename, logits.shape)
                upsampled_logits = torch.nn.functional.interpolate(
                    logits.unsqueeze(0),
                    size=tuple(image_size.cpu()),  # (height, width)
                    mode="bilinear",
                    align_corners=False,
                )
                save_logits(
                    config=config, logits=upsampled_logits, filename=image_filename
                )
            # # First, rescale logits to original image size

            # print(upsampled_logits.shape)

            # loss, logits = outputs.loss, outputs.logits
            # labels = batch["labels"].to(device)

            # upsampled_logits = torch.nn.functional.interpolate(
            #     logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            # )
            # predicted = upsampled_logits.argmax(dim=1)


def run_inference(config: Config, checkpoint_path: str, scale: float):
    print("Load Feature Extractor...")
    # config.dataset.input_size = 1340
    # feature_extractor = segformer.get_feature_extractor(config)
    print("Done.")

    # print("Preparing train and val dataloaders...")
    # test_dataloader = dataloader.get_test_dataloader(config, feature_extractor)
    # print("Done.")
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    print("Load model...")
    model = segformer.get_model(config)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cuda")["state_dict"]
    model.load_state_dict(checkpoint)
    print("Done.")

    # model, test_dataloader = accelerator.prepare(model, test_dataloader)
    print("Done.")
    simple_inference(config=config, model=model, scale=scale)
    # inference(model=model, dataloader=test_dataloader, config=config)


# feature_extractor = transformers.SegformerFeatureExtractor.from_pretrained(
#     "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
#     size=864,
#     num_labels=4,
#     id2label=id2label,
#     label2id=label2id,
#     # reduce_labels=True
# )

# model = transformers.SegformerForSemanticSegmentation.from_pretrained(
#     "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
#     num_labels=4,

#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True,

# )

# checkpoint = torch.load("experiments/segformer_864_b5_g8_adamW_cosine/checkpoints/model_004_miou_0.9010.pth", map_location='cuda')['state_dict']

# model.load_state_dict(checkpoint)
# model.to(device)
# model.eval()


# for i, image_path in enumerate(tqdm(list(root_dir.glob("*"))), start=0):
#     # print(image_path)
#     # image_path = Path("/home/artem/datasets/rzd_public/img_0.6830217821874891.png")
#     with torch.no_grad():
#         image = Image.open(str(image_path))

#         # prepare the image for the model
#         encoding = feature_extractor(image, return_tensors="pt")

#         pixel_values = encoding.pixel_values.to(device)
#         # forward pass
#         outputs = model(pixel_values=pixel_values)
#         # logits are of shape (batch_size, num_labels, height/4, width/4)
#         logits = outputs.logits.cpu()

#         # First, rescale logits to original image size
#         upsampled_logits = torch.nn.functional.interpolate(logits,
#                         size=image.size[::-1], # (height, width)
#                         mode='bilinear',
#                         align_corners=False)


#         seg = upsampled_logits.argmax(dim=1)[0]

#         color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
#         # color_seg_rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
#         for label, color in id2color.items():
#             color_seg[seg == label, :] = color
#             # color_seg_rgb[seg == label, :] = id2color_rgb[label]
#         mask = color_seg.astype(np.uint8)
#         mask = Image.fromarray(mask, mode='RGB')
#         mask.save(f"ensemble_b4_b5/{image_path.name}")
#         # img = np.array(image) * 0.5 + color_seg_rgb * 0.5
#         # img = img.astype(np.uint8)
#         # mask_and_image = Image.fromarray(img, mode='RGB')
#         # mask_and_image.save(f"/home/artem/datasets/rzd_pseudo_rgb/{image_path.name}")
#         # import cv2
#         # # cv2.imshow("img", img)
#         # # cv2.waitKey(0)
#         # cv2.imwrite(f"new_scam/{i}.png", img)


def main(
    cfg: Path = typer.Option("configs/baseline.yml", help="Путь до конфига"),
    checkpoint_path: Path = typer.Option("", help="Путь до чекпоинта модели"),
    scale: Optional[float] = typer.Option(None, help="Scale image"),
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
