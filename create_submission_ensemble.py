from pathlib import Path

import numpy as np
import torch
import transformers
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from schemas import Config


def get_sorted_files_from_dir(dir_path: Path):
    return sorted(list(dir_path.glob("*")))


root_dir = Path("").absolute()
public_dir = Path("/home/artem/datasets/rzd_public")
id2label = {0: "background", 1: "secondary_rails", 2: "main_rails", 3: "train_car"}
label2id = {"background": 0, "secondary_rails": 1, "main_rails": 2, "train_car": 3}
id2color = {0: [0, 0, 0], 1: [6, 6, 6], 2: [7, 7, 7], 3: [10, 10, 10]}
id2color_rgb = {0: [255, 255, 255], 1: [0, 0, 255], 2: [255, 0, 0], 3: [0, 255, 0]}
device = "cuda"


ensemble_models = [
    "predictions/segformer_864_b4_8_cosine_dark_and_light/predictions_0.651",
    "predictions/segformer_1024_b4_g16_adamW_cosine/predictions_0.741",
    "predictions/segformer_864_b4_8_cosine_dark_and_light/predictions_864",
    "predictions/segformer_1024_b4_g16_adamW_cosine/predictions_1024",
]

predictions = []

for prediction_dir in ensemble_models:
    prediction_data = dict()
    logits_dir = root_dir.joinpath(prediction_dir)
    predictions.append(get_sorted_files_from_dir(Path(prediction_dir)))


for i in tqdm(range(len(predictions[0]))):
    logits_list = []
    upsampled_logits = []
    for j in range(len(ensemble_models)):
        image_name = predictions[j][i].name
        image_path = public_dir.joinpath(image_name[:-4])
        logits_list.append(np.load(root_dir.joinpath(predictions[j][i])))

    image = Image.open(str(image_path))
    upsampled_logits_list = []
    for logits in logits_list:
        logits = torch.from_numpy(logits)

        one_upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        upsampled_logits_list.append(one_upsampled_logits)

    ensemble_upsampled_logits = sum(upsampled_logits_list) / len(upsampled_logits_list)
    seg = ensemble_upsampled_logits.argmax(dim=1)[0]

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    color_seg_rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in id2color.items():
        color_seg[seg == label, :] = color

    mask = color_seg.astype(np.uint8)
    mask = Image.fromarray(mask, mode="RGB")
    mask.save(f"final_ensemble/{image_path.name}")
