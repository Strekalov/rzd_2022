import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from models.segformer import get_feature_extractor
from schemas import Config

# def get_gamma_correction(config: Config):
#     gamma_limit = config.dataset.augmentation.gamma.gamma_limit
#     return A.RandomGamma(
#         always_apply=False, p=1.0, gamma_limit=(gamma_limit, gamma_limit), eps=1e-07
#     )


# def get_clahe(config: Config):
#     clip_limit = config.dataset.augmentation.clahe.clip_limit
#     height = config.dataset.augmentation.clahe.grid_height
#     width = config.dataset.augmentation.clahe.grid_width
#     return A.CLAHE(
#         always_apply=False,
#         p=1.0,
#         clip_limit=(clip_limit, clip_limit),
#         tile_grid_size=(height, width),
#     )


def is_dark(image: np.ndarray, threshold=127):
    return True if np.mean(image) < threshold else False


class RzdDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(
        self,
        root_dir: str,
        feature_extractor,
        config: Config,
        id2color=None,
        transform=None,
    ):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
        """
        self.root_dir = Path(root_dir)
        self.feature_extractor = feature_extractor
        self.img_dir = self.root_dir.joinpath("images")
        self.ann_dir = self.root_dir.joinpath("annotations")
        self.id2color = id2color
        self.transform = transform
        self.config = config

        if self.config.dataset.part == "light":
            with open("light_images_in_dataset.txt", mode="r") as f:
                image_file_names = f.read().splitlines()
        elif self.config.dataset.part == "dark":
            with open("dark_images_in_dataset.txt", mode="r") as f:
                image_file_names = f.read().splitlines()
        else:
            image_file_names = [
                image_path.name for image_path in self.img_dir.glob("*")
            ]

        # for image_path in self.img_dir.glob("*"):
        #     image_file_names.append(image_path)

        self.images = sorted(image_file_names)

        # # read annotations
        # annotation_file_names = []
        # for ann_path in self.ann_dir.glob("*"):
        #     annotation_file_names.append(ann_path)

        # self.annotations = sorted(annotation_file_names)

        # assert len(self.images) == len(
        #     self.annotations
        # ), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        cv2.setNumThreads(16)

        image = Image.open(self.img_dir.joinpath(self.images[idx]))
        annotation = Image.open(self.ann_dir.joinpath(self.images[idx]))

        # make 2D segmentation map (based on 3D one)
        # thanks a lot, Stackoverflow: https://stackoverflow.com/questions/61897492/finding-the-number-of-pixels-in-a-numpy-array-equal-to-a-given-color
        annotation = np.array(annotation)
        image = np.array(image)
        # if is_dark(image=image, threshold=75):
        #     clahe = get_clahe(config=self.config)
        #     # gamma = get_gamma_correction(config=self.config)
        #     image = clahe(image=image)['image']
        #     # image = gamma(image=image)["image"]
        transformed = self.transform(image=image, mask=annotation)
        image = transformed["image"]
        annotation = transformed["mask"]

        annotation_2d = np.zeros(
            (annotation.shape[0], annotation.shape[1]), dtype=np.uint8
        )  # height, width

        for id, color in self.id2color.items():
            annotation_2d[(annotation == color).all(axis=-1)] = id

        # randomly crop + pad both image and segmentation map to same size
        # feature extractor will also reduce labels!
        encoded_inputs = self.feature_extractor(
            Image.fromarray(image), Image.fromarray(annotation_2d), return_tensors="pt"
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class RzdDinamicScaleDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(
        self,
        root_dir: str,
        config: Config,
        scale: float,
        id2color=None,
        transform=None,
    ):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir.joinpath("images")
        self.ann_dir = self.root_dir.joinpath("annotations")
        self.id2color = id2color
        self.transform = transform
        self.config = config
        self.scale = scale

        if self.config.dataset.part == "light":
            with open("light_images_in_dataset.txt", mode="r") as f:
                image_file_names = f.read().splitlines()
        elif self.config.dataset.part == "dark":
            with open("dark_images_in_dataset.txt", mode="r") as f:
                image_file_names = f.read().splitlines()
        else:
            image_file_names = [
                image_path.name for image_path in self.img_dir.glob("*")
            ]

        # for image_path in self.img_dir.glob("*"):
        #     image_file_names.append(image_path)

        self.images = sorted(image_file_names)

        # # read annotations
        # annotation_file_names = []
        # for ann_path in self.ann_dir.glob("*"):
        #     annotation_file_names.append(ann_path)

        # self.annotations = sorted(annotation_file_names)

        # assert len(self.images) == len(
        #     self.annotations
        # ), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        cv2.setNumThreads(16)
        config = self.config

        image = Image.open(self.img_dir.joinpath(self.images[idx]))
        config.dataset.input_size = min(
            int(image.size[0] * self.scale), int(image.size[1] * self.scale)
        )
        feature_extractor = get_feature_extractor(config=config)
        annotation = Image.open(self.ann_dir.joinpath(self.images[idx]))

        # make 2D segmentation map (based on 3D one)
        # thanks a lot, Stackoverflow: https://stackoverflow.com/questions/61897492/finding-the-number-of-pixels-in-a-numpy-array-equal-to-a-given-color
        annotation = np.array(annotation)
        image = np.array(image)
        # if is_dark(image=image, threshold=75):
        #     clahe = get_clahe(config=self.config)
        #     # gamma = get_gamma_correction(config=self.config)
        #     image = clahe(image=image)['image']
        #     # image = gamma(image=image)["image"]
        transformed = self.transform(image=image, mask=annotation)
        image = transformed["image"]
        annotation = transformed["mask"]

        annotation_2d = np.zeros(
            (annotation.shape[0], annotation.shape[1]), dtype=np.uint8
        )  # height, width

        for id, color in self.id2color.items():
            annotation_2d[(annotation == color).all(axis=-1)] = id

        # randomly crop + pad both image and segmentation map to same size
        # feature extractor will also reduce labels!
        encoded_inputs = feature_extractor(
            Image.fromarray(image), Image.fromarray(annotation_2d), return_tensors="pt"
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class RzdPublicDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(
        self,
        root_dir: str,
        feature_extractor,
        config: Config,
        id2color=None,
        transform=None,
    ):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
        """
        self.root_dir = Path(root_dir)
        self.feature_extractor = feature_extractor
        self.id2color = id2color
        self.transform = transform
        self.config = config
        # read images
        image_file_names = []
        for image_path in self.root_dir.glob("*"):
            image_file_names.append(image_path)

        self.images = sorted(image_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        cv2.setNumThreads(16)
        image = Image.open(os.path.join(self.root_dir, self.images[idx]))

        # make 2D segmentation map (based on 3D one)
        # thanks a lot, Stackoverflow: https://stackoverflow.com/questions/61897492/finding-the-number-of-pixels-in-a-numpy-array-equal-to-a-given-color

        image = np.array(image)
        transformed = self.transform(image=image)
        image = transformed["image"]

        # randomly crop + pad both image and segmentation map to same size
        # feature extractor will also reduce labels!
        encoded_inputs = self.feature_extractor(
            Image.fromarray(image), return_tensors="pt"
        )
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return {
            "image": encoded_inputs,
            "image_size": image.shape[:2],
            "filename": self.images[idx].name,
        }
