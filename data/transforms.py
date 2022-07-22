import albumentations as A

from schemas import Config

train_transform = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        A.Blur(always_apply=False, p=0.2, blur_limit=(1, 3)),
        A.CoarseDropout(
            always_apply=False,
            p=0.4,
            max_holes=50,
            max_height=50,
            max_width=50,
            min_holes=20,
            min_height=30,
            min_width=30,
        ),
        A.MultiplicativeNoise(
            always_apply=False,
            p=0.4,
            multiplier=(0.9, 1.1),
            per_channel=True,
            elementwise=True,
        ),
        A.HueSaturationValue(
            always_apply=False,
            p=0.2,
            hue_shift_limit=(-20, 20),
            sat_shift_limit=(-30, 30),
            val_shift_limit=(-20, 20),
        ),
        A.ToGray(always_apply=False, p=0.2),
        # A.RandomBrightnessContrast(
        #     always_apply=False,
        #     p=0.2,
        #     brightness_limit=(-0.1, 0.1),
        #     contrast_limit=(-0.1, 0.1),
        #     brightness_by_max=True,
        # ),
    ]
)


val_transform = A.Compose([])


def get_train_transform(config: Config):
    if config.dataset.with_augs:
        return train_transform
    return val_transform
