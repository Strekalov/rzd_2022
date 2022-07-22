from pathlib import Path

import numpy as np
import typer
from PIL import Image
from tqdm import tqdm


def is_dark(image: np.ndarray, threshold=127):
    return True if np.mean(image) < threshold else False


def get_sorted_files_from_dir(dir_path: Path):
    return sorted(list(dir_path.glob("*")))


def main(
    dataset_path: Path = typer.Option("", help="Путь до датасета"),
):
    dataset_path
    images = get_sorted_files_from_dir(dataset_path)
    for image_path in tqdm(images):
        image = Image.open(image_path).convert("RGB")
        np_image = np.array(image)

        if is_dark(image=np_image, threshold=75):
            with open("dark_images_in_dataset.txt", "a") as f:
                f.write(f"{image_path.name}\n")
        else:
            with open("light_images_in_dataset.txt", "a") as f:
                f.write(f"{image_path.name}\n")


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
