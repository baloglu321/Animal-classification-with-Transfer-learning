import logging
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from smartcrop import detect

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_is_dir(path):
    if not os.path.isdir(path):
        raise ValueError(f"Provided path : {path} is not a directory")
    return True


def filter_images(list_of_files):
    valid_extensions = {".jpg", ".jpeg", ".png", "webp"}
    out = [
        file
        for file in list_of_files
        if any(file.endswith(ext) for ext in valid_extensions)
    ]
    return out


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def read_images_from_dir(dir_path):
    check_is_dir(dir_path)
    files = os.listdir(dir_path)
    image_files = filter_images(files)
    image_paths = [os.path.join(dir_path, file) for file in image_files]
    images = [load_image(image_path) for image_path in tqdm(image_paths)]
    logging.info(f"Loaded {len(images)} images from {dir_path}")
    return images


def get_images_from_dir(dir_path):
    check_is_dir(dir_path)
    files = os.listdir(dir_path)
    image_files = filter_images(files)
    image_paths = [os.path.join(dir_path, file) for file in image_files]
    return image_paths


def max_resolution_rescale(image, max_width, max_height):
    width, height = image.size
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize(
            (new_width, new_height), Image.LANCZOS
        )  # Image.LANCZOS bu metod küçültmede oluşan alizing problerini gidermek için
    return image


def min_resolution_filter(image, min_width, min_height):
    width, height = image.size
    return width >= min_width and height >= min_height


def plot_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def center_crop(image, new_width, new_height):
    width, height = image.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    cropped_image = image.crop((left, top, right, bottom))
    logging.info(f"Center cropped image to {new_width}X{new_height}")
    return cropped_image


def save_image(image, save_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise ValueError("Input image must be a numpy array or PIL image")
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(save_path)


def create_directory(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created : {dir_path}")
    else:
        print(f"Directory already exist : {dir_path}")


def save_images_to_dir(images, dir_path):
    create_directory(dir_path)
    check_is_dir(dir_path)

    for idx, image in tqdm(enumerate(images, 1)):
        save_path = os.path.join(dir_path, f"image_{idx}.png")
        save_image(image, save_path)
    return True
