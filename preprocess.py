import os

from image_utils import *

max_size = 1024
min_size = 224

file_list = os.listdir("raw_images")

for file in file_list:
    loaded_images = read_images_from_dir(f"./raw_images/{file}")
    resized_images = list(
        map(
            lambda x: max_resolution_rescale(x, max_size, max_size), tqdm(loaded_images)
        )
    )
    filtered_images = list(
        filter(
            lambda x: min_resolution_filter(x, min_size, min_size), tqdm(resized_images)
        )
    )
    cropped_images = list(map(lambda x: detect(x, square=True), tqdm(filtered_images)))

    try:
        save_images_to_dir(cropped_images, f"processed_images/{file}")
    except:
        continue
