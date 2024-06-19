import pandas as pd
import torch
from img2vec_pytorch import Img2Vec

from image_utils import *

file_list = os.listdir("raw_images")
create_directory("./embeddings")

for file in file_list:
    paths = get_images_from_dir(f"./processed_images/{file}")

    embeddings = []
    for iter in range(3):
        if iter == 2:
            images = [
                load_image(path)
                for path in paths[int(len(paths) / 3) * iter : len(paths) + 1]
            ]
        else:
            images = [
                load_image(path)
                for path in paths[
                    int(len(paths) / 3) * iter : int(len(paths) / 3) * (iter + 1)
                ]
            ]

        img2vec = Img2Vec(cuda=torch.cuda.is_available())

        embedding = img2vec.get_vec(images)

        print(embedding.shape)
        embeddings.append(embedding)

    df = pd.DataFrame(embeddings[0])
    for idx, embedding in enumerate(embeddings):
        if idx > 0:
            df_iter = pd.DataFrame(embeddings[idx])
            df = pd.concat([df, df_iter])

    print(df.shape)
    df["file_paths"] = paths

    df.to_csv(f"./embeddings/{file}_embeddings.csv", index=False)
