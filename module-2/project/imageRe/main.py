import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))
embedding_function = OpenCLIPEmbeddingFunction()


def get_single_image_embedding(image):
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)


def plot_results(query_path, ls_path_score, reverse=False):
    fig = plt.figure(figsize=(15, 9))
    fig.add_subplot(2, 3, 1)
    plt.imshow(read_image_from_path(query_path, size=(448, 448)))
    plt.title(f"Query Image: {query_path.split('/')[2]}", fontsize=16)
    plt.axis("off")

    for i, path in enumerate(sorted(ls_path_score, key=lambda x: x[1], reverse=reverse)[:5], 2):
        fig.add_subplot(2, 3, i)
        plt.imshow(read_image_from_path(path[0], size=(448, 448)))
        plt.title(
            f"Top {i-1}: {path[0].split('/')[2]} (Score: {path[1]:.4f})", fontsize=16)

        plt.axis("off")

    plt.savefig("output.png")
    plt.close(fig)


def read_image_from_path(path: str, size):
    image = Image.open(path).convert('RGB').resize(size)
    return np.array(image)


def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)


def get_l1_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []

        for idx_img in range(images_np.shape[0]):
            embedding = get_single_image_embedding(
                images_np[idx_img].astype(np.uint8))
            embedding_list.append(embedding)

        rates = absolute_difference(query_embedding, np.stack(embedding_list))
        ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


if "__main__" == __name__:
    root_img_path = f"{ROOT}/train/"
    query_path = f"{ROOT}/test/yawl/n04612504_4963.JPEG"
    size = (448, 448)
    query, ls_path_score = get_l1_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=False)

    # print(ls_path_score[:5])
