import cv2 as cv
import numpy as np

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_images(image_paths: list[str], base_path: str, target_size=(64, 64)):
    # Função que carrega cada imagem do caminho fornecido, redimensiona para 64x64 pixels e normaliza os valores dos pixels para o intervalo [0, 1].
    images = []
    for file_path in image_paths:
        full_path = base_path + "/" + file_path
        img = cv.imread(full_path, cv.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is not None:
            img = cv.resize(img, target_size)  # Resize the image to 64x64 pixels
            img = img / 255.0  # Normalize pixel values to [0, 1]
        else:
            img = np.zeros(target_size)  # If the image cannot be loaded, return a blank image

        plt.imshow(img, cmap="grey")
        plt.show()
        images.append(img)
    return np.array(images)


# def show_image(image_number):
#     # Função que exibe a imagem correspondente ao número fornecido usando Matplotlib. A imagem é carregada usando a função load_images e exibida em escala de cinza.
#     image_array = load_images(df['image_path'].values)
#     plt.imshow(image_array[image_number], cmap='gray')
#     plt.axis('off') 
#     plt.show
