import pickle
from sys import argv, exit
from pathlib import Path

import numpy as np

from image_manipulation import load_images

def main():

    if (len(argv) != 3):
        print("Número de Argumentos Inválido")
        print(f"Forma de uso: python {argv[0]} <cainho-modelo> <caminho-pasta-imagem>")
        exit(1)

    caminho_modelo = Path(argv[1])
    assert caminho_modelo.exists(), "Caminho do modelo não encontrado"

    caminho_pasta_imagem = Path(argv[2])
    assert caminho_pasta_imagem.exists(), "Caminho do imagem não encontrado"

    with open(caminho_modelo, "rb") as modelo:
        cnn = pickle.load(modelo)

    assert hasattr(cnn, "predict"), "Arquivo importado não é um modelo"

    imagem = load_images(["cachorro quente.png"], str(caminho_pasta_imagem), (64, 64))

    print(np.min(imagem), np.max(imagem))
    vetor_respostas = cnn.predict(imagem)

    print(np.argmax(vetor_respostas), np.amax(vetor_respostas))
    print(vetor_respostas)

    lista = ['tree', 'grass', 'penguin', 'horse', 'hexagon', 'envelope', 'eraser', 'finger', 'face', 'fan', 'flip flops', 'flamingo', 'fork', 'hospital', 'helmet', 'hat', 'hot dog', 'hot tub', 'mosquito', 'mouse', 'moustache', 'pillow', 'river', 'scissors', 'shoe', 'shark', 'square', 'stairs', 'television', 'toe', 'submarine', 'teapot', 'toilet', 'umbrella', 'violin', 'watermelon', 'yoga']

    print(lista[np.argmax(vetor_respostas)])

if __name__ == "__main__":
    main()
