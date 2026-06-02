import json
import numpy as np
import cv2 as cv

from sys import argv, exit
from pathlib import Path
from tensorflow.keras.models import load_model


def preprocessar_imagem(caminho_imagem: str, target_size=(64, 64)) -> np.ndarray:
    raw = np.fromfile(caminho_imagem, dtype=np.uint8)
    img = cv.imdecode(raw, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho_imagem}")
    img = cv.resize(img, target_size)
    img = img / 255.0
    return img[np.newaxis, ..., np.newaxis]  # (1, 64, 64, 1)


def main():
    if len(argv) < 3:
        print("Uso: python predicao.py <imagem.png> <numero-do-modelo>")
        print("Exemplo: python predicao.py desenho.png 1")
        exit(-1)

    caminho_imagem = Path(argv[1])
    assert caminho_imagem.exists(), f"Imagem não encontrada: {caminho_imagem}"

    numero_modelo = argv[2]

    caminho_modelo = Path("./output") / f"modelo{numero_modelo}.keras"
    assert caminho_modelo.exists(), f"Modelo não encontrado: {caminho_modelo} — rode o run.py primeiro"

    caminho_classes = Path("./output") / f"classes_modelo{numero_modelo}.json"
    assert caminho_classes.exists(), f"Classes não encontradas: {caminho_classes} — retreine o modelo para gerar este arquivo"

    with open(caminho_classes, "r") as f:
        classes = json.load(f)

    print(f"Carregando modelo {numero_modelo}...")
    model = load_model(caminho_modelo)

    print(f"Processando imagem: {caminho_imagem.name}")
    X = preprocessar_imagem(str(caminho_imagem))

    probabilidades = model.predict(X, verbose=0)[0]
    indice = np.argmax(probabilidades)
    classe_prevista = classes[indice]
    confianca = probabilidades[indice] * 100

    print(f"\nPrevisão: {classe_prevista} ({confianca:.1f}% de confiança)")
    print("\nTop 3:")
    top3 = np.argsort(probabilidades)[::-1][:3]
    for i in top3:
        print(f"  {classes[i]:<20} {probabilidades[i] * 100:.1f}%")


if __name__ == "__main__":
    main()
