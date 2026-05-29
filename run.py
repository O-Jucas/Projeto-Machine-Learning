import json
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")

from sys import argv, exit
from pathlib import Path

from image_manipulation import load_images
from training import encode_labels, create_cnn_model_1, create_cnn_model_2, create_cnn_model_3, training


def main():

    if len(argv) < 4:
        print("Insira o caminho das imagens para o treino do modelo")
        print(f"Forma de uso: python {argv[0]} <caminho-master-doodle-dataframe> <caminho-pasta-imagens> <json-words-to-keep>")
        exit(-1)

    # Verificando paths corretos

    caminho_dataset = Path(argv[1])
    assert caminho_dataset.exists(), "Caminho para o csv \"caminho-master-doodle-dataframe.csv\" incorreto"

    caminho_pasta_imagens = Path(argv[2])
    assert caminho_pasta_imagens.exists(), "Caminho para a pasta de imagens incorreto"

    caminho_json = Path(argv[3])
    assert caminho_json.exists(), "Caminho para o json de palavras selecionadas incorreto"

    # Abrindo json palavras

    with open(caminho_json, "r") as js:
        words_to_keep = json.load(js)

    assert isinstance(words_to_keep, list), "words_to_keep não é uma lista"
    assert len(words_to_keep) > 0, "lista words_to_keep é vazio"

    df = pd.read_csv(caminho_dataset)
    df_filtrado = df[df['word'].isin(words_to_keep)]

    # df_filtrado["image_path_corrigido"] = df_filtrado["image_path"]
    image_array = load_images(df_filtrado['image_path'].values, base_path=str(caminho_pasta_imagens), target_size=(64, 64))

    # Cria as colunas target numéricas com base no target string
    X_train, X_validate, y_train, y_validate = encode_labels(df_filtrado['word'].values, image_array, words_to_keep)

    # Cria modelo CNN
    model1 = create_cnn_model_1(input_shape=(64, 64, 1), num_classes=len(words_to_keep))
    model2 = create_cnn_model_2(input_shape=(64, 64, 1), num_classes=len(words_to_keep))
    model3 = create_cnn_model_3(input_shape=(64, 64, 1), num_classes=len(words_to_keep))

    # Treina modelo e apresenta resultados
    print("Treinando, bip bop")
    training(model1, X_train, y_train, X_validate, y_validate)
    training(model2, X_train, y_train, X_validate, y_validate)
    training(model3, X_train, y_train, X_validate, y_validate)
    print("Fim treino, zip zup")




if __name__ == "__main__":
    main()
