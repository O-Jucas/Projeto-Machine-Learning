import json
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from sys import argv, exit
from pathlib import Path

from treatments.image_manipulation import load_images
from training2 import encode_labels, create_cnn_model_1, create_cnn_model_2, create_cnn_model_3, training, analyze_training

from time import time_ns

modelo_treinado = 2

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
    df_filtrado = df[df['word'].isin(words_to_keep)].copy()

    # Remove do path o prefixo 'data/' que o CSV armazena mas não existe no disco
    df_filtrado['image_path_normalizado'] = df_filtrado['image_path'].str.replace(r'^data/', '', regex=True)

    # Filtra apenas categorias cujas pastas existem no disco
    words_to_keep = [w for w in words_to_keep if (caminho_pasta_imagens / w).exists()]
    df_filtrado = df_filtrado[df_filtrado['word'].isin(words_to_keep)]
    print(f"Categorias disponíveis no disco ({len(words_to_keep)}): {words_to_keep}")

    assert len(words_to_keep) > 0, "Nenhuma categoria do words_to_keep possui pasta no disco"

    image_array = load_images(df_filtrado['image_path_normalizado'].values, base_path=str(caminho_pasta_imagens), target_size=(64, 64))

    # Cria as colunas target numéricas com base no target string
    X_train, X_validate, y_train, y_validate = encode_labels(df_filtrado['word'].values, image_array, words_to_keep)

    if modelo_treinado != None:
        inicio = time_ns()
        
        # O hist vai capturar o retorno da função training()
        if modelo_treinado == 1:
            model1 = create_cnn_model_1(input_shape=(64, 64, 1), num_classes=len(words_to_keep))
            hist = training(model1, X_train, y_train, X_validate, y_validate)
            model = model1
            
        elif modelo_treinado == 2:
            model2 = create_cnn_model_2(input_shape=(64, 64, 1), num_classes=len(words_to_keep))
            hist = training(model2, X_train, y_train, X_validate, y_validate)
            model = model2

        elif modelo_treinado == 3:
            model3 = create_cnn_model_3(input_shape=(64, 64, 1), num_classes=len(words_to_keep))
            hist = training(model3, X_train, y_train, X_validate, y_validate)
            model = model3
            
        fim = time_ns()
        print(f"Modelo {modelo_treinado} -> {(fim - inicio) / 1e9:.2f}s")

        caminho_output = Path("./output")
        caminho_output.mkdir(exist_ok=True)

        # Salva o modelo, o histórico e os gráficos
        model.save(caminho_output / f"modelo{modelo_treinado}.keras")
        with open(caminho_output / f"historico_modelo{modelo_treinado}.json", "w") as f:
            json.dump(hist, f)
        analyze_training(hist, caminho_output, modelo_treinado)
            
    else:
        # Cria modelo CNN
        model1 = create_cnn_model_1(input_shape=(64, 64, 1), num_classes=len(words_to_keep))
        model2 = create_cnn_model_2(input_shape=(64, 64, 1), num_classes=len(words_to_keep))
        model3 = create_cnn_model_3(input_shape=(64, 64, 1), num_classes=len(words_to_keep))

        # Treina modelo e apresenta resultados
        print("Treinando, bip bop")

        inicio = time_ns()
        hist1 = training(model1, X_train, y_train, X_validate, y_validate)
        fim = time_ns()
        print(f"Modelo 1 -> {(fim - inicio) / 1e9:.2f}s")
        
        hist2 = training(model2, X_train, y_train, X_validate, y_validate)
        fim2 = time_ns()
        print(f"Modelo 2 -> {(fim2 - fim) / 1e9:.2f}s")
        
        hist3 = training(model3, X_train, y_train, X_validate, y_validate)
        fim3 = time_ns()
        print(f"Modelo 3 -> {(fim3 - fim2) / 1e9:.2f}s")
        print("Fim treino, zip zup")

        print(f"Tempo total -> {(fim3 - inicio) / 1e9:.2f}s")

        caminho_output = Path("./output")
        caminho_output.mkdir(exist_ok=True)

        modelos = [model1, model2, model3]
        historicos = [hist1, hist2, hist3]

        for i in range(3):
            modelos[i].save(caminho_output / f"modelo{i + 1}.keras")
            with open(caminho_output / f"historico_modelo{i + 1}.json", "w") as f:
                json.dump(historicos[i], f)
            analyze_training(historicos[i], caminho_output, i + 1)

if __name__ == "__main__":
    main()