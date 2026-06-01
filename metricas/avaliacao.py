import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv, exit
from pathlib import Path

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from image_manipulation import load_images
from training import encode_labels

def plotar_piores_confusoes(y_verdadeiro, y_previsto, nomes_das_classes, top_n=10):
    matriz_completa = confusion_matrix(y_verdadeiro, y_previsto)
    np.fill_diagonal(matriz_completa, 0) 
    
    erros_totais = matriz_completa.sum(axis=1) + matriz_completa.sum(axis=0)
    
    top_n = min(top_n, len(nomes_das_classes)) 
    indices_problematicos = erros_totais.argsort()[-top_n:][::-1]
    
    matriz_reduzida = matriz_completa[np.ix_(indices_problematicos, indices_problematicos)]
    nomes_reduzidos = [nomes_das_classes[i] for i in indices_problematicos]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_reduzida, annot=True, fmt='d', cmap='Reds',
                xticklabels=nomes_reduzidos, yticklabels=nomes_reduzidos)
    
    plt.title(f'Top {top_n} Maiores Confusões do Modelo', fontsize=16, pad=20)
    plt.ylabel('Classe Real (Gabarito)')
    plt.xlabel('Previsão da CNN')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plotar_curvas_aprendizado(caminho_historico, titulo="Curvas de Aprendizado"):
    with open(caminho_historico, "r") as f:
        hist = json.load(f)
        
    epocas = range(1, len(hist['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    sns.set_theme(style="whitegrid")
    
    plt.subplot(1, 2, 1)
    plt.plot(epocas, hist['loss'], label='Treino', color='blue', linestyle='--')
    plt.plot(epocas, hist['val_loss'], label='Validação', color='red', linewidth=2)
    plt.title(f'{titulo} - Erro (Loss)')
    plt.xlabel('Épocas')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epocas, hist['accuracy'], label='Treino', color='blue', linestyle='--')
    plt.plot(epocas, hist['val_accuracy'], label='Validação', color='red', linewidth=2)
    plt.title(f'{titulo} - Acurácia')
    plt.xlabel('Épocas')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    if len(argv) < 4:
        print("Uso: python avaliacao.py <caminho-dataset.csv> <caminho-pasta-imagens> <json-words-to-keep> <numero_do_modelo>")
        print("Exemplo: python avaliacao.py dados.csv imagens/ words_to_keep.json 1")
        exit(-1)

    caminho_dataset = Path(argv[1])
    caminho_pasta_imagens = Path(argv[2])
    caminho_json = Path(argv[3])
    
    num_modelo = argv[4] if len(argv) > 4 else "1"

    with open(caminho_json, "r") as js:
        words_to_keep = json.load(js)

    print("Carregando imagens para os gráficos...")
    df = pd.read_csv(caminho_dataset)
    df_filtrado = df[df['word'].isin(words_to_keep)].copy()
    
    df_filtrado['image_path_normalizado'] = df_filtrado['image_path'].str.replace(r'^data/', '', regex=True)
    words_to_keep = [w for w in words_to_keep if (caminho_pasta_imagens / w).exists()]
    df_filtrado = df_filtrado[df_filtrado['word'].isin(words_to_keep)]
    
    image_array = load_images(df_filtrado['image_path_normalizado'].values, base_path=str(caminho_pasta_imagens), target_size=(64, 64))

    _, X_validate, _, y_validate = encode_labels(df_filtrado['word'].values, image_array, words_to_keep)

    print(f"Carregando Modelo {num_modelo}...")
    model = load_model(f"output/modelo{num_modelo}.keras")

    print("Exibindo Curvas de Aprendizado...")
    plotar_curvas_aprendizado(f"output/historico_modelo{num_modelo}.json", f"Modelo {num_modelo}")

    print("Gerando predições para a Matriz de Confusão...")
    probabilidades = model.predict(X_validate)
    
    y_previsto_indices = np.argmax(probabilidades, axis=1)
    y_real_indices = np.argmax(y_validate, axis=1)

    nomes_ordenados = sorted(words_to_keep)

    print("Plotando Matriz de Confusão Cirúrgica...")
    plotar_piores_confusoes(y_real_indices, y_previsto_indices, nomes_ordenados, top_n=10)

if __name__ == "__main__":
    main()