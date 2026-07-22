import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv, exit
from pathlib import Path

from sklearn.metrics import classification_report, top_k_accuracy_score, accuracy_score
from tensorflow.keras.models import load_model

from image_manipulation import load_images
from training import encode_labels

def plotar_graficos_metricas(report_dict, top1, top5, nomes_das_classes):
    """Gera um painel visual com o resumo das métricas principais."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Gráfico de Barras: Acurácia Global (Top-1 vs Top-5)
    barras = axes[0].bar(['Top-1 (Cravar a palavra)', 'Top-5 (Estar entre as 5)'], 
                         [top1, top5], color=['#ff6666', '#66b3ff'])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Acurácia Global do Modelo', fontsize=14, pad=15)
    axes[0].set_ylabel('Acurácia')
    
    # Adicionando a porcentagem em cima das barras
    for barra in barras:
        altura = barra.get_height()
        axes[0].text(barra.get_x() + barra.get_width()/2., altura + 0.02,
                     f"{altura*100:.1f}%", ha='center', fontsize=12, fontweight='bold')

    # 2. Gráfico de Barras Horizontais: F1-Score por Categoria
    f1_scores = [report_dict[nome]['f1-score'] for nome in nomes_das_classes]
    
    sns.barplot(x=f1_scores, y=nomes_das_classes, ax=axes[1], hue=nomes_das_classes, palette="viridis", legend=False)
    axes[1].set_title('F1-Score por Categoria (Desempenho por Palavra)', fontsize=14, pad=15)
    axes[1].set_xlabel('F1-Score (0.0 a 1.0)')
    axes[1].set_xlim(0, 1.05)

    plt.tight_layout()
    plt.show()

def main():
    if len(argv) < 4:
        print("Uso: python metricas.py <caminho-dataset.csv> <caminho-pasta-imagens> <json-words-to-keep> <numero_do_modelo>")
        print("Exemplo: python metricas.py dados.csv imagens/ words_to_keep.json 1")
        exit(-1)

    caminho_dataset = Path(argv[1])
    caminho_pasta_imagens = Path(argv[2])
    caminho_json = Path(argv[3])
    
    num_modelo = argv[4] if len(argv) > 4 else "1"

    with open(caminho_json, "r") as js:
        words_to_keep = json.load(js)

    print("Carregando imagens para avaliação de métricas...")
    df = pd.read_csv(caminho_dataset)
    df_filtrado = df[df['word'].isin(words_to_keep)].copy()
    
    # Tratamento de caminhos para o Windows
    df_filtrado['image_path_normalizado'] = df_filtrado['image_path'].str.replace(r'^data/', '', regex=True)
    words_to_keep = [w for w in words_to_keep if (caminho_pasta_imagens / w).exists()]
    df_filtrado = df_filtrado[df_filtrado['word'].isin(words_to_keep)]
    
    image_array = load_images(df_filtrado['image_path_normalizado'].values, base_path=str(caminho_pasta_imagens), target_size=(64, 64))

    _, X_validate, _, y_validate = encode_labels(df_filtrado['word'].values, image_array, words_to_keep)

    print(f"Carregando Modelo {num_modelo}...")
    model = load_model(f"output/modelo{num_modelo}.keras")

    print("Gerando predições...")
    probabilidades = model.predict(X_validate)
    
    y_previsto_indices = np.argmax(probabilidades, axis=1)
    y_real_indices = np.argmax(y_validate, axis=1)

    # Calculando as métricas brutas
    top1_acc = accuracy_score(y_real_indices, y_previsto_indices)
    top5_acc = top_k_accuracy_score(y_real_indices, probabilidades, k=5)
    nomes_ordenados = sorted(words_to_keep)
    
    # Gerando o relatório duas vezes: uma para printar e outra como dicionário para o gráfico
    report_texto = classification_report(y_real_indices, y_previsto_indices, target_names=nomes_ordenados)
    report_dict = classification_report(y_real_indices, y_previsto_indices, target_names=nomes_ordenados, output_dict=True)

    # Imprime no terminal
    print("\n" + "="*50)
    print(f"MÉTRICAS DE DESEMPENHO (Modelo {num_modelo})")
    print("="*50)
    print(f">>> ACURÁCIA TOP-1: {top1_acc * 100:.2f}%")
    print(f">>> ACURÁCIA TOP-5: {top5_acc * 100:.2f}%\n")
    print(report_texto)

    # Abre a janela com os gráficos
    print("\nAbrindo painel gráfico de métricas...")
    plotar_graficos_metricas(report_dict, top1_acc, top5_acc, nomes_ordenados)

if __name__ == "__main__":
    main()