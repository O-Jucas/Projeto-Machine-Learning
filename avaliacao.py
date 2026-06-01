import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importa as métricas do Scikit-Learn
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import pickle # Caso ainda esteja usando o pickle

# Importa as funções que você já criou no seu projeto
from image_manipulation import load_images
from training import encode_labels

# ---------------------------------------------------------
# 1. FUNÇÃO DA MATRIZ DE CONFUSÃO 
# ---------------------------------------------------------
def plotar_piores_confusoes(y_verdadeiro, y_previsto, nomes_das_classes, top_n=10):
    matriz_completa = confusion_matrix(y_verdadeiro, y_previsto)
    np.fill_diagonal(matriz_completa, 0) # Zera os acertos
    
    erros_totais = matriz_completa.sum(axis=1) + matriz_completa.sum(axis=0)
    
    # Pega só o Top N de erros (limita pelo tamanho real de classes que temos)
    top_n = min(top_n, len(nomes_das_classes)) 
    indices_problematicos = erros_totais.argsort()[-top_n:][::-1]
    
    matriz_reduzida = matriz_completa[np.ix_(indices_problematicos, indices_problematicos)]
    nomes_reduzidos = [nomes_das_classes[i] for i in indices_problematicos]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_reduzida, annot=True, fmt='d', cmap='Reds',
                xticklabels=nomes_reduzidos, yticklabels=nomes_reduzidos)
    
    plt.title(f'Top {top_n} Maiores Confusões do Modelo', fontsize=16)
    plt.ylabel('Classe Real (Gabarito)')
    plt.xlabel('Previsão da CNN')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 2. SCRIPT PRINCIPAL DE AVALIAÇÃO
# ---------------------------------------------------------
def main():
    # 1. Carregar as mesmas palavras e dados usados no treino
    with open("../arquivos machine learning/words_to_keep.json", "r") as js:
        words_to_keep = json.load(js)
        
    df = pd.read_csv("../arquivos machine learning/master_doodle_dataframe.csv")
    df_filtrado = df[df['word'].isin(words_to_keep)].copy()
    df_filtrado['image_path_normalizado'] = df_filtrado['image_path'].str.replace(r'^data/', '', regex=True)

    caminho_imagens = "../arquivos machine learning/doodle"

    # Filtra apenas categorias cujas pastas existem no disco (igual ao run.py)
    words_to_keep = [w for w in words_to_keep if (Path(caminho_imagens) / w).exists()]
    df_filtrado = df_filtrado[df_filtrado['word'].isin(words_to_keep)]

    print("Carregando imagens para teste...")
    image_array = load_images(df_filtrado['image_path_normalizado'].values, base_path=caminho_imagens, target_size=(64, 64))
    
    # 2. Recriar os tensores de teste
    _, X_validate, _, y_validate = encode_labels(df_filtrado['word'].values, image_array, words_to_keep)
    
    # 3. Carregar o Modelo 1 salvo
    print("Carregando o modelo...")
    # Se usou model.save() -> model = load_model("output/modelo1.keras")
    with open("output/modelo1", "rb") as pkl:
        model = pickle.load(pkl)
    
    # 4. A HORA DA VERDADE: Fazer as predições
    print("Gerando predições...")
    probabilidades = model.predict(X_validate)
    
    # A CNN cospe 10 probabilidades (uma para cada palavra). 
    # O argmax pega o índice da maior probabilidade (o chute final)
    y_previsto_indices = np.argmax(probabilidades, axis=1)
    
    # O y_validate original está em One-Hot Encoding (ex: [0, 0, 1, 0...]). 
    # Precisamos converter de volta para índice
    y_real_indices = np.argmax(y_validate, axis=1)
    
    # 5. GERAR OS RELATÓRIOS
    print("\n--- RELATÓRIO DE DESEMPENHO (F1-Score, Precision, Recall) ---")
    # Como as classes foram ordenadas alfabeticamente pelo LabelEncoder, 
    # precisamos garantir que os nomes batam com os índices
    nomes_ordenados = sorted(words_to_keep)
    print(classification_report(y_real_indices, y_previsto_indices, target_names=nomes_ordenados))
    
    print("\nGerando Matriz de Confusão...")
    plotar_piores_confusoes(y_real_indices, y_previsto_indices, nomes_ordenados, top_n=10)

if __name__ == "__main__":
    main()