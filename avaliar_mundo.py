import json
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

from treatments.image_manipulation import load_images
from metricas.visualizacao import plotar_matriz_top10

def executar_avaliacao_global(caminho_dataset, caminho_pasta_imagens, sigla_pais):
    print(f"\n[MÓDULO DE AVALIAÇÃO] Iniciando choque cultural (Modelo {sigla_pais} vs Mundo)")
    
    caminho_output = Path("./output")
    arquivo_modelo = caminho_output / f"modelo_{sigla_pais}.keras"
    arquivo_classes = caminho_output / f"classes_{sigla_pais}.json"
    
    with open(arquivo_classes, "r") as f:
        classes_ordenadas = json.load(f)
        
    df = pd.read_csv(caminho_dataset)
    df['image_path_normalizado'] = df['image_path'].str.replace(r'^data/', '', regex=True)
    
    df_mundo = df[df['word'].isin(classes_ordenadas) & (df['countrycode'] != sigla_pais)].copy()
    
    print(f"-> Carregando {len(df_mundo)} imagens globais...")
    mapa_classes = {nome: idx for idx, nome in enumerate(classes_ordenadas)}
    image_array = load_images(df_mundo['image_path_normalizado'].values, base_path=str(caminho_pasta_imagens), target_size=(64, 64))
    X_mundo = np.stack(image_array)[..., np.newaxis]
    y_mundo_num = np.array([mapa_classes[w] for w in df_mundo['word'].values])

    print(f"-> Carregando a rede neural [{sigla_pais}]...")
    model = load_model(arquivo_modelo)
    
    pred_mundo = model.predict(X_mundo, verbose=0)
    y_prev_mundo = np.argmax(pred_mundo, axis=1)
    
    acc_global = accuracy_score(y_mundo_num, y_prev_mundo)
    print(f"\n>>> Acurácia OOD (Out-of-Distribution): {acc_global*100:.2f}%")
    
    print("-> Gerando Matriz do Choque Cultural...")
    plotar_matriz_top10(
        y_mundo_num, y_prev_mundo, classes_ordenadas,
        f"Choque Cultural: IA do [{sigla_pais}] avaliando o Resto do Mundo",
        caminho_output / f"matriz_2_TESTE_mundo_vs_{sigla_pais}.png"
    )
    print("[MÓDULO DE AVALIAÇÃO CONCLUÍDO]")