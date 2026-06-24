import json
import pandas as pd
import numpy as np
from time import time_ns
from pathlib import Path

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from treatments.image_manipulation import load_images
from training import create_cnn_model_1, create_cnn_model_2, create_cnn_model_3, training, analyze_training
from metricas.visualizacao import plotar_matriz_top10

def executar_treinamento(caminho_dataset, caminho_pasta_imagens, caminho_json, sigla_pais, arquitetura):
    print(f"\n[MÓDULO DE TREINO] Iniciando isolamento cultural para o país: [{sigla_pais}]")
    
    with open(caminho_json, "r") as js:
        words_to_keep = json.load(js)
    words_to_keep = [w for w in words_to_keep if w.strip()] 
    
    df = pd.read_csv(caminho_dataset)
    df['image_path_normalizado'] = df['image_path'].str.replace(r'^data/', '', regex=True)
    words_to_keep = [w for w in words_to_keep if (Path(caminho_pasta_imagens) / w).exists()]
    
    df_pais = df[df['word'].isin(words_to_keep) & (df['countrycode'] == sigla_pais)].copy()
    
# Filtra estritamente os dados da bolha cultural do país escolhido
    df_pais = df[df['word'].isin(words_to_keep) & (df['countrycode'] == sigla_pais)].copy()
    
    if df_pais.empty:
        raise ValueError(f"Nenhum dado encontrado para o país {sigla_pais}.")

    # --- TRATAMENTO DO PROBLEMA 80/20 ---
    MIN_DESENHOS_POR_CLASSE = 20  # Garante pelo menos 4 desenhos no bloco de validação (20% de 20)
    contagem_por_classe = df_pais['word'].value_counts()
    
    # Descobre quais palavras possuem amostras suficientes para o split
    classes_validas = contagem_por_classe[contagem_por_classe >= MIN_DESENHOS_POR_CLASSE].index.tolist()
    classes_invalidas = contagem_por_classe[contagem_por_classe < MIN_DESENHOS_POR_CLASSE]
    
    # Avisa o usuário caso alguma palavra tenha sido cortada por escassez cultural
    if not classes_invalidas.empty:
        print("\n[AVISO VIES CULTURAL] As seguintes classes foram removidas por falta de dados neste país:")
        for classe, qtd in classes_invalidas.items():
            print(f"  - {classe}: apenas {qtd} desenhos disponíveis (mínimo exigido: {MIN_DESENHOS_POR_CLASSE})")
            
    # Atualiza o dataframe e a lista oficial de classes apenas com o que sobrou
    df_pais = df_pais[df_pais['word'].isin(classes_validas)].copy()
    classes_ordenadas = sorted(classes_validas)
    # -------------------------------------
    
    mapa_classes = {nome: idx for idx, nome in enumerate(classes_ordenadas)}
    
    print(f"-> Carregando {len(df_pais)} imagens do país para a memória...")
    image_array = load_images(df_pais['image_path_normalizado'].values, base_path=str(caminho_pasta_imagens), target_size=(64, 64))
    X_pais = np.stack(image_array)[..., np.newaxis]
    y_pais_num = np.array([mapa_classes[w] for w in df_pais['word'].values])
    y_pais_onehot = to_categorical(y_pais_num, num_classes=len(classes_ordenadas))
    
    X_train, X_val, y_train, y_val = train_test_split(X_pais, y_pais_onehot, test_size=0.2, random_state=42)
    
    if arquitetura == 1:
        model = create_cnn_model_1(input_shape=(64, 64, 1), num_classes=len(classes_ordenadas))
    elif arquitetura == 2:
        model = create_cnn_model_2(input_shape=(64, 64, 1), num_classes=len(classes_ordenadas))
    else:
        model = create_cnn_model_3(input_shape=(64, 64, 1), num_classes=len(classes_ordenadas))
    
    print(f"-> Iniciando arquitetura {arquitetura}...")
    inicio = time_ns()
    hist = training(model, X_train, y_train, X_val, y_val)
    fim = time_ns()
    print(f"-> Treinamento finalizado em {(fim - inicio) / 1e9:.2f}s")
    
    caminho_output = Path("./output")
    caminho_output.mkdir(exist_ok=True)
    
    model.save(caminho_output / f"modelo_{sigla_pais}.keras")
    with open(caminho_output / f"classes_{sigla_pais}.json", "w") as f:
        json.dump(classes_ordenadas, f)
    analyze_training(hist, caminho_output, sigla_pais)
    
    print("-> Gerando Matriz de Viés Interno...")
    pred_pais = model.predict(X_pais, verbose=0)
    y_prev_pais = np.argmax(pred_pais, axis=1)
    plotar_matriz_top10(
        y_pais_num, y_prev_pais, classes_ordenadas,
        f"Viés Interno ({sigla_pais}): Confusões na própria cultura",
        caminho_output / f"matriz_1_TREINO_interno_{sigla_pais}.png"
    )
    print("[MÓDULO DE TREINO CONCLUÍDO]")