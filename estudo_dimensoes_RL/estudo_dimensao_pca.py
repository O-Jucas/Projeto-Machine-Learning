"""
Estudo de dimensão com PCA + Regressão Logística.

Reduz os 64x64 = 4096 pixels de cada desenho para k componentes via PCA
(k = 10, 20, 50, ...) e treina uma regressão logística em cada caso, comparando
com o baseline (logística sobre todos os 4096 pixels originais).

Uso:
    python estudo_dimensao_pca.py <caminho-master-doodle-dataframe.csv> <pasta-imagens> <words_to_keep.json>
"""

import json
import time
from pathlib import Path
from sys import argv, exit

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# image_manipulation.py mora em ../treatments relativo a este script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "treatments"))
from image_manipulation import load_images


# ----- Parâmetros do estudo (edite à vontade) -----
COMPONENTES = [10, 20, 30, 50, 100, 200]  # valores de k para o PCA
TARGET_SIZE = (64, 64)                     # resolução original da imagem
AMOSTRAS_POR_CLASSE = None                 # ex.: 3000 para rodar mais rápido; None = usa tudo


def carregar_dados(caminho_dataset, caminho_pasta_imagens, words_to_keep):
    df = pd.read_csv(caminho_dataset)
    df_filtrado = df[df["word"].isin(words_to_keep)]

    # Opcional: limita amostras por classe para acelerar o experimento.
    # Embaralha e pega os primeiros N de cada classe — robusto no pandas 3.0, onde
    # groupby.apply descartaria a coluna "word"; .head() ainda faz o cap automatico.
    if AMOSTRAS_POR_CLASSE is not None:
        df_filtrado = (
            df_filtrado.sample(frac=1, random_state=42)
            .groupby("word", group_keys=False)
            .head(AMOSTRAS_POR_CLASSE)
        )

    # No CSV o caminho é "data/<classe>/<arquivo>", mas a pasta doodle tem as classes
    # direto (sem subpasta "data/"). Removemos o prefixo para casar com a pasta real.
    caminhos = (
        pd.Series(df_filtrado["image_path"].values)
        .str.replace(r"^data[\\/]", "", regex=True)
        .values
    )
    imagens = load_images(
        caminhos,
        base_path=str(caminho_pasta_imagens),
        target_size=TARGET_SIZE,
    )

    # Diagnóstico: imagens que falham ao carregar viram zeros (ver image_manipulation.py).
    # Uma imagem real nunca é toda zero, então sum == 0 indica caminho errado.
    planas = imagens.reshape(len(imagens), -1)
    falhas = int(np.sum(planas.sum(axis=1) == 0))
    if falhas:
        print(f"[AVISO] {falhas}/{len(imagens)} imagens estão totalmente vazias "
              f"(provável caminho incorreto). Corrija antes de confiar no resultado.")

    X = planas.astype(np.float32)  # (N, 4096), float32 economiza metade da RAM

    encoder = LabelEncoder()
    y = encoder.fit_transform(df_filtrado["word"].values)

    print(f"Carregado: {X.shape[0]} imagens, {X.shape[1]} features, {len(encoder.classes_)} classes")
    return X, y, encoder


def treinar_logistica(rotulo, X_train, X_test, y_train, y_test):
    inicio = time.time()
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    tempo = time.time() - inicio

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    res = {
        "dimensoes": rotulo,
        "n_features": X_train.shape[1],
        "acc_treino": accuracy_score(y_train, pred_train),
        "acc_teste": accuracy_score(y_test, pred_test),
        "f1_macro_teste": f1_score(y_test, pred_test, average="macro"),
        "tempo_s": round(tempo, 2),
    }
    print(f"  k={rotulo:>10} | features={res['n_features']:>5} | "
          f"acc_teste={res['acc_teste']:.3f} | f1={res['f1_macro_teste']:.3f} | {res['tempo_s']}s")
    return res


def rodar_estudo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Padroniza com fit SÓ no treino (não vaza o teste)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    resultados = []

    # Baseline: logística sobre todos os pixels (sem PCA)
    print("Baseline (todos os pixels):")
    resultados.append(treinar_logistica("todos", X_train_s, X_test_s, y_train, y_test))

    # PCA: ajusta UMA vez com o maior k; os primeiros k componentes são os mesmos
    # para qualquer truncamento, então é só fatiar as colunas.
    k_max = max(COMPONENTES)
    pca = PCA(n_components=k_max, random_state=42)
    X_train_full = pca.fit_transform(X_train_s)
    X_test_full = pca.transform(X_test_s)

    print("PCA + logística:")
    for k in COMPONENTES:
        res = treinar_logistica(k, X_train_full[:, :k], X_test_full[:, :k], y_train, y_test)
        res["variancia_explicada"] = float(pca.explained_variance_ratio_[:k].sum())
        resultados.append(res)

    return pd.DataFrame(resultados), pca


def plotar(df_result, pca, caminho_output):
    # 1) Acurácia x número de componentes
    df_pca = df_result[df_result["dimensoes"] != "todos"]
    baseline = df_result[df_result["dimensoes"] == "todos"].iloc[0]

    fig, ax = plt.subplots()
    ax.plot(df_pca["n_features"], df_pca["acc_treino"], "o-", label="treino (PCA)")
    ax.plot(df_pca["n_features"], df_pca["acc_teste"], "o-", label="teste (PCA)")
    ax.axhline(baseline["acc_teste"], color="gray", linestyle="--",
               label=f"baseline 4096px (teste={baseline['acc_teste']:.2f})")
    ax.set_xlabel("nº de componentes do PCA")
    ax.set_ylabel("acurácia")
    ax.set_title("Acurácia x dimensão (PCA + Regressão Logística)")
    ax.legend()
    fig.savefig(caminho_output / "acuracia_x_dimensao.png", dpi=120, bbox_inches="tight")

    # 2) Variância explicada acumulada
    fig2, ax2 = plt.subplots()
    acumulada = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(acumulada) + 1), acumulada, "-")
    ax2.set_xlabel("nº de componentes")
    ax2.set_ylabel("variância explicada acumulada")
    ax2.set_title("Quanta informação cada componente retém")
    ax2.grid(True, alpha=0.3)
    fig2.savefig(caminho_output / "variancia_explicada.png", dpi=120, bbox_inches="tight")

    plt.show()


def main():
    if len(argv) < 4:
        print(f"Uso: python {argv[0]} <master_doodle_dataframe.csv> <pasta-imagens> <words_to_keep.json>")
        exit(-1)

    caminho_dataset = Path(argv[1])
    assert caminho_dataset.exists(), "Caminho do CSV incorreto"
    caminho_pasta_imagens = Path(argv[2])
    assert caminho_pasta_imagens.exists(), "Caminho da pasta de imagens incorreto"
    caminho_json = Path(argv[3])
    assert caminho_json.exists(), "Caminho do JSON de palavras incorreto"

    with open(caminho_json, "r") as js:
        words_to_keep = json.load(js)
    assert isinstance(words_to_keep, list) and len(words_to_keep) > 0, "words_to_keep inválido"

    X, y, _ = carregar_dados(caminho_dataset, caminho_pasta_imagens, words_to_keep)
    df_result, pca = rodar_estudo(X, y)

    caminho_output = Path("./output")
    caminho_output.mkdir(exist_ok=True)
    df_result.to_csv(caminho_output / "resultado_pca_logistica.csv", index=False)
    print("\nResultados salvos em output/resultado_pca_logistica.csv")
    print(df_result.to_string(index=False))

    plotar(df_result, pca, caminho_output)


if __name__ == "__main__":
    main()
