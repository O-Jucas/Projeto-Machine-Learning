import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sys import argv, exit
from pathlib import Path
from time import time_ns

from tensorflow.keras.utils import to_categorical

from treatments.image_manipulation import load_images
from training import encode_labels, create_cnn_model_1, create_cnn_model_2, create_cnn_model_3, training
from training2 import training as training_retorna_historico, analyze_training


def carregar_imagens_de_pasta(df, caminho_pasta_imagens, words_to_keep):
    """Filtra o df mestre pelas imagens que existem fisicamente em `caminho_pasta_imagens`
    e retorna (image_array, labels). A pasta (train ou teste) define a qual conjunto a
    imagem pertence: só sobrevivem as entradas do CSV cujo arquivo existe no disco."""
    df_local = df[df['word'].isin(words_to_keep)].copy()

    df_local['image_path_normalizado'] = (
        df_local['image_path']
        .astype(str)
        .str.replace(r'^data[\\/]', '', regex=True)
        .str.replace(r'\\', '/', regex=True)
    )
    df_local['caminho_imagem'] = df_local['image_path_normalizado'].apply(
        lambda p: caminho_pasta_imagens / Path(p)
    )
    df_local = df_local[df_local['caminho_imagem'].apply(lambda p: p.exists())]

    assert len(df_local) > 0, f"Nenhuma imagem válida encontrada em {caminho_pasta_imagens}"

    image_array = load_images(df_local['caminho_imagem'].values, target_size=(64, 64))
    return image_array, df_local['word'].values


def montar_split_por_pasta(train_imgs, train_labels, val_imgs, val_labels, words_to_keep):
    """Monta X/y de treino e validação a partir de duas pastas distintas, usando uma
    ordem de classes fixa (sorted(words_to_keep)) — a mesma que predicao.py/metricas.py
    assumem, garantindo que os índices das classes sejam consistentes em tudo."""
    classes = sorted(words_to_keep)
    num_classes = len(classes)
    indice_da_classe = {c: i for i, c in enumerate(classes)}

    X_train = np.stack(train_imgs)[..., np.newaxis]
    X_validate = np.stack(val_imgs)[..., np.newaxis]

    y_train = to_categorical([indice_da_classe[w] for w in train_labels], num_classes=num_classes)
    y_validate = to_categorical([indice_da_classe[w] for w in val_labels], num_classes=num_classes)

    return X_train, X_validate, y_train, y_validate


# Por favor, selecione o modelo a ser treinado ou None para treinar todos os modelos
modelo_treinado = 2

# Selecione também qual benchmark deseja realizar, ou None para não realizar nenhum benchmark
# Opções: None | "number_of_neurons" | "number_of_layers"
benchmark = "number_of_layers"


def run_benchmark_neuronios(X_train, y_train, X_validate, y_validate, classes, caminho_output):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    from Benchmarks.trn_numero_neuronios import (
        create_cnn_2_256,
        create_cnn_2_512,
        create_cnn_2_1024,
        create_cnn_2_2048,
    )

    num_classes = len(classes)

    # Subpastas: curvas/CSV em treinamento/, modelos salvos em modelos/
    caminho_treinamento = caminho_output / "treinamento"
    caminho_modelos = caminho_output / "modelos"
    caminho_treinamento.mkdir(parents=True, exist_ok=True)
    caminho_modelos.mkdir(parents=True, exist_ok=True)

    variantes = [
        ("256",  create_cnn_2_256),
        ("512",  create_cnn_2_512),
        ("1024", create_cnn_2_1024),
        ("2048", create_cnn_2_2048),
    ]

    resultados = []

    for nome, factory in variantes:
        print(f"\n--- Benchmark Modelo 2 | neurônios por camada Dense = {nome} ---")
        model = factory(input_shape=(64, 64, 1), num_classes=num_classes)

        inicio = time_ns()
        history = training_retorna_historico(model, X_train, y_train, X_validate, y_validate)
        fim = time_ns()
        tempo_ms = (fim - inicio) / 1e6

        val_loss = min(history["val_loss"])
        epocas   = len(history["val_accuracy"])

        # Como EarlyStopping(restore_best_weights=True), o modelo guarda os melhores pesos.
        # Avaliamos as predições no conjunto de validação para extrair as métricas.
        probabilidades = model.predict(X_validate, verbose=0)
        y_previsto = np.argmax(probabilidades, axis=1)
        y_real     = np.argmax(y_validate, axis=1)

        acc       = accuracy_score(y_real, y_previsto)
        recall    = recall_score(y_real, y_previsto, average="macro", zero_division=0)
        precisao  = precision_score(y_real, y_previsto, average="macro", zero_division=0)
        f1        = f1_score(y_real, y_previsto, average="macro", zero_division=0)
        # L1 / MAE: erro absoluto médio entre o vetor de probabilidades e o rótulo one-hot
        l1_mae    = float(np.mean(np.abs(probabilidades - y_validate)))

        resultados.append({
            "neurônios":     nome,
            "val_accuracy":  round(acc, 4),
            "sensibilidade": round(recall, 4),
            "precisao":      round(precisao, 4),
            "f1":            round(f1, 4),
            "L1_mae":        round(l1_mae, 4),
            "val_loss":      round(val_loss, 4),
            "épocas":        epocas,
            "tempo_s":       round(tempo_ms / 1000, 1),
        })

        # Salva o modelo + classes + histórico para validação posterior (predicao/metricas)
        sufixo = f"neuronios_{nome}"
        model.save(caminho_modelos / f"modelo_{sufixo}.keras")
        with open(caminho_modelos / f"classes_{sufixo}.json", "w") as f:
            json.dump(classes, f)
        with open(caminho_modelos / f"historico_{sufixo}.json", "w") as f:
            json.dump(history, f)

        analyze_training(history, caminho_treinamento, f"benchmark_neuronios_{nome}")
        print(f"  accuracy={acc:.4f}  sensibilidade={recall:.4f}  precisao={precisao:.4f}  "
              f"f1={f1:.4f}  L1={l1_mae:.4f}  val_loss={val_loss:.4f}  épocas={epocas}  tempo={tempo_ms/1000:.1f}s")

    df = pd.DataFrame(resultados)

    caminho_csv = caminho_treinamento / "benchmark_neuronios_modelo2.csv"
    df.to_csv(caminho_csv, index=False)

    print("\n" + "=" * 55)
    print("RESULTADOS — Benchmark Modelo 2 (número de neurônios)")
    print("=" * 55)
    print(df.to_string(index=False))
    print(f"\nCSV salvo em: {caminho_csv}")

    _salvar_graficos_comparativos(df, caminho_treinamento)
    print(f"Modelos salvos em: {caminho_modelos}")

    return df

def run_benchmark_camadas(X_train, y_train, X_validate, y_validate, classes, caminho_output):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    from Benchmarks.trn_numero_camadas import (
        create_cnn_2_1024_2_blocos,
        create_cnn_2_1024_3_blocos,
        create_cnn_2_1024_4_blocos,
        create_cnn_2_1024_3_blocos_1_Dense,
        create_cnn_2_1024_3_blocos_3_Dense,
        create_cnn_2_1024_3_blocos_4_Dense,
    )

    num_classes = len(classes)

    # Subpastas: curvas/CSV em treinamento/, modelos salvos em modelos/
    caminho_treinamento = caminho_output / "treinamento"
    caminho_modelos = caminho_output / "modelos"
    caminho_treinamento.mkdir(parents=True, exist_ok=True)
    caminho_modelos.mkdir(parents=True, exist_ok=True)

    variantes = [
        ("2_blocos",          create_cnn_2_1024_2_blocos),
        ("3_blocos",          create_cnn_2_1024_3_blocos),
        ("4_blocos",          create_cnn_2_1024_4_blocos),
        ("3_blocos_1_Dense",  create_cnn_2_1024_3_blocos_1_Dense),
        ("3_blocos_3_Dense",  create_cnn_2_1024_3_blocos_3_Dense),
        ("3_blocos_4_Dense",  create_cnn_2_1024_3_blocos_4_Dense),
    ]

    resultados = []

    for nome, factory in variantes:
        print(f"\n--- Benchmark Camadas | arquitetura = {nome} ---")
        model = factory(input_shape=(64, 64, 1), num_classes=num_classes)

        inicio = time_ns()
        history = training_retorna_historico(model, X_train, y_train, X_validate, y_validate)
        fim = time_ns()
        tempo_ms = (fim - inicio) / 1e6

        val_loss = min(history["val_loss"])
        epocas   = len(history["val_accuracy"])

        # Como EarlyStopping(restore_best_weights=True), o modelo guarda os melhores pesos.
        # Avaliamos as predições no conjunto de validação para extrair as métricas.
        probabilidades = model.predict(X_validate, verbose=0)
        y_previsto = np.argmax(probabilidades, axis=1)
        y_real     = np.argmax(y_validate, axis=1)

        acc       = accuracy_score(y_real, y_previsto)
        recall    = recall_score(y_real, y_previsto, average="macro", zero_division=0)
        precisao  = precision_score(y_real, y_previsto, average="macro", zero_division=0)
        f1        = f1_score(y_real, y_previsto, average="macro", zero_division=0)
        # L1 / MAE: erro absoluto médio entre o vetor de probabilidades e o rótulo one-hot
        l1_mae    = float(np.mean(np.abs(probabilidades - y_validate)))

        resultados.append({
            "camadas":       nome,
            "val_accuracy":  round(acc, 4),
            "sensibilidade": round(recall, 4),
            "precisao":      round(precisao, 4),
            "f1":            round(f1, 4),
            "L1_mae":        round(l1_mae, 4),
            "val_loss":      round(val_loss, 4),
            "épocas":        epocas,
            "tempo_s":       round(tempo_ms / 1000, 1),
        })

        # Salva o modelo + classes + histórico para validação posterior (predicao/metricas)
        sufixo = f"camadas_{nome}"
        model.save(caminho_modelos / f"modelo_{sufixo}.keras")
        with open(caminho_modelos / f"classes_{sufixo}.json", "w") as f:
            json.dump(classes, f)
        with open(caminho_modelos / f"historico_{sufixo}.json", "w") as f:
            json.dump(history, f)

        analyze_training(history, caminho_treinamento, f"benchmark_camadas_{nome}")
        print(f"  accuracy={acc:.4f}  sensibilidade={recall:.4f}  precisao={precisao:.4f}  "
              f"f1={f1:.4f}  L1={l1_mae:.4f}  val_loss={val_loss:.4f}  épocas={epocas}  tempo={tempo_ms/1000:.1f}s")

    df = pd.DataFrame(resultados)

    caminho_csv = caminho_treinamento / "benchmark_camadas_modelo2.csv"
    df.to_csv(caminho_csv, index=False)

    print("\n" + "=" * 55)
    print("RESULTADOS — Benchmark Modelo 2 (número de camadas)")
    print("=" * 55)
    print(df.to_string(index=False))
    print(f"\nCSV salvo em: {caminho_csv}")

    _graficos_por_eixo_camadas(df, caminho_treinamento)
    print(f"Modelos salvos em: {caminho_modelos}")

    return df


def _graficos_por_eixo_camadas(df, caminho_output):
    """Gera dois conjuntos de gráficos separados, um por eixo do estudo, para não
    misturar variações de natureza diferente num mesmo gráfico:
      - blocos convolucionais (mantendo 2 camadas Dense)
      - camadas Dense (mantendo 3 blocos convolucionais)
    O modelo `3_blocos` é o ponto comum aos dois eixos (3 blocos conv + 2 Dense)."""

    # (nome_no_df, rótulo_limpo_no_gráfico) — a ordem da lista define a ordem no eixo X
    grupo_conv = [
        ("2_blocos", "2 blocos"),
        ("3_blocos", "3 blocos"),
        ("4_blocos", "4 blocos"),
    ]
    grupo_dense = [
        ("3_blocos_1_Dense", "1 Dense"),
        ("3_blocos",         "2 Dense"),
        ("3_blocos_3_Dense", "3 Dense"),
        ("3_blocos_4_Dense", "4 Dense"),
    ]

    eixos = [
        (grupo_conv,  "Blocos convolucionais", "benchmark_camadas_blocosconv", "Número de Blocos Convolucionais"),
        (grupo_dense, "Camadas Dense",         "benchmark_camadas_dense",      "Número de Camadas Dense"),
    ]

    for grupo, rotulo_x, prefixo, titulo in eixos:
        ordem = [nome for nome, _ in grupo]
        # Reordena as linhas do df na ordem do eixo e aplica os rótulos limpos
        sub = df.set_index("camadas").loc[ordem].reset_index()
        sub["eixo"] = [rotulo for _, rotulo in grupo]

        _salvar_graficos_comparativos(
            sub, caminho_output,
            coluna_x="eixo",
            rotulo_x=rotulo_x,
            prefixo=prefixo,
            titulo=titulo,
        )

def _salvar_graficos_comparativos(df, caminho_output, coluna_x="neurônios",
                                  rotulo_x="Neurônios por camada Dense",
                                  prefixo="benchmark_neuronios_modelo2",
                                  titulo="Número de Neurônios"):
    xs = df[coluna_x].astype(str).tolist()

    # Gráfico de val_accuracy
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(xs, df["val_accuracy"], color="steelblue")
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("Val Accuracy (máximo)")
    ax.set_title(f"Benchmark Modelo 2 — {titulo}")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, df["val_accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho_output / f"{prefixo}_accuracy.png", dpi=150)
    plt.close(fig)

    # Gráfico de tempo de treino
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(xs, df["tempo_s"], color="coral")
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("Tempo de treino (s)")
    ax.set_title("Benchmark Modelo 2 — Tempo de Treinamento")
    for bar, val in zip(bars, df["tempo_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho_output / f"{prefixo}_tempo.png", dpi=150)
    plt.close(fig)

    # Gráfico de val_loss
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(xs, df["val_loss"], color="mediumseagreen")
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("Val Loss (mínimo)")
    ax.set_title("Benchmark Modelo 2 — Val Loss")
    for bar, val in zip(bars, df["val_loss"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho_output / f"{prefixo}_loss.png", dpi=150)
    plt.close(fig)

    # Gráfico de linhas: evolução das métricas conforme varia o eixo do benchmark
    # (a pergunta do colega — "como se comporta acurácia, sensibilidade, etc")
    fig, ax = plt.subplots(figsize=(8, 5))
    for coluna, rotulo in [
        ("val_accuracy",  "Acurácia"),
        ("sensibilidade", "Sensibilidade (recall)"),
        ("precisao",      "Precisão"),
        ("f1",            "F1-score"),
    ]:
        ax.plot(xs, df[coluna], marker="o", label=rotulo)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("Valor da métrica")
    ax.set_title(f"Benchmark Modelo 2 — Métricas vs {titulo}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho_output / f"{prefixo}_metricas.png", dpi=150)
    plt.close(fig)

    # Gráfico de L1 (MAE) — quanto menor, melhor calibrado
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(xs, df["L1_mae"], color="slateblue")
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("L1 / MAE das probabilidades")
    ax.set_title("Benchmark Modelo 2 — Erro L1 (MAE)")
    for bar, val in zip(bars, df["L1_mae"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho_output / f"{prefixo}_L1.png", dpi=150)
    plt.close(fig)

    print(f"Gráficos comparativos salvos em: {caminho_output}")


def main():

    if len(argv) < 4:
        print("Insira o caminho das imagens para o treino do modelo")
        print(f"Forma de uso: python {argv[0]} <caminho-master-doodle-dataframe> <caminho-pasta-train> <json-words-to-keep> [<caminho-pasta-teste>]")
        print("  - Sem a pasta de teste: validação por split aleatório 0.2 (comportamento antigo)")
        print("  - Com a pasta de teste: treino vem da pasta-train e validação da pasta-teste (held-out real)")
        exit(-1)

    caminho_dataset = Path(argv[1])
    assert caminho_dataset.exists(), "Caminho para o csv incorreto"

    caminho_pasta_imagens = Path(argv[2])
    assert caminho_pasta_imagens.exists(), "Caminho para a pasta de imagens (train) incorreto"

    caminho_json = Path(argv[3])
    assert caminho_json.exists(), "Caminho para o json de palavras selecionadas incorreto"

    caminho_pasta_teste = None
    if len(argv) > 4:
        caminho_pasta_teste = Path(argv[4])
        assert caminho_pasta_teste.exists(), "Caminho para a pasta de teste/validação incorreto"

    with open(caminho_json, "r") as js:
        words_to_keep = json.load(js)

    assert isinstance(words_to_keep, list), "words_to_keep não é uma lista"
    assert len(words_to_keep) > 0, "lista words_to_keep está vazia"

    df = pd.read_csv(caminho_dataset)

    # Mantém apenas categorias que possuem pasta no disco (no train e, se houver, no teste)
    print(f"Categorias solicitadas ({len(words_to_keep)}): {words_to_keep}")
    words_to_keep = [w for w in words_to_keep if (caminho_pasta_imagens / w).exists()]
    if caminho_pasta_teste is not None:
        words_to_keep = [w for w in words_to_keep if (caminho_pasta_teste / w).exists()]
    print(f"Categorias disponíveis no disco ({len(words_to_keep)}): {words_to_keep}")
    assert len(words_to_keep) > 0, "Nenhuma categoria do words_to_keep possui pasta no disco"

    if caminho_pasta_teste is not None:
        # ── Split por pasta: treino e validação vêm de pastas físicas distintas ──
        print(f"Carregando treino de:    {caminho_pasta_imagens}")
        train_imgs, train_labels = carregar_imagens_de_pasta(df, caminho_pasta_imagens, words_to_keep)
        print(f"Carregando validação de: {caminho_pasta_teste}")
        val_imgs, val_labels = carregar_imagens_de_pasta(df, caminho_pasta_teste, words_to_keep)
        print(f"Imagens -> treino: {len(train_labels)} | validação: {len(val_labels)}")

        X_train, X_validate, y_train, y_validate = montar_split_por_pasta(
            train_imgs, train_labels, val_imgs, val_labels, words_to_keep
        )
    else:
        # ── Comportamento antigo: split aleatório 0.2 sobre a pasta-train ──
        print("Aviso: nenhuma pasta de teste informada — usando split aleatório 0.2 sobre a pasta-train.")
        image_array, labels = carregar_imagens_de_pasta(df, caminho_pasta_imagens, words_to_keep)
        X_train, X_validate, y_train, y_validate = encode_labels(labels, image_array, words_to_keep)

    caminho_output = Path("./output")
    caminho_output.mkdir(exist_ok=True)

    # ── Modo benchmark ──────────────────────────────────────────
    if benchmark == "number_of_neurons" and modelo_treinado == 2:
        caminho_bench = Path("Benchmarks/neuronios")
        run_benchmark_neuronios(X_train, y_train, X_validate, y_validate, sorted(words_to_keep), caminho_bench)
        return

    if benchmark == "number_of_layers" and modelo_treinado == 2:
        caminho_bench = Path("Benchmarks/camadas")
        run_benchmark_camadas(X_train, y_train, X_validate, y_validate, sorted(words_to_keep), caminho_bench)
        return

    # ── Modo treino único ────────────────────────────────────────
    if modelo_treinado is not None:
        factories = {
            1: create_cnn_model_1,
            2: create_cnn_model_2,
            3: create_cnn_model_3,
        }
        assert modelo_treinado in factories, f"modelo_treinado deve ser 1, 2 ou 3 (recebeu {modelo_treinado})"

        model = factories[modelo_treinado](input_shape=(64, 64, 1), num_classes=len(words_to_keep))

        inicio = time_ns()
        training(model, X_train, y_train, X_validate, y_validate)
        fim = time_ns()
        print(f"Modelo {modelo_treinado} -> {(fim - inicio) / 1e6:.0f}ms")

        model.save(caminho_output / f"modelo{modelo_treinado}.keras")
        with open(caminho_output / f"classes_modelo{modelo_treinado}.json", "w") as f:
            json.dump(sorted(words_to_keep), f)
        return

    # ── Modo treino todos os modelos ─────────────────────────────
    model1 = create_cnn_model_1(input_shape=(64, 64, 1), num_classes=len(words_to_keep))
    model2 = create_cnn_model_2(input_shape=(64, 64, 1), num_classes=len(words_to_keep))
    model3 = create_cnn_model_3(input_shape=(64, 64, 1), num_classes=len(words_to_keep))

    print("Treinando todos os modelos...")

    inicio = time_ns()
    training(model1, X_train, y_train, X_validate, y_validate)
    t1 = time_ns()
    training(model2, X_train, y_train, X_validate, y_validate)
    t2 = time_ns()
    training(model3, X_train, y_train, X_validate, y_validate)
    t3 = time_ns()

    print(f"Modelo 1 -> {(t1 - inicio) / 1e6:.0f}ms")
    print(f"Modelo 2 -> {(t2 - t1) / 1e6:.0f}ms")
    print(f"Modelo 3 -> {(t3 - t2) / 1e6:.0f}ms")
    print(f"Total    -> {(t3 - inicio) / 1e6:.0f}ms")

    classes_salvas = sorted(words_to_keep)
    for i, model in enumerate([model1, model2, model3], start=1):
        model.save(caminho_output / f"modelo{i}.keras")
        with open(caminho_output / f"classes_modelo{i}.json", "w") as f:
            json.dump(classes_salvas, f)


if __name__ == "__main__":
    main()
