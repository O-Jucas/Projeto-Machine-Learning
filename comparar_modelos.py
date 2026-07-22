import os

# Silencia os logs verbosos do TensorFlow/oneDNN antes de importá-lo.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sys import argv, exit
from pathlib import Path
from tensorflow.keras.models import load_model

# Reaproveita o pré-processamento e a descoberta do JSON de classes do predicao.py
# para manter uma única fonte de verdade sobre como a imagem é lida e como os
# rótulos são localizados ao lado de cada modelo.
from predicao import preprocessar_imagem, descobrir_caminho_classes


# Uso:
#   python comparar_modelos.py <caminho-da-imagem> [classe-verdadeira]
#
# Roda a imagem em cada grupo de modelos de benchmark e gera 3 gráficos
# comparativos (neurônios, blocos convolucionais e camadas Dense), mostrando a
# confiança de cada modelo. Se a classe verdadeira não for passada, ela é
# inferida do nome do arquivo (ex.: "Drums.png" -> "drums").

RAIZ = Path(__file__).resolve().parent
MODELOS_NEURONIOS = RAIZ / "Benchmarks" / "neuronios" / "modelos"
MODELOS_CAMADAS = RAIZ / "Benchmarks" / "camadas" / "modelos"

# Cada grupo segue exatamente a convenção de nomes/rótulos do run.py
# (_graficos_por_eixo_camadas), para os gráficos baterem com o resto do projeto.
GRUPOS = [
    {
        "chave": "neuronios",
        "titulo": "Neurônios por camada Dense",
        "rotulo_x": "Neurônios por camada Dense",
        "modelos": [
            ("256",  MODELOS_NEURONIOS / "modelo_neuronios_256.keras"),
            ("512",  MODELOS_NEURONIOS / "modelo_neuronios_512.keras"),
            ("1024", MODELOS_NEURONIOS / "modelo_neuronios_1024.keras"),
            ("2048", MODELOS_NEURONIOS / "modelo_neuronios_2048.keras"),
        ],
    },
    {
        "chave": "blocos",
        "titulo": "Blocos convolucionais",
        "rotulo_x": "Número de Blocos Convolucionais",
        "modelos": [
            ("2 blocos", MODELOS_CAMADAS / "modelo_camadas_2_blocos.keras"),
            ("3 blocos", MODELOS_CAMADAS / "modelo_camadas_3_blocos.keras"),
            ("4 blocos", MODELOS_CAMADAS / "modelo_camadas_4_blocos.keras"),
        ],
    },
    {
        "chave": "dense",
        "titulo": "Camadas Dense",
        "rotulo_x": "Número de Camadas Dense",
        "modelos": [
            ("1 Dense", MODELOS_CAMADAS / "modelo_camadas_3_blocos_1_Dense.keras"),
            # O modelo `3_blocos` é o ponto comum: 3 blocos conv + 2 camadas Dense.
            ("2 Dense", MODELOS_CAMADAS / "modelo_camadas_3_blocos.keras"),
            ("3 Dense", MODELOS_CAMADAS / "modelo_camadas_3_blocos_3_Dense.keras"),
            ("4 Dense", MODELOS_CAMADAS / "modelo_camadas_3_blocos_4_Dense.keras"),
        ],
    },
]

VERDE = "#2e9e5b"      # acertou (previsão == classe verdadeira)
VERMELHO = "#c0392b"   # errou
AZUL = "steelblue"     # classe verdadeira desconhecida (sem como julgar acerto)


def normalizar(texto: str) -> str:
    """Normaliza um rótulo para comparação: minúsculas, sem acentos de
    separação, '_' e '-' viram espaço. Ex.: 'Hot_Dog' -> 'hot dog'."""
    return texto.strip().lower().replace("_", " ").replace("-", " ")


def inferir_classe_verdadeira(caminho_imagem: Path, classes) -> str | None:
    """Tenta casar o nome do arquivo com uma das classes conhecidas.
    Retorna None se não houver correspondência (imagem 'anônima')."""
    alvo = normalizar(caminho_imagem.stem)
    mapa = {normalizar(c): c for c in classes}
    if alvo in mapa:
        return mapa[alvo]
    # tenta sem dígitos finais, ex.: "drums2" -> "drums"
    alvo_sem_num = alvo.rstrip("0123456789 ").strip()
    return mapa.get(alvo_sem_num)


def avaliar_grupo(grupo, caminho_imagem: Path, classe_verdadeira_arg):
    """Roda a imagem em cada modelo do grupo e devolve uma lista de dicts com
    o resultado de cada um (rótulo, previsão, confiança e confiança na classe
    verdadeira)."""
    resultados = []
    for rotulo, caminho_modelo in grupo["modelos"]:
        if not caminho_modelo.exists():
            print(f"  [AVISO] modelo não encontrado, pulando: {caminho_modelo.name}")
            continue

        classes = json.load(open(descobrir_caminho_classes(caminho_modelo)))
        model = load_model(caminho_modelo)

        _, altura, largura, _ = model.input_shape
        X = preprocessar_imagem(caminho_imagem, target_size=(largura, altura))
        probs = model.predict(X, verbose=0)[0]

        idx_prev = int(np.argmax(probs))
        classe_prevista = classes[idx_prev]
        conf_prevista = float(probs[idx_prev] * 100)

        # Confiança na classe verdadeira (se ela for conhecida e existir no modelo).
        classe_verdadeira = classe_verdadeira_arg or inferir_classe_verdadeira(caminho_imagem, classes)
        conf_verdadeira = None
        acertou = None
        if classe_verdadeira and classe_verdadeira in classes:
            conf_verdadeira = float(probs[classes.index(classe_verdadeira)] * 100)
            acertou = (classe_prevista == classe_verdadeira)

        resultados.append({
            "rotulo": rotulo,
            "classe_prevista": classe_prevista,
            "conf_prevista": conf_prevista,
            "classe_verdadeira": classe_verdadeira,
            "conf_verdadeira": conf_verdadeira,
            "acertou": acertou,
        })

        marca = "" if acertou is None else ("  OK" if acertou else "  ERRO")
        print(f"  {rotulo:<10} -> {classe_prevista:<12} ({conf_prevista:5.1f}%){marca}")
    return resultados


def gerar_grafico(grupo, resultados, caminho_imagem: Path, caminho_saida: Path):
    """Gera um gráfico de barras para o grupo: imagem de entrada à esquerda e
    barras de confiança à direita. A barra mostra a confiança na classe
    verdadeira quando ela é conhecida (verde = acertou, vermelho = errou);
    caso contrário, mostra a confiança da própria previsão (azul)."""
    if not resultados:
        return None

    tem_verdade = all(r["conf_verdadeira"] is not None for r in resultados)
    classe_verdadeira = next((r["classe_verdadeira"] for r in resultados if r["classe_verdadeira"]), None)

    xs = [r["rotulo"] for r in resultados]
    if tem_verdade:
        alturas = [r["conf_verdadeira"] for r in resultados]
        cores = [VERDE if r["acertou"] else VERMELHO for r in resultados]
        rotulo_y = f"Confiança na classe correta '{classe_verdadeira}' (%)"
    else:
        alturas = [r["conf_prevista"] for r in resultados]
        cores = [AZUL] * len(resultados)
        rotulo_y = "Confiança da previsão (%)"

    fig, (ax_img, ax) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 2.6]}
    )

    # --- Imagem de entrada (como o modelo a enxerga: 64x64 em tons de cinza) ---
    X = preprocessar_imagem(caminho_imagem, target_size=(64, 64))
    ax_img.imshow(X[0, :, :, 0], cmap="gray")
    ax_img.set_title("Imagem de entrada", fontsize=10)
    ax_img.axis("off")

    # --- Barras comparativas ---
    barras = ax.bar(xs, alturas, color=cores, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(grupo["rotulo_x"])
    ax.set_ylabel(rotulo_y)
    ax.set_title(f"Comparação — {grupo['titulo']}")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    for barra, r in zip(barras, resultados):
        topo = barra.get_height()
        # Valor da barra em destaque
        ax.text(barra.get_x() + barra.get_width() / 2, topo + 1.5,
                f"{topo:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        # Quando errou, mostra o que o modelo previu no lugar
        if r["acertou"] is False:
            ax.text(barra.get_x() + barra.get_width() / 2, topo / 2,
                    f"previu:\n{r['classe_prevista']}\n({r['conf_prevista']:.0f}%)",
                    ha="center", va="center", fontsize=7.5, color="white")

    # Legenda só quando a cor carrega significado (acerto/erro)
    if tem_verdade:
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=VERDE, edgecolor="black", label="Classificou certo"),
            Patch(facecolor=VERMELHO, edgecolor="black", label="Classificou errado"),
        ], fontsize=8, loc="upper right")

    fig.suptitle(f"Imagem: {caminho_imagem.name}", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    destino = caminho_saida / f"comparacao_{grupo['chave']}_{caminho_imagem.stem}.png"
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return destino


def main():
    if len(argv) < 2:
        print("Uso: python comparar_modelos.py <caminho-da-imagem> [classe-verdadeira]")
        print("  Roda a imagem em todos os modelos de benchmark e gera 3 gráficos")
        print("  comparativos (neurônios, blocos e Dense).")
        print("  A classe verdadeira é opcional; se omitida, é inferida do nome do arquivo.")
        exit(-1)

    caminho_imagem = Path(argv[1])
    assert caminho_imagem.exists(), f"Imagem não encontrada: {caminho_imagem}"
    assert caminho_imagem.is_file(), f"O caminho não é um arquivo de imagem: {caminho_imagem}"

    classe_verdadeira_arg = normalizar(argv[2]) if len(argv) > 2 else None
    if classe_verdadeira_arg:
        # mantém o casing original digitado para exibição, mas compara normalizado
        classe_verdadeira_arg = argv[2].strip()

    caminho_saida = RAIZ / "comparacoes"
    caminho_saida.mkdir(exist_ok=True)

    print(f"Imagem: {caminho_imagem}")
    print("=" * 60)

    gerados = []
    for grupo in GRUPOS:
        print(f"\n--- {grupo['titulo']} ---")
        resultados = avaliar_grupo(grupo, caminho_imagem, classe_verdadeira_arg)
        destino = gerar_grafico(grupo, resultados, caminho_imagem, caminho_saida)
        if destino:
            gerados.append(destino)

    print("\n" + "=" * 60)
    print("Gráficos gerados:")
    for g in gerados:
        print(f"  {g}")


if __name__ == "__main__":
    main()
