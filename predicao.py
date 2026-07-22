import json
import numpy as np
import cv2 as cv

from sys import argv, exit
from pathlib import Path
from tensorflow.keras.models import load_model


# python .\predicao.py "Benchmarks/camadas/modelos/modelo_camadas_3_blocos.keras" "C:\Users\lucas\OneDrive\Área de Trabalho\produtividade\ArqComp\arquivos machine learning\testes-criados\drums"

# python .\predicao.py "Benchmarks/neuronios/modelos/modelo_neuronios_256.keras" "C:\Users\lucas\OneDrive\Área de Trabalho\produtividade\ArqComp\arquivos machine learning\testes-criados\Megaphone Re"

# python .\predicao.py "Benchmarks/camadas/modelos/neuronios/modelos/modelo_neuronios_256.keras" "testes-criados/Piano.png"

EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".bmp"}


def descobrir_caminho_classes(caminho_modelo: Path) -> Path:
    """Encontra o JSON de classes salvo ao lado do modelo. O run.py grava
    `modelo1.keras`  -> `classes_modelo1.json` e
    `modelo_neuronios_256.keras` -> `classes_neuronios_256.json`,
    então tentamos as duas convenções de nome e usamos a que existir."""
    stem = caminho_modelo.stem
    candidatos = [caminho_modelo.with_name(f"classes_{stem}.json")]
    if stem.startswith("modelo_"):
        candidatos.append(caminho_modelo.with_name(f"classes_{stem[len('modelo_'):]}.json"))

    for candidato in candidatos:
        if candidato.exists():
            return candidato

    raise FileNotFoundError(
        f"JSON de classes não encontrado ao lado do modelo (procurei: "
        f"{', '.join(c.name for c in candidatos)}) — retreine para gerá-lo"
    )


def preprocessar_imagem(caminho_imagem: Path, target_size=(64, 64)) -> np.ndarray:
    raw = np.fromfile(str(caminho_imagem), dtype=np.uint8)
    img = cv.imdecode(raw, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho_imagem}")
    img = cv.resize(img, target_size)
    img = img / 255.0
    return img[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)


def coletar_imagens(caminho_teste: Path):
    """Retorna uma lista de (caminho_imagem, rotulo_verdadeiro|None).

    - Arquivo de imagem  -> uma única entrada sem rótulo verdadeiro.
    - Pasta com subpastas -> cada subpasta é tratada como uma classe e o nome
      dela vira o rótulo verdadeiro (estrutura `teste/<palavra>/*.png`).
    - Pasta plana de imagens -> entradas sem rótulo verdadeiro.
    """
    if caminho_teste.is_file():
        return [(caminho_teste, None)]

    subpastas = sorted(p for p in caminho_teste.iterdir() if p.is_dir())
    if subpastas:
        itens = []
        for pasta in subpastas:
            for img in sorted(pasta.iterdir()):
                if img.suffix.lower() in EXTENSOES_IMAGEM:
                    itens.append((img, pasta.name))
        return itens

    return [
        (img, None)
        for img in sorted(caminho_teste.iterdir())
        if img.suffix.lower() in EXTENSOES_IMAGEM
    ]


def prever(model, classes, caminho_imagem: Path, target_size):
    X = preprocessar_imagem(caminho_imagem, target_size)
    probabilidades = model.predict(X, verbose=0)[0]
    indice = int(np.argmax(probabilidades))
    return classes[indice], float(probabilidades[indice] * 100), probabilidades


def main():
    if len(argv) < 3:
        print("Uso: python predicao.py <caminho-do-modelo> <caminho-do-teste>")
        print("  <caminho-do-teste> pode ser uma imagem única ou uma pasta.")
        print("  Se a pasta tiver subpastas por classe (ex: teste/<palavra>/),")
        print("  a acurácia é calculada comparando com o rótulo verdadeiro.")
        print("Exemplo: python predicao.py ./output/modelo1.keras ./data/teste")
        exit(-1)

    caminho_modelo = Path(argv[1])
    assert caminho_modelo.exists(), f"Modelo não encontrado: {caminho_modelo}"

    caminho_teste = Path(argv[2])
    assert caminho_teste.exists(), f"Caminho de teste não encontrado: {caminho_teste}"

    caminho_classes = descobrir_caminho_classes(caminho_modelo)
    with open(caminho_classes, "r") as f:
        classes = json.load(f)

    print(f"Carregando modelo: {caminho_modelo}")
    model = load_model(caminho_modelo)

    # Usa o tamanho de entrada do próprio modelo (treinado em 64x64).
    _, altura, largura, _ = model.input_shape
    target_size = (largura, altura)

    itens = coletar_imagens(caminho_teste)
    assert len(itens) > 0, f"Nenhuma imagem encontrada em: {caminho_teste}"

    # Caso simples: uma única imagem -> saída detalhada com top-3.
    if len(itens) == 1 and itens[0][1] is None:
        caminho_imagem, _ = itens[0]
        print(f"Processando imagem: {caminho_imagem.name}")
        classe_prevista, confianca, probabilidades = prever(model, classes, caminho_imagem, target_size)

        print(f"\nPrevisão: {classe_prevista} ({confianca:.1f}% de confiança)")
        print("\nTop 3:")
        for i in np.argsort(probabilidades)[::-1][:3]:
            print(f"  {classes[i]:<20} {probabilidades[i] * 100:.1f}%")
        return

    # Caso lote: prediz cada imagem e, se houver rótulo verdadeiro, mede acurácia.
    print(f"Avaliando {len(itens)} imagem(ns)...\n")
    acertos = 0
    total_rotulado = 0

    for caminho_imagem, rotulo_verdadeiro in itens:
        classe_prevista, confianca, _ = prever(model, classes, caminho_imagem, target_size)

        if rotulo_verdadeiro is None:
            print(f"  {caminho_imagem.name:<30} -> {classe_prevista} ({confianca:.1f}%)")
        else:
            certo = classe_prevista == rotulo_verdadeiro
            total_rotulado += 1
            acertos += int(certo)
            marca = "OK " if certo else "ERRO"
            print(f"  [{marca}] {caminho_imagem.name:<30} verdadeiro={rotulo_verdadeiro:<15} "
                  f"previsto={classe_prevista} ({confianca:.1f}%)")

    if total_rotulado > 0:
        print(f"\nAcurácia: {acertos}/{total_rotulado} = {acertos / total_rotulado * 100:.1f}%")


if __name__ == "__main__":
    main()
