import shutil
from typing import List
from pathlib import Path
import sys


def criar_pasta_vazia(caminho: Path, nome_pasta: str) -> Path:
    pasta_destino = caminho / nome_pasta
    if pasta_destino.exists():
        shutil.rmtree(pasta_destino)

    pasta_destino.mkdir()
    return pasta_destino


def separar_treino_teste(
        caminho_origem: Path,
        caminho_destino: Path,
        test_ratio=0.2,
        lista_targets: List = []
) -> None:
    assert caminho_origem.exists(), "Caminho fornecido nao existe"
    assert len(lista_targets) > 0, "Forneca uma lista de targets"

    caminho_train = criar_pasta_vazia(caminho_destino, "train")
    caminho_test = criar_pasta_vazia(caminho_destino, "test")

    for pasta_desenho in caminho_origem.iterdir():
        if pasta_desenho.name not in lista_targets:
            continue

        caminho_train_desenho = criar_pasta_vazia(caminho_train, pasta_desenho.name)
        caminho_test_desenho = criar_pasta_vazia(caminho_test, pasta_desenho.name)

        lista_desenhos = list(enumerate(pasta_desenho.iterdir()))

        test_ratio_int = int(test_ratio * 100)
        for idx, desenho in lista_desenhos:
            if idx % 100 < test_ratio_int:
                shutil.copy(desenho, caminho_test_desenho)

            else:
                shutil.copy(desenho, caminho_train_desenho)


def main():
    import json

    with open("./words_to_keep.json", "rb") as js:
        lista_palavras = json.load(js)
        print(lista_palavras)

    caminho_desenhos = Path("/home/dbclima/Repositories/dbclima/ufrj/2026_1/Aprendizado_de_Maquina/trabalho_final/archive/doodle/data")
    separar_treino_teste(
        caminho_desenhos,
        Path("./data/"),
        0.2,
        lista_palavras
    )

if __name__ == "__main__":
    main()
