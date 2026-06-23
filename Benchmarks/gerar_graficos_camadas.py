"""Regenera os gráficos comparativos do benchmark de camadas a partir do CSV já salvo,
separando os dois eixos (blocos convolucionais e camadas Dense) — sem precisar retreinar.

Uso (a partir da raiz do projeto):
    python Benchmarks/gerar_graficos_camadas.py [caminho-do-csv]
"""
import sys
from pathlib import Path

import pandas as pd

# Permite importar run.py mesmo rodando de dentro de Benchmarks/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run import _graficos_por_eixo_camadas


def main():
    if len(sys.argv) > 1:
        caminho_csv = Path(sys.argv[1])
    else:
        caminho_csv = Path("Benchmarks/camadas/treinamento/benchmark_camadas_modelo2.csv")

    assert caminho_csv.exists(), f"CSV não encontrado: {caminho_csv}"

    df = pd.read_csv(caminho_csv)
    _graficos_por_eixo_camadas(df, caminho_csv.parent)
    print("Gráficos por eixo gerados a partir do CSV existente.")


if __name__ == "__main__":
    main()
