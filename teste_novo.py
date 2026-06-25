import matplotlib
matplotlib.use("Agg")

from sys import argv, exit
from pathlib import Path
from avaliar_mundo import executar_avaliacao_global

def main():
    if len(argv) < 5:
        print("Uso: python testar.py <csv> <pasta_img> <json_palavras> <SIGLA_PAIS>")
        print("Exemplo: python testar.py doodle_dataframe.csv doodle/ words_to_keep.json HU")
        exit(-1)

    caminho_dataset = Path(argv[1])
    caminho_pasta_imagens = Path(argv[2])
    sigla_pais = argv[4].upper()
    
    print(f"\n==================================================")
    print(f"MODO DE AVALIAÇÃO RÁPIDA - MODELO: [{sigla_pais}]")
    print(f"==================================================")

    # Verifica se o modelo já existe antes de tentar testar
    if not (Path("./output") / f"modelo_{sigla_pais}.keras").exists():
        print(f"[ERRO] O modelo modelo_{sigla_pais}.keras não existe na pasta output.")
        print(f"Você precisa rodar o run.py com o país {sigla_pais} primeiro para treinar.")
        return

    # Pula direto para a Fase 4 (O Choque Cultural)
    executar_avaliacao_global(caminho_dataset, caminho_pasta_imagens, sigla_pais)
    print("\n[SUCESSO] Nova matriz de avaliação gerada na pasta output!")

if __name__ == "__main__":
    main()