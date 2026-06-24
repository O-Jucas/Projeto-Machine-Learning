import matplotlib
matplotlib.use("Agg")

from sys import argv, exit
from pathlib import Path

# Importando as funções principais dos nossos módulos
from treinar_vies_cultural import executar_treinamento
from avaliar_mundo import executar_avaliacao_global

# Configuração Central
ARQUITETURA_ESCOLHIDA = 1

def main():
    if len(argv) < 5:
        print("Uso: python run.py <csv> <pasta_img> <json_palavras> <SIGLA_PAIS>")
        print("Exemplo: python run.py master.csv doodle/ words.json BR")
        exit(-1)

    caminho_dataset = Path(argv[1])
    caminho_pasta_imagens = Path(argv[2])
    caminho_json = Path(argv[3])
    sigla_pais = argv[4].upper()
    
    print(f"\n{'='*50}")
    print(f"INICIANDO PIPELINE DE VIÉS CULTURAL")
    print(f"{'='*50}")

    # 1. Gira a chave do Treinamento
    executar_treinamento(
        caminho_dataset, 
        caminho_pasta_imagens, 
        caminho_json, 
        sigla_pais, 
        ARQUITETURA_ESCOLHIDA
    )

    # 2. Gira a chave do Teste e Avaliação
    executar_avaliacao_global(
        caminho_dataset, 
        caminho_pasta_imagens, 
        sigla_pais
    )

    print(f"\n[SUCESSO] Pipeline sequencial concluído com sucesso. Artefatos em ./output")

if __name__ == "__main__":
    main()