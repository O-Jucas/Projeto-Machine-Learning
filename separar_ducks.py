from pathlib import Path

caminho_desenhos = Path("/home/dbclima/Repositories/dbclima/ufrj/2026_1/Aprendizado_de_Maquina/trabalho_final/archive/doodle/data")
caminho_patos = caminho_desenhos / "duck"
caminho_drums = caminho_desenhos / "drums"

for i, arquivo in enumerate(caminho_patos.iterdir()):
    print(i, arquivo)

    if i % 2 == 0:
        arquivo.move(caminho_drums / arquivo.name)