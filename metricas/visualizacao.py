import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plotar_matriz_top10(y_verdadeiro, y_previsto, nomes_das_classes, titulo, caminho_salvar):
    """Gera o mapa de calor focado nas 10 maiores confusões."""
    matriz_completa = confusion_matrix(y_verdadeiro, y_previsto)
    np.fill_diagonal(matriz_completa, 0) # Zera acertos para destacar erros
    
    erros_totais = matriz_completa.sum(axis=1) + matriz_completa.sum(axis=0)
    top_n = min(10, len(nomes_das_classes))
    indices_problematicos = erros_totais.argsort()[-top_n:][::-1]
    
    matriz_reduzida = matriz_completa[np.ix_(indices_problematicos, indices_problematicos)]
    nomes_reduzidos = [nomes_das_classes[i] for i in indices_problematicos]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_reduzida, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=nomes_reduzidos, yticklabels=nomes_reduzidos)
    
    plt.title(titulo, fontsize=15, pad=20)
    plt.ylabel('Real (O que foi desenhado)')
    plt.xlabel('Previsto (Aposta da CNN)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
    plt.close()