# scripts/utils.py
import numpy as np

# Função para criar sequências temporais (com 12 meses)
def criar_sequencias_por_vinculo(df, colunas_rubricas, timesteps=12):
    sequencias = []
    vinculos = df['isn_vinculo'].unique()

    for vinculo in vinculos:
        # Filtrar os dados para o vínculo atual
        dados_vinculo = df[df['isn_vinculo'] == vinculo][colunas_rubricas].values
        
        # Verificar se o número de meses é menor que o timesteps
        if len(dados_vinculo) < timesteps:
            # Aplicar padding nas linhas (meses) para completar 12 meses
            dados_vinculo_pad = np.pad(dados_vinculo, ((0, timesteps - len(dados_vinculo)), (0, 0)), mode='constant', constant_values=0)
        else:
            dados_vinculo_pad = dados_vinculo
         
        # Criar sequências de tamanho `timesteps` a partir dos dados do vínculo
        for i in range(len(dados_vinculo_pad) - timesteps + 1):
            sequencias.append(dados_vinculo_pad[i:i+timesteps])
    
    return np.array(sequencias)