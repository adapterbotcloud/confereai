import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Carregar o arquivo CSV
caminho_csv = os.path.join('data', 'historico_pagamento_2020_2024_15.csv')

df = pd.read_csv(caminho_csv, delimiter=';')

# Inicializar listas para guardar os dados de treino e teste
df_treino_list = []
df_teste_list = []


# Iterar sobre cada ano de 2000 a 2024
for ano in range(2000, 2025):
    # Filtrar os dados para o ano atual
    df_ano = df[df['num_ano'] == ano]
    
    if len(df_ano) > 0:
        # Dividir os dados do ano atual em 80% treino e 20% teste
        df_ano_treino, df_ano_teste = train_test_split(df_ano, test_size=0.2, random_state=42)
        
        # Adicionar os dados divididos às listas correspondentes
        df_treino_list.append(df_ano_treino)
        df_teste_list.append(df_ano_teste)
    else:
        print(f"Sem dados para o ano {ano}")

# Concatenar os dados de todos os anos em dois DataFrames, um para treino e outro para teste
df_treino = pd.concat(df_treino_list)
df_teste = pd.concat(df_teste_list)

# Salvar os arquivos de treino e teste
df_treino.to_csv('data/treino/treino.csv', index=False, sep=';')
df_teste.to_csv('data/teste/teste.csv', index=False, sep=';')

print("Divisão concluída. Arquivos 'treino.csv' e 'teste.csv' salvos.")
