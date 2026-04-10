# scripts/preprocessamento.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def carregar_dados(caminho_csv):
    todas_rubricas = [
        8, 10, 11, 13, 14, 15, 16, 18, 19, 21, 24, 30, 39, 41, 46, 51, 54, 62, 68, 
        72, 78, 79, 80, 86, 91, 94, 97, 100, 101, 106, 140, 151, 152, 158, 160, 161, 
        165, 168, 169, 171, 185, 188, 199, 200, 204, 207, 209, 244, 248, 251, 252, 
        253, 255, 256, 272, 274, 276, 293, 294, 302, 303, 312, 316, 317, 327, 329, 
        346, 365, 371, 380, 666, 669, 680, 1000, 1185, 1232, 1233, 1235, 1236, 1249, 
        1261, 1271, 1277, 1352, 1353, 1367, 1368, 1369, 1404, 1407, 1454, 1455, 1478, 
        1489, 1514, 1553]

    # Carrega o CSV
    df = pd.read_csv(caminho_csv, delimiter=';')
    
    # Cria uma tabela pivotando as rubricas como colunas, agora incluindo cod_cargo
    df_pivot = df.pivot_table(index=['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'],
                              columns='isn_rubrica', values='vlr_calculado', fill_value=0)
    
    # Resetando o índice para que as colunas 'isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes' voltem a ser colunas normais
    df_pivot.reset_index(inplace=True)
    
    # Garantir que todas as rubricas estejam presentes como colunas
    df_final = df_pivot.reindex(columns=['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'] + todas_rubricas, fill_value=0)

    # Ordenando os dados por isn_vinculo, num_ano e num_mes
    df_final = df_final.sort_values(by=['isn_vinculo', 'num_ano', 'num_mes'])

    return df_final


def normalizar_dados(df, colunas_rubricas):
    scaler = MinMaxScaler()
    df[colunas_rubricas] = scaler.fit_transform(df[colunas_rubricas])
    return df, scaler

if __name__ == "__main__":
    caminho_csv = os.path.join('data', 'historico_pagamento_2020_2024_15.csv')
    df_pivot = carregar_dados(caminho_csv)
    colunas_rubricas = df_pivot.columns[4:]
    df_normalizado, scaler = normalizar_dados(df_pivot, colunas_rubricas)
    df_normalizado.to_csv(os.path.join('data', 'pagamentos_normalizados.csv'), index=False)
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
