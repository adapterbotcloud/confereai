import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
import joblib
from utils import criar_sequencias

def detectar_anomalias_autoencoder_por_mes(df, autoencoder, limiar_percentil=96):
    anomalias_por_mes = []

    #219

    # Filtrar os dados para cada combinação de ano, mês e cod_cargo
    anos_meses_cargos = df[['num_ano', 'num_mes', 'cod_cargo']].drop_duplicates()

    for _, row in anos_meses_cargos.iterrows():
        ano = row['num_ano']
        mes = row['num_mes']
        cod_cargo = row['cod_cargo']

        # Filtrar o dataframe para o ano, mês e cargo específicos
        df_mes = df[(df['num_ano'] == ano) & (df['num_mes'] == mes) & (df['cod_cargo'] == cod_cargo)]
        dados_rubricas = df_mes[df_mes.columns[4:]].values  # Seleciona as rubricas (ajuste a coluna de acordo com sua estrutura)

        # Reconstrução e erro do autoencoder
        reconstrucoes = autoencoder.predict(dados_rubricas)
        erros = np.mean(np.power(dados_rubricas - reconstrucoes, 2), axis=1)

        # Limiar baseado no percentil para o mês e cargo atuais
        limiar = np.percentile(erros, limiar_percentil)
        anomalias = erros > limiar

        # Armazenar resultados do mês e cargo
        df_mes['Erro_Reconstrucao'] = erros
        df_mes['Anomalia_Autoencoder'] = anomalias
        anomalias_por_mes.append(df_mes)

    # Concatenar todos os meses e cargos em um único DataFrame
    return pd.concat(anomalias_por_mes, ignore_index=True)

if __name__ == "__main__":
    # Carregar dados normalizados
    df_normalizado = pd.read_csv(os.path.join('data', 'pagamentos_normalizados.csv'))

    # Selecionar as rubricas
    #colunas_rubricas = df_normalizado.columns[4:]
    #dados_rubricas = df_normalizado[colunas_rubricas].values

    # Carregar o modelo Autoencoder treinado
    autoencoder = load_model(os.path.join('models', 'autoencoder_pagamentos_p110.keras'))

    # Detectar anomalias por mês usando o Autoencoder
    df_anomalias = detectar_anomalias_autoencoder_por_mes(df_normalizado, autoencoder)

    # Salvar os resultados em um arquivo CSV
    df_anomalias.to_csv(os.path.join('data', 'anomalias_detectadas.csv'), index=False)

    # Total de anomalias detectadas pelo Autoencoder
    total_anomalias_autoencoder = df_anomalias['Anomalia_Autoencoder'].sum()
    print(f"Total de anomalias detectadas pelo Autoencoder: {total_anomalias_autoencoder}")
