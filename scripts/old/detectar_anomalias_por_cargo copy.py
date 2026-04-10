# scripts/detectar_anomalias_por_cargo.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Model
import os
import joblib
from utils import criar_sequencias_por_vinculo

def detectar_anomalias_autoencoder2(dados, autoencoder, limiar_percentil=96):
    reconstrucoes = autoencoder.predict(dados)
    erros = np.mean(np.power(dados - reconstrucoes, 2), axis=1)
    limiar = np.percentile(erros, limiar_percentil)
    anomalias = erros > limiar
    return anomalias, erros, limiar

def detectar_anomalias_autoencoder(dados, autoencoder, limiar_percentil=96):
    # Fazer a reconstrução dos dados usando o autoencoder
    reconstrucoes = autoencoder.predict(dados)
    
    # Calcular o erro de reconstrução por linha
    erros_por_linha = np.mean(np.power(dados - reconstrucoes, 2), axis=1)
    
    # Definir o limiar de anomalia para o erro de reconstrução por linha
    limiar_erro_total = np.percentile(erros_por_linha, limiar_percentil)
    
    # Identificar quais linhas têm erro total maior que o limiar
    anomalias_por_linha = erros_por_linha > limiar_erro_total
    
    # Calcular o erro de reconstrução por rubrica (colunas) dentro de cada linha
    erros_por_rubrica = np.abs(dados - reconstrucoes)
    
    # Definir o limiar de erro por rubrica com base no percentil
    limiar_por_rubrica = np.percentile(erros_por_rubrica, limiar_percentil, axis=0)
    
    # Identificar rubricas anômalas (True se o erro da rubrica for maior que o limiar)
    anomalias_por_rubrica = erros_por_rubrica > limiar_por_rubrica
    
    return anomalias_por_linha, erros_por_linha, erros_por_rubrica, anomalias_por_rubrica

    # Retornar as linhas anômalas, os erros por linha, o limiar de erro total, e as rubricas responsáveis
    return anomalias_por_linha, erros_por_linha, limiar_erro_total, rubricas_responsaveis    

def detectar_anomalias_lstm(sequencias, modelo_lstm, limiar_percentil=96):
    predicoes = modelo_lstm.predict(sequencias)
    erros = np.mean(np.power(sequencias - predicoes, 2), axis=(1, 2))
    limiar = np.percentile(erros, limiar_percentil)
    anomalias = erros > limiar
    return anomalias, erros, limiar 

if __name__ == "__main__":
    # Carregar os dados normalizados
    df_normalizado = pd.read_csv(os.path.join('data', 'pagamentos_normalizados.csv'))
    
    # Supondo que 'cod_cargo' seja a coluna com os cargos dos servidores
    cargos_unicos = df_normalizado['cod_cargo'].unique()

    # Usar apenas as rubricas sem aplicar normalização adicional
    colunas_rubricas = df_normalizado.columns[4:]  # Selecionando as rubricas a partir da 4ª coluna

    # Definir o número de timesteps para o LSTM
    timesteps = 12

    # Lista para armazenar os resultados de anomalias
    todas_anomalias = []

    for cargo in cargos_unicos:
        print(f"Detectando anomalias para o cargo: {cargo}")

        # Filtrar os dados pelo cargo atual
        df_cargo = df_normalizado[df_normalizado['cod_cargo'] == cargo]
        dados_rubricas = df_cargo[colunas_rubricas].values

        # Carregar o modelo Autoencoder e LSTM treinados para o cargo
        modelo_autoencoder_path = os.path.join('models', f'autoencoder_pagamentos_{cargo}.keras')
        #modelo_lstm_path = os.path.join('models', f'lstm_pagamentos_{cargo}.keras')

        #if not os.path.exists(modelo_autoencoder_path) or not os.path.exists(modelo_lstm_path):
        #    print(f"Modelos para o cargo {cargo} não encontrados. Pulando...")
        #    continue

        autoencoder = load_model(modelo_autoencoder_path)
        #modelo_lstm = load_model(modelo_lstm_path)

        # Detectar anomalias usando o Autoencoder
        anomalias_autoencoder, erros_autoencoder, limiar_autoencoder = detectar_anomalias_autoencoder(dados_rubricas, autoencoder)

        # Criar sequências temporais para o LSTM
        #sequencias = criar_sequencias_por_vinculo(df_cargo,colunas_rubricas, timesteps)
        
        # Detectar anomalias usando o LSTM
        #anomalias_lstm, erros_lstm, limiar_lstm = detectar_anomalias_lstm(sequencias, modelo_lstm)

        # Ajustar o DataFrame para registrar os resultados
        df_cargo = df_cargo.copy()
        df_cargo.loc[:, 'Erro_Reconstrucao'] = erros_autoencoder
        df_cargo.loc[:, 'Anomalia_Autoencoder'] = anomalias_autoencoder

        # Para o LSTM, apenas as amostras válidas (após as primeiras `timesteps`)
        #indices_validos = np.arange(timesteps, len(df_cargo))
        #df_cargo = df_cargo.iloc[indices_validos]
        #df_cargo.loc[:, 'Erro_LSTM'] = erros_lstm
        #df_cargo.loc[:, 'Anomalia_LSTM'] = anomalias_lstm

        # Adicionar os resultados do cargo atual na lista total
        todas_anomalias.append(df_cargo)

    # Concatenar todas as anomalias detectadas para todos os cargos
    df_anomalias_total = pd.concat(todas_anomalias, ignore_index=True)

    # Salvar os resultados em um arquivo CSV
    df_anomalias_total.to_csv(os.path.join('data', 'anomalias_detectadas_por_cargo.csv'), index=False)

    # Calcular e imprimir o total de anomalias detectadas
    total_anomalias_autoencoder = np.sum(df_anomalias_total['Anomalia_Autoencoder'])
    # total_anomalias_lstm = np.sum(df_anomalias_total['Anomalia_LSTM'])
    print(f"Total de anomalias detectadas pelo Autoencoder: {total_anomalias_autoencoder}")
    #print(f"Total de anomalias detectadas pela LSTM: {total_anomalias_lstm}")
