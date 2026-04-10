# scripts/detectar_anomalias.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
import joblib
from utils import criar_sequencias

def detectar_anomalias_autoencoder(dados, autoencoder, limiar_percentil=96):
    reconstrucoes = autoencoder.predict(dados)
    erros = np.mean(np.power(dados - reconstrucoes, 2), axis=1)
    limiar = np.percentile(erros, limiar_percentil)
    anomalias = erros > limiar
    return anomalias, erros, limiar

def detectar_anomalias_lstm(sequencias, modelo_lstm, limiar_percentil=96):
    predicoes = modelo_lstm.predict(sequencias)
    erros = np.mean(np.power(sequencias - predicoes, 2), axis=(1, 2))
    limiar = np.percentile(erros, limiar_percentil)
    anomalias = erros > limiar
    return anomalias, erros, limiar

if __name__ == "__main__":
    df_normalizado = pd.read_csv(os.path.join('data', 'pagamentos_normalizados.csv'))
    colunas_rubricas = df_normalizado.columns[3:]
    dados_rubricas = df_normalizado[colunas_rubricas].values
    autoencoder = load_model(os.path.join('models', 'autoencoder_pagamentos_p110.keras'))
    modelo_lstm = load_model(os.path.join('models', 'lstm_pagamentos_p110.keras'))
    anomalias_autoencoder, erros_autoencoder, limiar_autoencoder = detectar_anomalias_autoencoder(dados_rubricas, autoencoder)
    timesteps = 12
    sequencias = criar_sequencias(dados_rubricas, timesteps)
    anomalias_lstm, erros_lstm, limiar_lstm = detectar_anomalias_lstm(sequencias, modelo_lstm)
    df_anomalias = df_normalizado.copy()
    df_anomalias['Erro_Reconstrucao'] = erros_autoencoder
    df_anomalias['Anomalia_Autoencoder'] = anomalias_autoencoder
    indices_validos = np.arange(timesteps, len(df_anomalias))
    df_anomalias = df_anomalias.iloc[indices_validos]
    df_anomalias['Erro_LSTM'] = erros_lstm
    df_anomalias['Anomalia_LSTM'] = anomalias_lstm
    df_anomalias.to_csv(os.path.join('data', 'anomalias_detectadas.csv'), index=False)
    total_anomalias_autoencoder = np.sum(anomalias_autoencoder)
    total_anomalias_lstm = np.sum(anomalias_lstm)
    print(f"Total de anomalias detectadas pelo Autoencoder: {total_anomalias_autoencoder}")
    print(f"Total de anomalias detectadas pela LSTM: {total_anomalias_lstm}")
