# scripts/treinar_lstm.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import os
from utils import criar_sequencias

def treinar_lstm(sequencias, epochs=100, batch_size=128):
    timesteps = sequencias.shape[1]
    n_features = sequencias.shape[2]
    modelo_lstm = Sequential()
    modelo_lstm.add(LSTM(50, activation='relu', input_shape=(timesteps, n_features), return_sequences=False))
    modelo_lstm.add(RepeatVector(timesteps))
    modelo_lstm.add(LSTM(50, activation='relu', return_sequences=True))
    modelo_lstm.add(TimeDistributed(Dense(n_features)))
    modelo_lstm.compile(optimizer='adam', loss='mse')
    modelo_lstm.fit(sequencias, sequencias, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1)
    return modelo_lstm

if __name__ == "__main__":
    df_normalizado = pd.read_csv(os.path.join('data', 'pagamentos_normalizados.csv'))
    colunas_rubricas = df_normalizado.columns[3:]
    dados_rubricas = df_normalizado[colunas_rubricas].values
    timesteps = 12
    sequencias = criar_sequencias(dados_rubricas, timesteps)
    modelo_lstm = treinar_lstm(sequencias)
    modelo_lstm.save(os.path.join('models', 'lstm_pagamentos_p110.keras'))
