# scripts/treinar_lstm_por_cargo.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import os
from utils import criar_sequencias

def codificar_com_autoencoder(dados, autoencoder):
    # Usar a camada de codificação do Autoencoder
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=1).output)
    dados_codificados = encoder.predict(dados)
    return dados_codificados

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
    # Carregar os dados normalizados
    df_normalizado = pd.read_csv(os.path.join('data', 'pagamentos_normalizados.csv'))
    
    # Supondo que 'cod_cargo' seja a coluna com os cargos dos servidores
    cargos_unicos = df_normalizado['cod_cargo'].unique()

    timesteps = 12

    for cargo in cargos_unicos:
       
        # Filtrar os dados pelo cargo atual
        df_cargo = df_normalizado[df_normalizado['cod_cargo'] == cargo]
        dados_rubricas = df_cargo[df_cargo.columns[4:]].values  # Ajustar as colunas de rubricas se necessário

        # Carregar o modelo Autoencoder treinado para o cargo
        modelo_autoencoder_path = os.path.join('models', f'autoencoder_pagamentos_{cargo}.keras')
        
        if not os.path.exists(modelo_autoencoder_path):
            print(f"Modelo Autoencoder para o cargo {cargo} não encontrado. Pulando...")
            continue

        autoencoder = load_model(modelo_autoencoder_path)

        # Codificar os dados usando o Autoencoder
        dados_codificados = codificar_com_autoencoder(dados_rubricas, autoencoder)

        # Criar sequências temporais para o LSTM
        sequencias_codificadas = criar_sequencias(dados_codificados, timesteps)

        # Treinar o modelo LSTM para este cargo
        modelo_lstm = treinar_lstm(sequencias_codificadas)

        # Salvar o modelo LSTM treinado para o cargo
        modelo_lstm_path = os.path.join('models', f'lstm_pagamentos_{cargo}.keras')
        modelo_lstm.save(modelo_lstm_path)
       