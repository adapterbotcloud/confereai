# scripts/treinar_lstm_por_cargo_paralelo.py

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import os
from concurrent.futures import ProcessPoolExecutor
from utils import criar_sequencias_por_vinculo

np.set_printoptions(threshold=np.inf)

def configurar_gpu():
    # Configurar o growth da memória para GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth habilitado nas GPUs")
        except RuntimeError as e:
            print(e)

def codificar_com_autoencoder(dados, autoencoder):
    # Usar a camada de codificação do Autoencoder
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=1).output)
    dados_codificados = encoder.predict(dados)
    return dados_codificados

def treinar_lstm(sequencias, cargo, epochs=100, batch_size=128):

    timesteps = sequencias.shape[1]
    n_features = sequencias.shape[2]
    modelo_lstm = Sequential()
    modelo_lstm.add(LSTM(50, activation='relu', input_shape=(timesteps, n_features), return_sequences=False))
    modelo_lstm.add(RepeatVector(timesteps))
    modelo_lstm.add(LSTM(50, activation='relu', return_sequences=True))
    modelo_lstm.add(TimeDistributed(Dense(n_features)))
    modelo_lstm.compile(optimizer='adam', loss='mse')
    modelo_lstm.fit(sequencias, sequencias, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1)

    # Salvar o modelo LSTM treinado para o cargo
    modelo_lstm_path = os.path.join('models', f'lstm_pagamentos_{cargo}.keras')
    modelo_lstm.save(modelo_lstm_path)
    print(f"Modelo LSTM para o cargo {cargo} salvo em {modelo_lstm_path}")
    return modelo_lstm_path

def treinar_lstm_para_cargo(cargo, df_normalizado, timesteps=3):

     # Configurar GPU dentro do processo paralelo
    configurar_gpu()

    print(f"Treinando LSTM para o cargo: {cargo}")

     # Organizar os dados por ano e mês antes de criar as sequências
    ##df_cargo = df_cargo.sort_values(by=['isn_vinculo','num_ano', 'num_mes'])

    # Filtrar os dados pelo cargo atual
    df_cargo = df_normalizado[df_normalizado['cod_cargo'] == cargo]
    #dados_rubricas = df_cargo[df_cargo.columns[4:]].values  # Ajustar as colunas de rubricas se necessário
    #Selecionar as colunas de rubricas (a partir da 4ª coluna, ajuste se necessário)
    colunas_rubricas = df_cargo.columns[4:]

    # Carregar o modelo Autoencoder treinado para o cargo
    #modelo_autoencoder_path = os.path.join('models', f'autoencoder_pagamentos_{cargo}.keras')
    
    #if not os.path.exists(modelo_autoencoder_path):
    #    print(f"Modelo Autoencoder para o cargo {cargo} não encontrado. Pulando...")
    #    return None

    #autoencoder = load_model(modelo_autoencoder_path)

    # Codificar os dados usando o Autoencoder
    #dados_codificados = codificar_com_autoencoder(dados_rubricas, autoencoder)

    # Criar sequências temporais para o LSTM
    sequencias_codificadas = criar_sequencias_por_vinculo(df_cargo,colunas_rubricas, timesteps)

    # Verificar se as sequências foram geradas corretamente
    if sequencias_codificadas.size == 0:
        print(f"Falha ao criar sequências para o cargo {cargo}. Pulando...")
        return None

    # Treinar o modelo LSTM para este cargo
    return treinar_lstm(sequencias_codificadas, cargo)

if __name__ == "__main__":
    # Carregar os dados normalizados
    df_normalizado = pd.read_csv(os.path.join('data', 'pagamentos_normalizados.csv'))
    
    # Supondo que 'cod_cargo' seja a coluna com os cargos dos servidores
    cargos_unicos = df_normalizado['cod_cargo'].unique()

    # Definir o número máximo de workers (processos paralelos)
    max_workers = 1

    # Usar ProcessPoolExecutor para paralelizar o treinamento por cargo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(treinar_lstm_para_cargo, cargo, df_normalizado) for cargo in cargos_unicos]

    # Aguardar a finalização de todos os treinamentos paralelos
    for future in futures:
        future.result()

    print("Treinamento LSTM concluído para todos os cargos.")
