# scripts/treinar_autoencoder.py

import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import os

def treinar_autoencoder(dados, encoding_dim=100, epochs=200, batch_size=256):
    input_dim = dados.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoding_dim2 = encoding_dim//2

    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    encoded = Dense(encoding_dim2, activation='relu')(encoded)

    decoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(dados, dados, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1)
    return autoencoder

if __name__ == "__main__":

    df_normalizado = pd.read_csv(os.path.join('data', 'pagamentos_normalizados.csv'))
    
    # Supondo que 'Cargo' seja a coluna com os cargos dos servidores
    cargos_unicos = df_normalizado['cod_cargo'].unique()
    
    # Usar apenas as rubricas sem aplicar normalização adicional
    colunas_rubricas = df_normalizado.columns[4:]  # Selecionando as rubricas a partir da 4ª coluna

    for cargo in cargos_unicos:
        
        # Filtrar os dados por cargo
        df_cargo = df_normalizado[df_normalizado['cod_cargo'] == cargo]
        dados_rubricas = df_cargo[colunas_rubricas].values
        
        # Treinar o Autoencoder para este cargo
        autoencoder = treinar_autoencoder(dados_rubricas)
        
        # Criar o caminho de destino para salvar o modelo
        modelo_path = os.path.join('models', f'autoencoder_pagamentos_{cargo}.keras')
        
        # Salvar o modelo
        autoencoder.save(modelo_path)
      