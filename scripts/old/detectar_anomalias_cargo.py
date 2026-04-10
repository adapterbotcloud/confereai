import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

def detectar_anomalias_autoencoder_cargo(dados, autoencoder, limiar_percentil=98):
    # Reconstrução usando o Autoencoder
    reconstrucoes = autoencoder.predict(dados)
    
    # Cálculo dos erros de reconstrução
    erros = np.mean(np.power(dados - reconstrucoes, 2), axis=1)
    
    # Definir limiar com base no percentil especificado
    limiar = np.percentile(erros, limiar_percentil)
    
    # Detectar anomalias (erros que excedem o limiar)
    anomalias = erros > limiar
    
    return anomalias, erros, limiar

if __name__ == "__main__":

    # Carregar dados normalizados
    df_normalizado = pd.read_csv(os.path.join('data', 'pagamentos_normalizados.csv'))

    # Supondo que 'cod_cargo' seja a coluna com os cargos dos servidores
    cargos_unicos = df_normalizado['cod_cargo'].unique()

    # Lista para armazenar todas as anomalias
    todas_anomalias = []

    for cargo in cargos_unicos:
        print(f"Detectando anomalias para o cargo: {cargo}")

        # Filtrar os dados pelo cargo atual
        df_cargo = df_normalizado[df_normalizado['cod_cargo'] == cargo]
        dados_rubricas = df_cargo[df_cargo.columns[4:]].values  # Ajustar as colunas de rubricas se necessário

        # Carregar o modelo Autoencoder treinado para o cargo
        modelo_path = os.path.join('models', f'autoencoder_pagamentos_{cargo}.keras')
        
        if not os.path.exists(modelo_path):
            print(f"Modelo para o cargo {cargo} não encontrado. Pulando...")
            continue

        autoencoder = load_model(modelo_path)

        # Detectar anomalias usando o Autoencoder para o cargo específico
        anomalias, erros, limiar = detectar_anomalias_autoencoder_cargo(dados_rubricas, autoencoder)

         # Adicionar os resultados ao DataFrame do cargo atual usando .loc[]
        df_cargo = df_cargo.copy()
        df_cargo.loc[:, 'Erro_Reconstrucao'] = erros
        df_cargo.loc[:, 'Anomalia_Autoencoder'] = anomalias
        df_cargo.loc[:, 'Limiar_Autoencoder'] = limiar

        # Adicionar as anomalias detectadas para este cargo na lista total
        todas_anomalias.append(df_cargo)

    # Concatenar todas as anomalias detectadas para todos os cargos
    df_anomalias_total = pd.concat(todas_anomalias, ignore_index=True)

    # Salvar os resultados em um arquivo CSV
    df_anomalias_total.to_csv(os.path.join('data', 'anomalias_detectadas.csv'), index=False)

    # Total de anomalias detectadas pelo Autoencoder
    total_anomalias_autoencoder = df_anomalias_total['Anomalia_Autoencoder'].sum()
    print(f"Total de anomalias detectadas pelo Autoencoder: {total_anomalias_autoencoder}")
