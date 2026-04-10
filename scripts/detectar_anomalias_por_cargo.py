import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

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

if __name__ == "__main__":
    # Definir o caminho base para os arquivos normalizados por cargo
    caminho_base = 'data/teste'
    
    # Identificar todos os arquivos de pagamento normalizado por cargo
    arquivos_pagamentos = [f for f in os.listdir(caminho_base) if f.startswith('pagamentos_normalizados_cargo_') and f.endswith('.csv')]

    # Lista para armazenar os resultados de anomalias
    todas_anomalias = []

    for arquivo in arquivos_pagamentos:
        # Extrair o nome do cargo a partir do nome do arquivo
        cargo = arquivo.replace('pagamentos_normalizados_cargo_', '').replace('.csv', '')
        caminho_arquivo = os.path.join(caminho_base, arquivo)
        
        print(f"Detectando anomalias para o cargo: {cargo}")

        # Carregar os dados normalizados para o cargo atual
        df_cargo = pd.read_csv(caminho_arquivo)
        colunas_rubricas = df_cargo.columns[4:]  # Selecionando as rubricas a partir da 4ª coluna
        dados_rubricas = df_cargo[colunas_rubricas].values

        # Carregar o modelo Autoencoder treinado para o cargo
        modelo_autoencoder_path = os.path.join('models', f'autoencoder_pagamentos_{cargo}.keras')

        # Verificar se o modelo existe antes de tentar carregar
        if not os.path.exists(modelo_autoencoder_path):
            print(f"Modelo Autoencoder para o cargo {cargo} não encontrado. Pulando...")
            continue
        else:
            print(f"Carregando modelo Autoencoder para o cargo {cargo}")

        # Tentar carregar o modelo
        try:
            autoencoder = load_model(modelo_autoencoder_path)
        except Exception as e:
            print(f"Erro ao carregar o modelo para o cargo {cargo}: {e}")
            continue

        # Detectar anomalias usando o Autoencoder
        anomalias_autoencoder, erros_autoencoder, erros_por_rubrica, anomalias_por_rubrica = detectar_anomalias_autoencoder(dados_rubricas, autoencoder)

        # Ajustar o DataFrame para registrar os resultados
        df_cargo = df_cargo.copy()
        df_cargo.loc[:, 'Erro_Reconstrucao'] = erros_autoencoder
        df_cargo.loc[:, 'Anomalia_Autoencoder'] = anomalias_autoencoder

        # Adicionar uma coluna para armazenar a lista de rubricas anômalas e seus erros
        df_cargo['Rubricas_Anomalas'] = None

        # Iterar sobre as linhas para criar a lista de rubricas anômalas com erros de reconstrução
        for i, anomalia in enumerate(anomalias_autoencoder):
            if anomalia:
                # Pegar os índices das rubricas que foram anômalas
                rubricas_anomalas = np.where(anomalias_por_rubrica[i])[0]
                # Pegar os nomes das rubricas que foram anômalas
                lista_rubricas_anomalas = [
                    f"{colunas_rubricas[idx]}: {erros_por_rubrica[i, idx]}" for idx in rubricas_anomalas
                ]
                # Adicionar a lista de rubricas anômalas à coluna correspondente
                df_cargo.at[df_cargo.index[i], 'Rubricas_Anomalas'] = lista_rubricas_anomalas

        # Adicionar os resultados do cargo atual na lista total
        todas_anomalias.append(df_cargo)

    # Concatenar todas as anomalias detectadas para todos os cargos
    df_anomalias_total = pd.concat(todas_anomalias, ignore_index=True)

    # Salvar os resultados em um arquivo CSV
    df_anomalias_total.to_csv(os.path.join('data', 'anomalias_detectadas_por_cargo_com_rubricas.csv'), index=False)

    # Calcular e imprimir o total de anomalias detectadas
    total_anomalias_autoencoder = np.sum(df_anomalias_total['Anomalia_Autoencoder'])
    print(f"Total de anomalias detectadas pelo Autoencoder: {total_anomalias_autoencoder}")
