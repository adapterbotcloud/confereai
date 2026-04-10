import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def calcular_erro_reconstrucao(dados, autoencoder):
    """Função para calcular o erro de reconstrução (MSE) dos dados usando o autoencoder."""
    reconstrucoes = autoencoder.predict(dados)
    erros_por_linha = np.mean(np.power(dados - reconstrucoes, 2), axis=1)
    return erros_por_linha

def verificar_overfitting(X_train, X_test, autoencoder):
    """Função para verificar overfitting comparando os erros de reconstrução entre treino e teste."""
    
    # Fazer previsões nos dados de treino
    train_mse = calcular_erro_reconstrucao(X_train, autoencoder)
    train_loss_avg = np.mean(train_mse)

    # Fazer previsões nos dados de teste
    test_mse = calcular_erro_reconstrucao(X_test, autoencoder)
    test_loss_avg = np.mean(test_mse)

    # Exibir os erros médios de reconstrução
    print(f"Erro médio de reconstrução nos dados de treino: {train_loss_avg}")
    print(f"Erro médio de reconstrução nos dados de teste: {test_loss_avg}")

    # Comparar os erros para identificar overfitting
    if test_loss_avg > train_loss_avg:
        print("Possível overfitting detectado: o erro de reconstrução é maior no conjunto de teste.")
    else:
        print("Não há sinal claro de overfitting.")
    
    return train_loss_avg, test_loss_avg

if __name__ == "__main__":
    # Definir o caminho base para os arquivos normalizados de treino e teste
    caminho_treino = 'data/treino'
    caminho_teste = 'data/teste'

    # Lista para armazenar os resultados de cada cargo
    resultados = []

    # Identificar todos os arquivos de pagamento normalizado por cargo na pasta de treino
    arquivos_treino = [f for f in os.listdir(caminho_treino) if f.startswith('pagamentos_normalizados_cargo_') and f.endswith('.csv')]

    for arquivo_treino in arquivos_treino:
        # Extrair o nome do cargo a partir do nome do arquivo
        cargo = arquivo_treino.replace('pagamentos_normalizados_cargo_', '').replace('.csv', '')
        caminho_arquivo_treino = os.path.join(caminho_treino, arquivo_treino)
        caminho_arquivo_teste = os.path.join(caminho_teste, f'pagamentos_normalizados_cargo_{cargo}.csv')

        print(f"Verificando overfitting para o cargo: {cargo}")

        # Verificar se o arquivo de teste correspondente existe
        if not os.path.exists(caminho_arquivo_teste):
            print(f"Arquivo de teste para o cargo {cargo} não encontrado. Pulando...")
            continue

        # Carregar os dados de treino e teste
        df_treino = pd.read_csv(caminho_arquivo_treino)
        df_teste = pd.read_csv(caminho_arquivo_teste)

        # Selecionar as rubricas a partir da 4ª coluna (as colunas antes da 4ª não são rubricas)
        colunas_rubricas = df_treino.columns[4:]
        X_train = df_treino[colunas_rubricas].values
        X_test = df_teste[colunas_rubricas].values

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

        # Verificar overfitting nos dados de treino e teste
        train_loss_avg, test_loss_avg = verificar_overfitting(X_train, X_test, autoencoder)

        # Adicionar os resultados para o cargo atual na lista de resultados
        resultados.append({
            'Cargo': cargo,
            'Erro_Treino': train_loss_avg,
            'Erro_Teste': test_loss_avg,
            'Overfitting': test_loss_avg > train_loss_avg
        })

    # Criar um DataFrame com os resultados
    df_resultados = pd.DataFrame(resultados)

    # Salvar o DataFrame em um arquivo CSV
    caminho_resumo = 'resumo_resultados_overfitting.csv'
    df_resultados.to_csv(caminho_resumo, index=False)

    print(f"Resumo dos resultados salvo em: {caminho_resumo}")
