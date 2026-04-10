import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def carregar_dados_por_cargo(df, cargo):
    # Filtra os dados para o cargo específico
    df_cargo = df[df['cod_cargo'] == cargo]
    
    # Cria uma tabela pivotando as rubricas como colunas
    df_pivot = df_cargo.pivot_table(index=['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'],
                                    columns='isn_rubrica', values='vlr_calculado', fill_value=0)
    
    # Resetando o índice para que as colunas 'isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes' voltem a ser colunas normais
    df_pivot.reset_index(inplace=True)
    
    # Ordenando os dados por isn_vinculo, num_ano e num_mes
    df_pivot = df_pivot.sort_values(by=['isn_vinculo', 'num_ano', 'num_mes'])

    return df_pivot

def normalizar_dados(df, colunas_rubricas):
    # Cria o MinMaxScaler e normaliza as colunas de rubricas
    scaler = MinMaxScaler()
    df[colunas_rubricas] = scaler.fit_transform(df[colunas_rubricas])
    return df, scaler

def salvar_normalizado_por_cargo(df, caminho_destino):
    # Identifica os cargos únicos
    cargos_unicos = df['cod_cargo'].unique()

    for cargo in cargos_unicos:
        print(f"Processando o cargo: {cargo}")
        
        # Carrega os dados do cargo específico
        df_pivot_cargo = carregar_dados_por_cargo(df, cargo)

           
        # Identifica as rubricas que possuem pelo menos um valor diferente de zero
        rubricas_presentes = [col for col in df_pivot_cargo.columns if col not in ['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'] and (df_pivot_cargo[col] != 0).any()]
        
        # Normaliza as rubricas presentes
        df_normalizado, scaler = normalizar_dados(df_pivot_cargo, rubricas_presentes)
        
      
        # Define o caminho para salvar os resultados normalizados
        nome_arquivo = f'pagamentos_normalizados_cargo_{cargo}.csv'
        caminho_arquivo = os.path.join(caminho_destino, nome_arquivo)
        df_normalizado.to_csv(caminho_arquivo, index=False)
        
        # Salvar o scaler para o cargo
        scaler_path = os.path.join(caminho_destino, f'scaler_cargo_{cargo}.pkl')
        joblib.dump(scaler, scaler_path)
        
        print(f"Dados normalizados para o cargo {cargo} salvos em {caminho_arquivo}")
        print(f"Scaler salvo em {scaler_path}")

if __name__ == "__main__":
    caminho_csv = os.path.join('data', 'historico_pagamento_2020_2024_15.csv')
    
    # Carregar os dados gerais (sem pivotar)
    df = pd.read_csv(caminho_csv, delimiter=';')
    
    # Definir o caminho para salvar os arquivos normalizados
    caminho_destino = os.path.join('data')
    
    # Salvar os dados normalizados por cargo, removendo rubricas sem registros
    salvar_normalizado_por_cargo(df, caminho_destino)
