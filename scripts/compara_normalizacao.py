import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

def gerar_graficos_distribuicao(df, colunas_rubricas, titulo, caminho_pasta, cargo):
    # Criar a pasta, se não existir
    os.makedirs(caminho_pasta, exist_ok=True)
    
    # Gerar os gráficos de distribuição para as rubricas selecionadas e salvar
    for coluna in colunas_rubricas:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[coluna], kde=True)
        plt.title(f'{titulo} - Rubrica: {coluna}')
        plt.xlabel('Valores')
        plt.ylabel('Frequência')

        # Salvar o gráfico
        caminho_grafico = os.path.join(caminho_pasta, f'{cargo}_{titulo}_{coluna}.png')
        plt.savefig(caminho_grafico)
        plt.close()  # Fecha o gráfico para não ocupar memória

def normalizar_dados(df, colunas_rubricas, normalizador):
    # Aplica o normalizador escolhido (MinMaxScaler ou RobustScaler)
    df_normalizado = df.copy()
    df_normalizado[colunas_rubricas] = normalizador.fit_transform(df[colunas_rubricas])
    return df_normalizado

def comparar_normalizacoes(df, rubricas_presentes, caminho_pasta, cargo):
    # Gerar gráficos da distribuição inicial (antes de qualquer normalização)
    gerar_graficos_distribuicao(df, rubricas_presentes, 'Distribuicao_Inicial', caminho_pasta, cargo)

    # Aplicar e gerar gráficos para MinMaxScaler
    normalizador_minmax = MinMaxScaler()
    df_normalizado_minmax = normalizar_dados(df, rubricas_presentes, normalizador_minmax)
    gerar_graficos_distribuicao(df_normalizado_minmax, rubricas_presentes, 'MinMaxScaler', caminho_pasta, cargo)

    # Aplicar e gerar gráficos para RobustScaler
    normalizador_robust = RobustScaler()
    df_normalizado_robust = normalizar_dados(df, rubricas_presentes, normalizador_robust)
    gerar_graficos_distribuicao(df_normalizado_robust, rubricas_presentes, 'RobustScaler', caminho_pasta, cargo)

def salvar_normalizado_por_cargo(df, caminho_pasta):
    # Identifica os cargos únicos
    cargos_unicos = df['cod_cargo'].unique()

    for cargo in cargos_unicos:
        print(f"Processando o cargo: {cargo}")
        
        # Carrega os dados do cargo específico
        df_pivot_cargo = carregar_dados_por_cargo(df, cargo)
        
        # Identifica as rubricas que possuem pelo menos um valor diferente de zero
        rubricas_presentes = [col for col in df_pivot_cargo.columns if col not in ['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'] and (df_pivot_cargo[col] != 0).any()]
        
        # Comparar as normalizações e salvar os gráficos na pasta "normalizacao"
        comparar_normalizacoes(df_pivot_cargo, rubricas_presentes, caminho_pasta, cargo)

if __name__ == "__main__":
    # Caminho para o arquivo CSV
    caminho_csv = os.path.join('data', 'historico_pagamento_15.csv')
    
    # Carregar os dados gerais (sem pivotar)
    df = pd.read_csv(caminho_csv, delimiter=';')

    # Caminho para salvar os gráficos de normalização
    caminho_pasta_normalizacao = os.path.join('normalizacao')
    
    # Gerar gráficos de normalização sem salvar os dados, apenas os gráficos em uma pasta
    salvar_normalizado_por_cargo(df, caminho_pasta_normalizacao)
