import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Função para carregar os dados por cargo
def carregar_dados_por_cargo(df, cargo):
    # Filtra os dados para o cargo específico
    df_cargo = df[df['cod_cargo'] == cargo]
    
    # Cria uma tabela pivotando as rubricas como colunas
   # df_pivot = df_cargo.pivot_table(index=['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'],
    #                                columns='isn_rubrica', values='vlr_calculado', fill_value=0)

                                    # Cria uma tabela pivotando as rubricas como colunas, mantendo valores ausentes como NaN
    df_pivot = df_cargo.pivot_table(index=['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'],
                                columns='isn_rubrica', values='vlr_calculado', aggfunc='mean')

    
    # Resetando o índice para que as colunas 'isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes' voltem a ser colunas normais
    df_pivot.reset_index(inplace=True)
    
    # Ordenando os dados por isn_vinculo, num_ano e num_mes
    df_pivot = df_pivot.sort_values(by=['isn_vinculo', 'num_ano', 'num_mes'])
    return df_pivot

# Função para gerar Boxplot consolidado
def gerar_boxplot_consolidado(df, rubricas_presentes, caminho_pasta, cargo):
    df_boxplot = df[['cod_cargo'] + rubricas_presentes].melt(id_vars=['cod_cargo'], var_name='Rubrica', value_name='Valor')
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Rubrica', y='Valor', hue='cod_cargo', data=df_boxplot)
    plt.title(f'Boxplot consolidado das rubricas para o cargo: {cargo}')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Salvar o gráfico
    caminho_grafico = os.path.join(caminho_pasta, f'boxplot_consolidado_{cargo}.png')
    plt.savefig(caminho_grafico)
    plt.close()

# Função para gerar Heatmap consolidado
def gerar_heatmap_consolidado(df, rubricas_presentes, caminho_pasta, cargo):
    # Calcular a média das rubricas por cargo
    df_heatmap = df[rubricas_presentes].mean().reset_index()
    df_heatmap.columns = ['Rubrica', 'Valor_Medio']

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heatmap.pivot_table(values='Valor_Medio', index='Rubrica'), annot=True, cmap='coolwarm')
    plt.title(f'Heatmap de rubricas para o cargo: {cargo}')
    
    # Salvar o gráfico
    caminho_grafico = os.path.join(caminho_pasta, f'heatmap_consolidado_{cargo}.png')
    plt.savefig(caminho_grafico)
    plt.close()

# Função para gerar histograma consolidado
def gerar_histograma_consolidado(df, rubricas_presentes, caminho_pasta, cargo):
    plt.figure(figsize=(12, 6))
    sns.histplot(df[rubricas_presentes].melt(value_name='Valor')['Valor'], kde=True)
    plt.title(f'Histograma consolidado dos valores de rubricas - Cargo {cargo}')
    plt.xlabel('Valor das Rubricas')
    plt.ylabel('Frequência')

    # Salvar o gráfico
    caminho_grafico = os.path.join(caminho_pasta, f'histograma_consolidado_{cargo}.png')
    plt.savefig(caminho_grafico)
    plt.close()

# Função para gerar Violin Plot consolidado
def gerar_violinplot_consolidado(df, rubricas_presentes, caminho_pasta, cargo):
    df_violin = df[['cod_cargo'] + rubricas_presentes].melt(id_vars=['cod_cargo'], var_name='Rubrica', value_name='Valor')
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='Rubrica', y='Valor', hue='cod_cargo', data=df_violin, split=True)
    plt.title(f'Violin Plot das rubricas para o cargo: {cargo}')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Salvar o gráfico
    caminho_grafico = os.path.join(caminho_pasta, f'violinplot_consolidado_{cargo}.png')
    plt.savefig(caminho_grafico)
    plt.close()

# Função principal para gerar todos os gráficos consolidados
def gerar_graficos_consolidados(df, caminho_pasta_normalizacao):
    # Identificar cargos únicos
    cargos_unicos = df['cod_cargo'].unique()

    for cargo in cargos_unicos:
        print(f"Processando o cargo: {cargo}")
        
        # Carregar os dados do cargo específico
        df_pivot_cargo = carregar_dados_por_cargo(df, cargo)
        
        # Identificar as rubricas presentes
        rubricas_presentes = [col for col in df_pivot_cargo.columns if col not in ['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'] and (df_pivot_cargo[col] != 0).any()]
        
        # Gerar gráficos de diferentes tipos para cada cargo
        gerar_boxplot_consolidado(df_pivot_cargo, rubricas_presentes, caminho_pasta_normalizacao, cargo)
        gerar_heatmap_consolidado(df_pivot_cargo, rubricas_presentes, caminho_pasta_normalizacao, cargo)
        gerar_histograma_consolidado(df_pivot_cargo, rubricas_presentes, caminho_pasta_normalizacao, cargo)
        gerar_violinplot_consolidado(df_pivot_cargo, rubricas_presentes, caminho_pasta_normalizacao, cargo)

if __name__ == "__main__":
    # Caminho para o arquivo CSV
    caminho_csv = os.path.join('data', 'historico_pagamento_15.csv')

    # Carregar os dados gerais
    df = pd.read_csv(caminho_csv, delimiter=';')

    # Definir o caminho para salvar os gráficos de normalização
    caminho_pasta_normalizacao = os.path.join('normalizacao_consolidada')

    # Criar a pasta de normalização se não existir
    os.makedirs(caminho_pasta_normalizacao, exist_ok=True)

    # Gerar todos os gráficos consolidados
    gerar_graficos_consolidados(df, caminho_pasta_normalizacao)
