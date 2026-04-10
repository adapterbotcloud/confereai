import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import os
import joblib

def carregar_dados_por_cargo(df, cargo):
    # Filtra os dados para o cargo específico
    df_cargo = df[df['cod_cargo'] == cargo]
    
    # Cria uma tabela pivotando as rubricas como colunas
    df_pivot = df_cargo.pivot_table(index=['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'],
                                    columns='isn_rubrica', values='vlr_calculado', aggfunc='mean')
    
    # Resetando o índice para que as colunas 'isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes' voltem a ser colunas normais
    df_pivot.reset_index(inplace=True)
    
    # Ordenando os dados por isn_vinculo, num_ano e num_mes
    df_pivot = df_pivot.sort_values(by=['isn_vinculo', 'num_ano', 'num_mes'])

    return df_pivot

def substituir_valores_ausentes(df, colunas_rubricas, metodo='media'):
    """Substituir valores ausentes por média ou mediana nas colunas de rubricas."""
    if metodo == 'media':
        df[colunas_rubricas] = df[colunas_rubricas].fillna(df[colunas_rubricas].mean())
    elif metodo == 'mediana':
        df[colunas_rubricas] = df[colunas_rubricas].fillna(df[colunas_rubricas].median())
    return df

def normalizar_dados(df, colunas_rubricas):
    """Normalizar dados com MinMaxScaler após tratar valores ausentes."""
    # Substituir valores ausentes por média antes da normalização
    df = substituir_valores_ausentes(df, colunas_rubricas, metodo='media')  # Ou 'mediana'
    
    # Aplicar MinMaxScaler para normalizar as rubricas
    scaler = MinMaxScaler()
    df[colunas_rubricas] = scaler.fit_transform(df[colunas_rubricas])
    
    return df, scaler

def normalizar_dados2(df, colunas_rubricas):
    """Normalizar dados com RobustScaler após tratar valores ausentes."""
    # Substituir valores ausentes por mediana antes da normalização
    df = substituir_valores_ausentes(df, colunas_rubricas, metodo='mediana')  # Ou 'media'
    
    # Aplicar RobustScaler para normalizar as rubricas
    scaler = RobustScaler()
    df[colunas_rubricas] = scaler.fit_transform(df[colunas_rubricas])
    
    return df, scaler

def dividir_por_ano(df_pivot_cargo):
    df_treino_list = []
    df_teste_list = []
    
    # Itera por cada ano e realiza a divisão
    for ano in df_pivot_cargo['num_ano'].unique():
        df_ano = df_pivot_cargo[df_pivot_cargo['num_ano'] == ano]
        
        # Verifica se há amostras suficientes para a divisão
        if len(df_ano) > 1:
            # Dividir os dados deste ano em 80% treino e 20% teste
            df_ano_treino, df_ano_teste = train_test_split(df_ano, test_size=0.2, random_state=42)
            df_treino_list.append(df_ano_treino)
            df_teste_list.append(df_ano_teste)
        else:
            # Se houver apenas 1 ou poucas amostras, coloca tudo no treino
            df_treino_list.append(df_ano)

    # Concatenar as listas em DataFrames finais de treino e teste
    df_treino = pd.concat(df_treino_list) if df_treino_list else pd.DataFrame()
    df_teste = pd.concat(df_teste_list) if df_teste_list else pd.DataFrame()
    
    return df_treino, df_teste

def salvar_normalizado_por_cargo(df, caminho_treino, caminho_teste):
    # Identifica os cargos únicos
    cargos_unicos = df['cod_cargo'].unique()

    for cargo in cargos_unicos:
        print(f"Processando o cargo: {cargo}")
        
        # Carrega os dados do cargo específico
        df_pivot_cargo = carregar_dados_por_cargo(df, cargo)
        
        # Identifica as rubricas que possuem pelo menos um valor diferente de zero
        rubricas_presentes = [col for col in df_pivot_cargo.columns if col not in ['isn_vinculo', 'cod_cargo', 'num_ano', 'num_mes'] and (df_pivot_cargo[col] != 0).any()]
        
        # Dividir os dados por ano e por cargo, respeitando a proporção
        df_treino, df_teste = dividir_por_ano(df_pivot_cargo)

        # Normaliza os dados de treino e teste separadamente
        if not df_treino.empty:
            df_treino_normalizado, scaler_treino = normalizar_dados(df_treino, rubricas_presentes)
            # Define o caminho para salvar os resultados normalizados de treino
            nome_arquivo_treino = f'pagamentos_normalizados_cargo_{cargo}.csv'
            caminho_arquivo_treino = os.path.join(caminho_treino, nome_arquivo_treino)
            df_treino_normalizado.to_csv(caminho_arquivo_treino, index=False)
            
            # Salvar o scaler para o cargo
            scaler_path = os.path.join(caminho_treino, f'scaler_cargo_{cargo}.pkl')
            joblib.dump(scaler_treino, scaler_path)
            
            print(f"Dados normalizados de treino para o cargo {cargo} salvos em {caminho_arquivo_treino}")
            print(f"Scaler salvo em {scaler_path}")
        
        if not df_teste.empty:
            df_teste_normalizado, scaler_teste = normalizar_dados(df_teste, rubricas_presentes)
            # Define o caminho para salvar os resultados normalizados de teste
            nome_arquivo_teste = f'pagamentos_normalizados_cargo_{cargo}.csv'
            caminho_arquivo_teste = os.path.join(caminho_teste, nome_arquivo_teste)
            df_teste_normalizado.to_csv(caminho_arquivo_teste, index=False)

            # Salvar o scaler para o cargo
            scaler_path = os.path.join(caminho_teste, f'scaler_cargo_{cargo}.pkl')
            joblib.dump(scaler_teste, scaler_path)

            print(f"Dados normalizados de teste para o cargo {cargo} salvos em {caminho_arquivo_teste}")
            print(f"Scaler salvo em {scaler_path}")

if __name__ == "__main__":
    caminho_csv = os.path.join('data', 'historico_pagamento_15.csv')
    
    # Carregar os dados gerais (sem pivotar)
    df = pd.read_csv(caminho_csv, delimiter=';')
    
    # Definir o caminho para salvar os arquivos normalizados de treino e teste
    caminho_treino = os.path.join('data', 'treino')
    caminho_teste = os.path.join('data', 'teste')
    
    # Criar as pastas de treino e teste se não existirem
    os.makedirs(caminho_treino, exist_ok=True)
    os.makedirs(caminho_teste, exist_ok=True)
    
    # Salvar os dados normalizados por cargo, dividindo em treino e teste por ano
    salvar_normalizado_por_cargo(df, caminho_treino, caminho_teste)
