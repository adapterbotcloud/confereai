import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Masking
from sklearn.model_selection import train_test_split
import os
from concurrent.futures import ProcessPoolExecutor

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

def treinar_autoencoder(dados, cargo, encoding_dim, epochs=400, batch_size=256):
    # Configurar GPU dentro do processo paralelo
    configurar_gpu()

    input_dim = dados.shape[1]

    input_layer = Input(shape=(input_dim,))

    # Aplicar máscara para ignorar valores zero
    #masked_input = Masking(mask_value=0.0)(input_layer)

    #encoding_dim = max(10, encoding_dim)

    # Criar o codificador e decodificador com base no `encoding_dim` dinâmico
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    encoded = Dense(encoding_dim // 2, activation='relu')(encoded)

    decoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Verificar se há amostras suficientes para usar o validation_split
    if dados.shape[0] > 10:  # Se houver mais de 10 amostras
        autoencoder.fit(dados, dados, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1)
    else:
        # Caso contrário, treina sem usar o conjunto de validação
        print(f"Cargo {cargo} tem apenas {dados.shape[0]} amostras. Treinando sem validação.")
        autoencoder.fit(dados, dados, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0)
    
    # Criar o caminho de destino para salvar o modelo
    modelo_path = os.path.join('models', f'autoencoder_pagamentos_{cargo}.keras')

    # Verificar e criar o diretório 'models/' se não existir
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Salvar o modelo
    autoencoder.save(modelo_path)
    print(f"Modelo Autoencoder para o cargo {cargo} salvo em {modelo_path}")

    return modelo_path

def processar_cargo(cargo, caminho_arquivo):
    # Carregar o arquivo de pagamento normalizado do cargo
    print(f"Carregando dados para o cargo {cargo}...")
    df_cargo = pd.read_csv(caminho_arquivo)
    
    # Selecionar as rubricas (todas as colunas exceto as primeiras 4: isn_vinculo, cod_cargo, num_ano, num_mes)
    colunas_rubricas = df_cargo.columns[4:]
    dados_rubricas = df_cargo[colunas_rubricas].values
    
    # Ajustar o `encoding_dim` com base na quantidade de rubricas
    num_rubricas = dados_rubricas.shape[1]
    encoding_dim = max(12, num_rubricas)  # Exemplo: usar metade do número de rubricas, com um mínimo de 4

    # Treinar o Autoencoder para este cargo
    return treinar_autoencoder(dados_rubricas, cargo, encoding_dim)

if __name__ == "__main__":
    # Definir o caminho base para os arquivos normalizados por cargo
    caminho_base = 'data/treino'
    
    # Identificar todos os arquivos de pagamento normalizado por cargo
    arquivos_pagamentos = [f for f in os.listdir(caminho_base) if f.startswith('pagamentos_normalizados_cargo_') and f.endswith('.csv')]
    
    # Definir o número máximo de processos paralelos
    max_workers = 4

    # Usar ProcessPoolExecutor para paralelizar o treinamento por cargo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for arquivo in arquivos_pagamentos:
            # Extrair o nome do cargo a partir do nome do arquivo
            cargo = arquivo.replace('pagamentos_normalizados_cargo_', '').replace('.csv', '')
            caminho_arquivo = os.path.join(caminho_base, arquivo)
            
            # Submeter o treinamento do autoencoder para o cargo
            futures.append(executor.submit(processar_cargo, cargo, caminho_arquivo))

    # Aguardar a finalização de todos os treinamentos paralelos
    for future in futures:
        modelo_salvo = future.result()
        if modelo_salvo:
            print(f"Modelo salvo em: {modelo_salvo}")

    print("Treinamento Autoencoder concluído para todos os cargos.")
