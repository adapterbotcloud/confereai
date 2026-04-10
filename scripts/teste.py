from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Dados originais com menos de 12 meses
x_train_cargo_a = np.array([
    # Servidor 1 do Cargo A com 3 meses
    [[0.5, 0.75, 0.6667, 0.6667], 
     [0.525, 0.75, 0.6833, 0.7], 
     [0.55, 0.75, 0.7, 0.7333]],

    # Servidor 2 do Cargo A com 3 meses
    [[0.4, 0.65, 0.5, 0.4], 
     [0.45, 0.7, 0.55, 0.45], 
     [0.5, 0.75, 0.6, 0.5]]
])

# Aplicar padding para garantir que todos os servidores tenham 12 meses
x_train_pad = pad_sequences(x_train_cargo_a, maxlen=12, dtype='float32', padding='post', value=0.0)

print(x_train_pad)
