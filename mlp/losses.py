import numpy as np

# Função de entropia cruzada, usada no cálculo da perda
def cross_entropy(y, y_pred):
    return -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1)) # Adiciona-se 1e-8 para evitar caso de log(0)