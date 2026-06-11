import numpy as np
from .activations import relu_derivative

# Função de Descida de Gradiente Estocástica, utilizada para a atualização de pesos baseado no 
def sgd(self, X, y, n, z1, a1, z2, a2, y_pred):
    # Backpropagation
    delta3 = (y_pred - y) / n
    d_weights3 = a2.T @ delta3
    d_bias3 = np.sum(delta3, axis=0, keepdims=True)

    delta2 = (delta3 @ self.weights3.T) * relu_derivative(z2)
    d_weights2 = a1.T @ delta2
    d_bias2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = (delta2 @ self.weights2.T) * relu_derivative(z1)
    d_weights1 = X.T @ delta1
    d_bias1 = np.sum(delta1, axis=0, keepdims=True)

    return d_weights1, d_weights2, d_weights3, d_bias1, d_bias2, d_bias3