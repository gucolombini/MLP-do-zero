import numpy as np
from .activations import relu, softmax
from .losses import cross_entropy
from .optimizers import sgd

# Um percéptron com 2 camadas ocultas (ReLU -> ReLU -> softmax)
# Baseado no código do artigo https://abtinmy.github.io/CS-SBU-NeuralNetwork/lectures/introduction/MLP-Scratch-Iris
class MLP:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # inicialização He (Kaiming initialization) para ReLU
        # evita gradientes muito pequenos/explodindo e mantém variância estável entre camadas
        self.weights1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2 / input_size)
        self.weights2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2 / hidden1_size)
        self.weights3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2 / hidden2_size)

        # Vieses iniciados em 0
        self.bias1 = np.zeros((1, self.hidden1_size))
        self.bias2 = np.zeros((1, self.hidden2_size))
        self.bias3 = np.zeros((1, self.output_size))

    def forward(self, X):
        z1 = X @ self.weights1 + self.bias1
        a1 = relu(z1)
        z2 = a1 @ self.weights2 + self.bias2
        a2 = relu(z2)
        z3 = a2 @ self.weights3 + self.bias3
        a3 = softmax(z3)
        return z1, a1, z2, a2, z3, a3

    def fit(self, X, y, epochs=1000, batch_size=64):
        n = X.shape[0]

        history = {
            "loss": [],
            "accuracy": []
        }

        for epoch in range(epochs):

            # Embaralha os dados em ordem aleatória
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

            # Treinamos o modelo em batches para evitar sobrecarregar a máquina
            total_loss = 0 # Aqui guardaremos a soma de toda a perda de cada um dos batches.

            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                m = X_batch.shape[0]
            
                # Forward pass
                z1, a1, z2, a2, z3, y_pred = self.forward(X_batch)

                loss = cross_entropy(y_batch, y_pred)
                total_loss += loss * m

                # Backpropagation
                d_weights1, d_weights2, d_weights3, d_bias1, d_bias2, d_bias3 = sgd(self, X_batch, y_batch, m, z1, a1, z2, a2, y_pred)

                # Atualiza pesos e vieses
                self.weights1 -= self.learning_rate * d_weights1
                self.weights2 -= self.learning_rate * d_weights2
                self.weights3 -= self.learning_rate * d_weights3
                self.bias1 -= self.learning_rate * d_bias1
                self.bias2 -= self.learning_rate * d_bias2
                self.bias3 -= self.learning_rate * d_bias3

            # Perda da época
            epoch_loss = total_loss / n

            # Precisão da época (importante: usa forward no dataset inteiro)
            _, _, _, _, _, y_pred_full = self.forward(X)
            y_pred_labels = np.argmax(y_pred_full, axis=1)
            y_true_labels = np.argmax(y, axis=1)

            acc = np.mean(y_pred_labels == y_true_labels)

            history["loss"].append(epoch_loss)
            history["accuracy"].append(acc)

            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")

        return history

    def predict(self, X):
        _, _, _, _, _, y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
