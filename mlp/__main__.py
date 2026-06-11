import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from .network import MLP
from datetime import datetime
import json


def accuracy(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    return np.mean(y_true == y_pred)


def plot_history(history, save_path=None):
    epochs = range(len(history["loss"]))

    plt.figure(figsize=(12,5))

    # Perda
    plt.subplot(1,2,1)
    plt.plot(epochs, history["loss"])
    plt.title("Loss ao longo do treino")
    plt.xlabel("Época")
    plt.ylabel("Perda")

    # Precisão
    plt.subplot(1,2,2)
    plt.plot(epochs, history["accuracy"])
    plt.title("Precisão ao longo do treino")
    plt.xlabel("Época")
    plt.ylabel("Precisão")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def train(args):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    classes = np.unique(y_train)

    y_train = np.eye(len(classes))[np.searchsorted(classes, y_train)]
    y_test = np.eye(len(classes))[np.searchsorted(classes, y_test)]

    input_size = X_train.shape[1]
    output_size = len(y_train[0])

    model = MLP(
        input_size=input_size,
        hidden1_size=args.hidden1,
        hidden2_size=args.hidden2,
        output_size=output_size,
        learning_rate=args.lr
    )

    losses = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("\n===== RESULTADOS =====")
    print(f"Accuracy: {acc:.4f}")

    # Salva as métricas e o gráfico de histórico em um diretório organizado por timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Salva gráfico de histórico
    plot_path = os.path.join(run_dir, "history.png")
    plot_history(losses, save_path=plot_path)

    # Salva métricas em JSON
    metrics = {
        "accuracy": float(acc),
        "final_loss": float(losses["loss"][-1]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nRun salvo em: {run_dir}")

    if args.plot_history:
        plot_history(losses)


def main():
    parser = argparse.ArgumentParser(
        prog="mlp",
        description="CLI para treinamento e avaliação do MLP"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de épocas"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamanho do batch"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate"
    )

    parser.add_argument(
        "--hidden1",
        type=int,
        default=64,
        help="Neurônios da primeira camada oculta"
    )

    parser.add_argument(
        "--hidden2",
        type=int,
        default=32,
        help="Neurônios da segunda camada oculta"
    )

    parser.add_argument(
        "--plot-history",
        action="store_true"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()