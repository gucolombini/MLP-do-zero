# MLP do Zero

Implementação em Python de um percéptron com duas camadas ocultas sem utilizar bibliotecas externas, apenas funções de cálculo básicas do NumPy.

A implementação atual utiliza o dataset MNIST para treinamento e execução.

# Instruções de Execução

## Requisitos:

- Python3
- NumPy
- Tensorflow (Para importar dataset MNIST)
- Matplotlib

**Opcional: Crie um ambiente virtual para executar a aplicação** 
```bash
python -m venv venv

# (Windows)
/.venv/Scripts/Activate.ps1

# (Linux)
source .venv/Scripts/activate
```

Instale as dependências de bibliotecas de Python (se já não possuir elas)
```bash
pip install -U numpy tensorflow matplotlib

# OU

pip install -r requirements.txt
```

Feito isso, clone o repositório. Você já pode executar o pacote na raiz dele usando o seguinte comando:
```bash
python -m mlp --help
```

Fazendo isso, você verá instruções dos possíveis argumentos de execução que você pode customizar:

- --epochs: determinar o número de épocas (padrão: 100 épocas)
- --batch-size: determinar o tamanho do batch de treinamento (padrão: 64)
- --lr: determinar a taxa de aprendizado (padrâo: 0.01)
- --hidden1 e --hidden2: determinar a quantidade de neurônios em cada camada (padrão: 64 e 32, respectivamente)
- --plot-history: abre plot com gráficos da perda e precisão ao longo do aprendizado (esse arquivo é gerado e salvo por padrão independente desse parâmetro)

Exemplo de execução: 
```bash
python -m mlp --epochs 500 --lr 0.001 --hidden1 128 --hidden2 64
```

Após o treinamento do modelo, os resultados serão salvos dentro de uma subpasta em *results/run_(data atual)_(hora_atual)*.

# Estrutura de Arquivos

```
.
├── README.md              ← este arquivo
├── mlp/
│   ├── __init__.py
│   ├── network.py         ← implementação do MLP
│   ├── activations.py     ← funções de ativação ReLU e softmax 
│   ├── losses.py          ← cross-entropy
│   └── optimizers.py      ← SGD
├── notebooks/
│   └── experimentos.ipynb ← notebook utilizado para experimentação
├── results/
│   └── run_(data)_(hora)  ← plots e métricas de uma sessão de treino
└── requirements.txt
```

# Validação e comparação de resultados

A MLP foi desenvolvida com o dataset MNIST em mente, então 