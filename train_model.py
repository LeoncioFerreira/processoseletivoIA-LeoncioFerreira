import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Variáveis globais
MODEL_PATH = "model.h5"
EPOCHS = 5
BATCH_SIZE = 128
SEED = 42

# Configuracao da seed
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


def load_data():
    """
    Carrega e pré-processa o dataset MNIST.

    Etapas realizadas:
    - Carregamento dos dados de treino e teste
    - Normalização dos pixels para o intervalo [0, 1]
    - Adição de dimensão de canal (necessária para CNN)

    Returns:
        tuple:
            (x_train, y_train), (x_test, y_test)
            onde:
            - x_* possuem shape (N, 28, 28, 1)
            - y_* são os rótulos correspondentes
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)


def build_model():
    """
    Constrói e compila uma CNN simples para classificação de dígitos.

    Decisão arquitetural:
    O modelo conecta o Flatten diretamente à camada Dense(10), sem camadas
    densas intermediárias.

    Testes indicaram que uma camada intermediária não trouxe ganhos relevantes
    de acurácia, pois o MNIST é um problema de baixa complexidade.

    Essa escolha reduz o número de parâmetros e o tamanho do modelo,
    mantendo desempenho adequado e tornando a arquitetura mais eficiente
    para aplicações de Edge AI.

    Arquitetura:
    - Entrada: 28x28x1
    - Conv2D (8 filtros, ReLU)
    - MaxPooling2D
    - Conv2D (16 filtros, ReLU)
    - MaxPooling2D
    - Flatten
    - Dense (10 neurônios, Softmax)

    A arquitetura foi projetada para ser leve e eficiente,
    adequada para execução em ambientes de Edge AI.

    Returns:
        tf.keras.Model: modelo compilado pronto para treinamento
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    """
    Executa o pipeline completo de treinamento:

    Etapas:
    1. Carrega e prepara os dados
    2. Constrói o modelo CNN
    3. Realiza o treinamento
    4. Avalia o modelo no conjunto de teste
    5. Salva o modelo treinado em disco

    Outputs:
    - Impressão das métricas (loss e accuracy)
    - Arquivo 'model.h5' salvo no diretório atual
    """

    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=EPOCHS,
              batch_size=BATCH_SIZE, verbose=2)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"test_loss: {test_loss:.4f}")
    print(f"test_accuracy: {test_accuracy:.4f}")
    model.save(MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
