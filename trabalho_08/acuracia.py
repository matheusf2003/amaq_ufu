import numpy as np
from pesos_salvos import v, bv, w, bw  # Carregar os pesos salvos
from tensorflow.keras.datasets import mnist

# Carregar dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pré-processamento: achatar as imagens 28x28 para vetores de 784 elementos e normalizar
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# Transformar as saídas em one-hot encoding
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

# Função de ativação sigmoid bipolar
def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

# Testar a rede usando os pesos salvos
def testar_rede(x_test):
    resultados = []
    for i in range(len(x_test)):
        # Forward pass
        zin = np.dot(x_test[i], v) + bv
        z = bipolar_sigmoid(zin)

        yin = np.dot(z, w) + bw
        y = bipolar_sigmoid(yin)

        # A saída y representa uma previsão no formato one-hot. Vamos pegar o índice com o maior valor.
        predicao = np.argmax(y)
        resultados.append(predicao)
    
    return resultados

# Executar o teste
predicoes = testar_rede(x_test)

# Comparar as previsões com as saídas reais (y_test)
acertos = np.sum(np.argmax(y_test_onehot, axis=1) == predicoes)
total = len(y_test)

# Calcular a acurácia
acuracia = acertos / total
print(f"Acurácia: {acuracia * 100:.2f}%")
