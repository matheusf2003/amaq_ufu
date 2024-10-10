import numpy as np
from pesos_salvos200 import v, bv, w, bw  # Carregar os pesos salvos
from tensorflow.keras.datasets import mnist

# Função de ativação: sigmoid bipolar (mantendo a original)
def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

# Carregar dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pré-processamento: achatar as imagens 28x28 para vetores de 784 elementos e normalizar
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# Transformar as saídas em one-hot encoding
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

todos_y = []
print('Teste da rede treinada')
for padroes in range(100):
    zin = np.dot(x_test[padroes], v) + bv
    z = bipolar_sigmoid(zin)
    
    yin = np.dot(z, w) + bw
    y = bipolar_sigmoid(yin)
    todos_y.append(np.argmax(y))
    
    print(f"t: {y_test[padroes]:.6f}   y: {np.argmax(y):.6f}")