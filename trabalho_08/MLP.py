import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist     # dataset MNIST

# Função de ativação: sigmoid bipolar (mantendo a original)
def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

# Derivada da sigmoid bipolar
def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + x) * (1 - x)

# Carregar dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pré-processamento: achatar as imagens 28x28 para vetores de 784 elementos e normalizar
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# Transformar as saídas em one-hot encoding
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]


# Parâmetros da rede
neuroniosentrada = x_train.shape[1]  # 784 neurônios de entrada (28x28 pixels)
neuroniosescondidos = 64             # Pode ser ajustado
neuroniossaida = 10                  # 10 classes (dígitos de 0 a 9)
alfa = 0.03                          # Taxa de aprendizado
numciclo = 10000                     # Número de ciclos de treinamento
errototaladmissivel = 0.001

# Inicializar pesos e bias
v = np.random.rand(neuroniosentrada, neuroniosescondidos) - 0.5
bv = np.random.rand(neuroniosescondidos) - 0.5
w = np.random.rand(neuroniosescondidos, neuroniossaida) - 0.5
bw = np.random.rand(neuroniossaida) - 0.5

# Loop de treinamento
erroquadraticototal = np.zeros(numciclo)
ciclo = 0
errototal = 10

while ciclo < numciclo and errototal > errototaladmissivel:
    ciclo += 1
    errototal = 0

    for i in range(100):
        # Forward pass
        zin = np.dot(x_train[i], v) + bv
        z = bipolar_sigmoid(zin)

        yin = np.dot(z, w) + bw
        y = bipolar_sigmoid(yin)  # Modifique aqui para a função softmax, se desejar.

        # Calcular erro e backpropagation
        target = y_train_onehot[i]
        erro = target - y

        # Atualização de pesos
        deltinhaw = erro * bipolar_sigmoid_derivative(y)
        deltaw = alfa * np.outer(z, deltinhaw)

        deltabw = alfa * deltinhaw
        deltinhav = np.dot(deltinhaw, w.T) * bipolar_sigmoid_derivative(z)
        deltav = alfa * np.outer(x_train[i], deltinhav)

        deltabv = alfa * deltinhav

        # Atualizar pesos e bias
        w += deltaw
        bw += deltabw
        v += deltav
        bv += deltabv

        # Acumular erro quadrático
        errototal += np.sum(erro ** 2)

    erroquadraticototal[ciclo - 1] = errototal

    if ciclo % 100 == 0:
        print(f"Ciclo {ciclo}, Erro total: {errototal}")

# Plotar curva de erro
plt.plot(erroquadraticototal, 'r.')
plt.title('Curva do Erro Quadratico Total')
plt.xlabel('Ciclos')
plt.ylabel('Erro quadrático')
plt.savefig('grafico03.png')

print('Treinamento finalizado')

# Salvar pesos e bias em um arquivo .py
with open('pesos_salvos.py', 'w') as f:
    f.write(f'v = {v.tolist()}\n')
    f.write(f'bv = {bv.tolist()}\n')
    f.write(f'w = {w.tolist()}\n')
    f.write(f'bw = {bw.tolist()}\n')

print("Pesos e bias salvos em 'pesos_salvos.py'")

print('Teste da rede treinada')
for padroes in range(100):
    zin = np.dot(x_test[padroes], v) + bv
    z = bipolar_sigmoid(zin)
    
    yin = np.dot(z, w) + bw
    y = bipolar_sigmoid(yin)
    
    print(f"t: {y_test[padroes]:.6f}   y: {y[0]:.6f}")