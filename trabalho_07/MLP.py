import numpy as np
import matplotlib.pyplot as plt

# Activation function: bipolar sigmoid
def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

# Derivative of bipolar sigmoid
def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + x) * (1 - x)

# Training data
x = np.array([[0], [.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9], [1.0]])
t = np.array([[-.9602], [-.5770], [-.0729], [.3771], [.6405], [.6600], [.4609], [.1336], [-.2013], [-.4344], [-.5000]])

# Plot training data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Dados")
plt.xlabel("x")
plt.ylabel("y")
for i in range(len(t)):
    if t[i] == 1:
        plt.plot(x[i], t[i], 'bd')
    else:
        plt.plot(x[i], t[i], 'rd')

# Network parameters
neuroniosentrada = len(x[0])
neuroniosescondidos = 4
neuroniossaida = 1
alfa = 0.03
errototaladmissivel = 0.001
# numciclo = int(input('Entre com o número de ciclos máximo= '))
numciclo = 10000

# Initialize weights and biases
v = np.random.rand(neuroniosentrada, neuroniosescondidos) - 0.5
bv = np.random.rand(neuroniosescondidos) - 0.5
w = np.random.rand(neuroniosescondidos, neuroniossaida) - 0.5
bw = np.random.rand() - 0.5

# Training loop
erroquadraticototal = np.zeros(numciclo)
ciclo = 0
errototal = 10

while ciclo < numciclo and errototal > errototaladmissivel:
    ciclo += 1
    errototal = 0
    
    for padroes in range(4):
        zin = np.dot(x[padroes], v) + bv
        z = bipolar_sigmoid(zin)
        
        yin = np.dot(z, w) + bw
        y = bipolar_sigmoid(yin)
        
        deltinhaw = (t[padroes] - y) * bipolar_sigmoid_derivative(y)
        deltaw = alfa * deltinhaw * z
        
        deltabw = alfa * deltinhaw
        
        deltinhav = deltinhaw * w.flatten() * bipolar_sigmoid_derivative(z)
        deltav = alfa * np.outer(x[padroes], deltinhav)
        
        deltabv = alfa * deltinhav
        
        w += deltaw.reshape(-1, 1)
        bw += deltabw
        v += deltav
        bv += deltabv
        
        errototal += 0.5 * ((t[padroes] - y) ** 2)
    
    erroquadraticototal[ciclo - 1] = errototal[0]

# Plot error curve
plt.subplot(1, 2, 2)
plt.plot(erroquadraticototal, 'r.')
plt.title('Curva do Erro Quadratico Total')
plt.xlabel('Ciclos')
plt.ylabel('Erro quadratico')


print('Fim do treinamento')
print('Erro quadrático total final:', errototal)
print('Ciclos:', ciclo)

todos_y = []
plt.subplot(1, 2, 1)
print('Teste da rede treinada')
for padroes in range(len(x)):
    zin = np.dot(x[padroes], v) + bv
    z = bipolar_sigmoid(zin)
    
    yin = np.dot(z, w) + bw
    y = bipolar_sigmoid(yin)
    todos_y.append(y[0])
    
    print(f"t: {t[padroes][0]:.6f}   y: {y[0]:.6f}")

plt.plot([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], todos_y)

plt.savefig('grafico.png')