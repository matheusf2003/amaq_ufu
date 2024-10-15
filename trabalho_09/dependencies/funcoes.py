import numpy as np

# Função de ativação: sigmoid bipolar (mantendo a original)
def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

# Derivada da sigmoid bipolar
def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + x) * (1 - x)

# Função para treinamento do MLP, retorna o erro quadratico, enquanto os pesos serão salvos no arquivo -> arquivo_pesos
def MLP_train(
        x_train,
        y_train_onehot,
        neuroniosentrada,
        neuroniosescondidos,
        neuroniossaida,
        alfa,
        numciclo,
        errototaladmissivel,
        arquivo_pesos):
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

        for i in range(len(x_train)):
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

        if ciclo % 1000 == 0:
            print(f"Ciclo {ciclo}, Erro total: {errototal}")


    # Salvar pesos e bias em um arquivo .py
    with open(f'{arquivo_pesos}', 'w') as f:
        f.write(f"# rede testada com {len(x_train)} imagens e {numciclo} ciclos, com erro final = {erroquadraticototal[-1]}\n")
        f.write(f'v = {v.tolist()}\n')
        f.write(f'bv = {bv.tolist()}\n')
        f.write(f'w = {w.tolist()}\n')
        f.write(f'bw = {bw.tolist()}\n')

    print(f"Pesos e bias salvos em '{arquivo_pesos}'")

    return erroquadraticototal

def dict_to_list(dict):
    x_out = []
    y_out = []
    for i in range(len(dict)):
        x_out.append(list(dict[i].values()))
        if x_out[i][4] == "Iris-setosa":
            y_out.append([1, 0, 0])     #y_out.append(np.eye(3)[0])
        elif x_out[i][4] == "Iris-versicolor":
            y_out.append([0, 1, 0])
        elif x_out[i][4] == "Iris-virginica":
            y_out.append([0, 0, 1])
        x_out[i] = x_out[i][:-1]
    return (x_out, y_out)

