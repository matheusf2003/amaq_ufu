import numpy as np
import pandas
from dependencies.funcoes import dict_to_list
from results.pesos_salvos import v, bv, w, bw  # Carregar os pesos salvos

# Caminho para arquivo com a base de dados
file_path = "data_base/data_base.ods"

# Leitura do arquivo .xlsx
df = pandas.read_excel(file_path)

data = df.to_dict(orient="records")
(x_test, y_test) = dict_to_list(data)

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
acertos = np.sum(np.argmax(y_test, axis=1) == predicoes)
total = len(y_test)

# Calcular a acurácia
acuracia = acertos / total
print(f"Acurácia: {acuracia * 100:.2f}%")
