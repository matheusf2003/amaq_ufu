import pandas
import dependencies.funcoes as funcoes
import matplotlib.pyplot as plt

def main():
    # Caminho para arquivo com a base de dados
    file_path = "data_base/data_base.ods"

    # Leitura do arquivo .xlsx
    df = pandas.read_excel(file_path)

    data = df.to_dict(orient="records")
    (x_train, y_train) = funcoes.dict_to_list(data)

    # Parâmetros da rede
    erroquadraticototal = funcoes.MLP_train(
                            x_train,
                            y_train,
                            neuroniosentrada=len(x_train[0]),
                            neuroniosescondidos=64,             # Pode ser ajustado
                            neuroniossaida=3,                   # 3 classes
                            alfa=0.03,                          # Taxa de aprendizado
                            numciclo=10000,                     # Número de ciclos de treinamento
                            errototaladmissivel=0.001,
                            arquivo_pesos="results/pesos_salvos.py")

    # Plotar curva de erro
    plt.plot(erroquadraticototal, 'r.')
    plt.title('Curva do Erro Quadratico Total')
    plt.xlabel('Ciclos')
    plt.ylabel('Erro quadrático')
    plt.savefig('results/grafico.png')

    print('Treinamento finalizado')




if __name__ == "__main__":
	main()
