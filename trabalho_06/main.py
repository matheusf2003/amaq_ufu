from funcoes import adaline, neoronio, correlacao_pearson
from basededados import base_dados
import matplotlib.pyplot as plt

def main():
    pesos = adaline([base_dados['x']], base_dados['y'])
    saida = []
    for entrada in base_dados['x']:
        saida.append(neoronio(entrada, pesos[0], pesos[1]))

    r = correlacao_pearson(base_dados['x'], base_dados['y'])
    
    # Plotar os dados
    plt.scatter(base_dados['x'], base_dados['y'], color='red', label='Pontos experimentais')  # Pontos medidos
    plt.plot(base_dados['x'], saida, color='blue', label='Regressão Linear')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regressão Linear com ADALINE')
    plt.legend()
    plt.text(1.5, -1.3, f"Correlação de Pearson = {r}\nCoeficiente de Determinação = {r**2}", fontsize=8, color='g')
    # Adicionar título e rótulos aos eixos
    plt.savefig('grafico.png')

    

if __name__ == "__main__":
    main()