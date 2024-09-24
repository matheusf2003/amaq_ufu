from funcoes import adaline, neoronio
from basededados import base_dados
import matplotlib.pyplot as plt

def main():
    pesos = adaline([base_dados['x']], base_dados['y'])
    saida = []
    for entrada in base_dados['x']:
        saida.append(neoronio(entrada, pesos[0], pesos[1]))
    
    # Plotar os dados
    plt.scatter(base_dados['x'], base_dados['y'], color='red', label='Pontos experimentais')  # Pontos medidos
    plt.plot(base_dados['x'], saida)

    # Adicionar título e rótulos aos eixos
    plt.savefig('grafico.png')

    

if __name__ == "__main__":
    main()