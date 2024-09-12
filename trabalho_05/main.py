from base_dados import base_dados
from random import uniform as randomfloat   # função para criar pesos aleatorios
import matplotlib.pyplot as plt

def main():
    pesos = adaline([base_dados["s1"], base_dados["s2"]],base_dados['t'])

    # Plotar os dados
    plt.plot(pesos[2])

    # Adicionar título e rótulos aos eixos
    plt.title('Erro Quadrático durante o Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático')
    plt.savefig('grafico.png')
    
    y = []
    for i in range(len(base_dados["s1"])):
        y.append(hebb_rule([base_dados["s1"][i], base_dados["s2"][i]], pesos[0], pesos[1]))
        print(f"x1 = {base_dados["s1"][i]}, x2 = {base_dados["s2"][i]}, y = {hebb_rule([base_dados["s1"][i], base_dados["s2"][i]], pesos[0], pesos[1])}")

    print(f"Acuracia de {100 * acuracia(y, base_dados['t'])}%")



def adaline(entradas, saidas):
    w = [randomfloat(-.5,.5) for x in range(len(entradas))] # weights
    l_rate = 0.01 # taxa de aprendizagem(0<l_rate<=1)
    b = 0 # bias
    t = 0.001 # tolerância / condição de parada
    num_ciclos = 160
    erros = []
    for ciclo in range(num_ciclos):
        erroquadratico = 0
        maior_alteracao = 0
        for j in range(len(entradas[0])):
            y_in = b
            for i in range(len(entradas)):
                y_in += entradas[i][j] * w[i]
            erroquadratico += (saidas[j]-y_in)**2
            for i in range(len(entradas)):
                new_w = l_rate * entradas[i][j] * (saidas[j] - y_in)
                w[i] += new_w
                if new_w > maior_alteracao:
                    maior_alteracao = new_w
            b += l_rate * (saidas[j] - y_in)
        erros.append(erroquadratico)
        if maior_alteracao < t:
            print(f"encerrado no ciclo: {ciclo}")
            break
    return w, b, erros


def hebb_rule(entrada, pesos, bias):
    y_liq = bias
    for i in range(len(pesos)):
        y_liq += pesos[i] * entrada[i]
    
    return 1 if y_liq >= 0 else -1


def acuracia(y0, y1):
    erro = 0
    for i in range(len(y0)):
        if y0[i] != y1[i]:
            erro += 1
    return (len(y0) - erro) / len(y0)



if __name__ == "__main__":
    main()