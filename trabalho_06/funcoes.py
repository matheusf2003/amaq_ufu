from random import uniform as randomfloat   # função para criar pesos aleatorios

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
    return w, b

def neoronio(entrada, pesos, bias):
    y_liq = bias
    for i in range(len(pesos)):
        y_liq += pesos[i] * entrada
    
    return y_liq