from dicionario_num import bipolar_numbers as numeros

def main():
    pesos = n_perceptron(numeros)
    numero_ruido = [
        1,  1,  1,  1,  -1,  1,  1,  1,  1,
        1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, -1, 1, -1, -1, -1, -1, -1, -1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,
        -1, -1, -1, -1, -1, -1, -1, -1,  1,
        -1, -1, -1, -1, -1, -1, 1, -1,  1,
        -1, -1, -1, 1, -1, -1, -1, 1,  1,
        -1, -1, -1, -1, -1, -1, -1, -1,  1,
        1,  -1,  -1,  1,  1,  -1,  1,  1,  -1
    ]


    print_numero_9x9(numero_ruido)
    print()

    numero_encontrado = reconhece_numero(numero_ruido, pesos)
    for i in range(len(numero_encontrado)):
        print(f"Saída para {i} = {numero_encontrado[i]}")
        if numero_encontrado[i] == 1:
            saida = i
    print(f"\nNúmero reconhecida como: número {saida}\n")
    print_numero_9x9(numeros[saida])


def perceptron(entradas, saidas): # encontra os pesos para 1 neurónio
    b = 0 # bias
    w = [0 for x in range(len(entradas[0]))] # weights
    l_rate = 1 # learning rate
    while True:
        new_w = False
        for j in range(len(entradas)):
            y_in = b
            for i in range(len(entradas[j])):
                y_in += entradas[j][i] * w[i]
            y = 1 if (y_in > 0) else -1
            if y != saidas[j]:
                new_w = True
                for i in range(len(entradas[j])):
                    w[i] += l_rate * entradas[j][i] * saidas[j]
                b += l_rate * saidas[j]
        if not new_w:
            return w, b

def n_perceptron(dicionario_entradas): # encontra os pesos para n neurónios
    dicionario_pesos = {}
    array = list(dicionario_entradas.values())
    
    for numero in dicionario_entradas:
        saida = []
        for key in dicionario_entradas.keys():
            saida.append(1 if key == numero else -1)
        dicionario_pesos[numero] = perceptron(array, saida)
    
    return dicionario_pesos # formato da saida: {'numero' : ([lista de pesos], bias)}

def hebb_rule(numero, pesos, bias):
    y_liq = bias
    for i in range(len(pesos)):
        y_liq += pesos[i] * numero[i]
    
    return 1 if y_liq > 0 else -1

def reconhece_numero(numero, pesos):
    saida = []
    for numero_analise, w_b in pesos.items():
        saida.append(hebb_rule(numero, w_b[0], w_b[1]))
    return saida

def print_numero_9x9(numero):
    for i in range(9):
        for j in range(9):
            print(f"{'.' if numero[9*i+j] == -1 else '#'}", end='')
        print()

if __name__ == "__main__":
    main()