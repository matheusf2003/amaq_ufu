from dicionario_letras import letras

def main():
    pesos = n_perceptron(letras)
    
    print("Letra a ser  reconhecida:")
    #Letra d com ruidos
    letraD = [1, 1, 1, 1, 1, -1, -1, 
            -1, 1, -1, -1, -1, 1, -1, 
            -1, 1, -1, -1, -1, -1, 1, 
            -1, 1, -1, -1, -1, -1, 1, 
            -1, 1, -1, 1, -1, -1, 1, 
            -1, 1, -1, -1, -1, -1, 1, 
            -1, 1, -1, -1, -1, -1, 1, 
            -1, 1, -1, -1, -1, 1, 1, 
            1, 1, 1, 1, 1, -1, 1]
    print_letra_9x7(letraD)

    letra_encontrada = reconhece_letra(letraD, pesos)
    print(f"\nLetra reconhecida como: letra {letra_encontrada}")
    print_letra_9x7(letras[letra_encontrada])
    #print(hebb_rule(letras['k'], w, b))


def perceptron(letras, saidas): # encontra os pesos para 1 neurónio
    b = 0 # bias
    w = [0 for x in range(len(letras[0]))] # weights
    l_rate = 1 # learning rate
    while True:
        new_w = False
        for j in range(len(letras)):
            y_in = b
            for i in range(len(letras[j])):
                y_in += letras[j][i] * w[i]
            y = 1 if (y_in > 0) else -1
            if y != saidas[j]:
                new_w = True
                for i in range(len(letras[j])):
                    w[i] += l_rate * letras[j][i] * saidas[j]
                b += l_rate * saidas[j]
        if not new_w:
            return w, b

def n_perceptron(dicionario_letras): # encontra os pesos para n neurónios
    dicionario_pesos = {}
    letras = list(dicionario_letras.values())
    
    for letra in dicionario_letras:
        saida = []
        for key in dicionario_letras.keys():
            saida.append(1 if key == letra else -1)
        dicionario_pesos[letra] = perceptron(letras, saida)
    
    return dicionario_pesos # formato da saida: {'letra' : ([lista de pesos], bias)}

def hebb_rule(letra, pesos, bias):
    y_liq = bias
    for i in range(len(pesos)):
        y_liq += pesos[i] * letra[i]
    
    return 1 if y_liq >= 0 else -1

def reconhece_letra(letra, pesos):
    for letra_analise, w_b in pesos.items():
        if hebb_rule(letra, w_b[0], w_b[1]) == 1:
            return letra_analise
    return -1

def print_letra_9x7(letra):
    for i in range(9):
        for j in range(7):
            print(f"{'.' if letra[7*i+j] == -1 else '#'}", end='')
        print()

if __name__ == "__main__":
    main()