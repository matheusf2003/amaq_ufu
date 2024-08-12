from tabulate import tabulate # Importa a função tabulate do módulo tabulate

def main():
    operacoes_log = [
    "Constante 0 (F0)",
    "AND",
    "A AND NOT B (A · ¬B)",
    "A",
    "NOT A AND B (¬A · B)",
    "B",
    "XOR (A ⊕ B)",
    "OR (A + B)",
    "NOR (¬(A + B))",
    "Equivalência (XNOR)",
    "NOT B (¬B)",
    "B implica A (B → A)",
    "NOT A (¬A)",
    "A implica B (A → B)",
    "NAND (¬(A · B))",
    "Constante 1 (F1)"]

    print("Selecione a operação lógica que deseja realizar:")
    for i in range(16):
        print(f"\t{i+1}. {operacoes_log[i]}")
    print("\t17. Todas as operações")
    opcao = int(input("Digite a opção desejada: "))

    if opcao == 17:
        print('\n')
        for i in range(1, 17):
            tabela = tabela_verdade_bipolar(i)
            print(f"="*69)
            print(f"\nOperação: {operacoes_log[i-1]}\n")
            print(f"Tabela com pesos atualizados para a função: \"{operacoes_log[i-1]}\"(bipolar).")
            tabela_hebb = regra_hebb(tabela)
            print(tabulate(tabela_hebb, ["x1", "x2", "target", "\u0394w1", "\u0394w2", "\u0394b", "w1", "w2", "b"], tablefmt="fancy_grid"))
            print(f"A regra de Hebb {"PODE" if (neuronio_MP(tabela_hebb)) else "NÃO PODE"} encontrar os pesos da rede neural correspondete a função logica: \"{operacoes_log[i-1]}\", quando usada a representação bipolar.\n")

    else:
        tabela = tabela_verdade_bipolar(opcao)
        print(f"\nOperação: {operacoes_log[opcao-1]}")
        tabela_hebb = regra_hebb(tabela)
        print(tabulate(tabela_hebb, ["x1", "x2", "target", "\u0394w1", "\u0394w2", "\u0394b", "w1", "w2", "b"], tablefmt="fancy_grid"))
        print(f"A regra de Hebb {"PODE" if (neuronio_MP(tabela_hebb)) else "NÃO PODE"} encontrar os pesos da rede neural correspondete a função logica: \"{operacoes_log[opcao-1]}\", quando usada a representação bipolar.")
        


def tabela_verdade_bipolar(opcao):
    tabela = []
    match opcao:
        case 1:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, -1])
        case 2:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, 1 if (a == b == 1) else -1])
        
        case 3:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, 1 if (a == b*(-1) == 1) else -1])
        
        case 4:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, a])
        
        case 5:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, 1 if (a*(-1) == b == 1) else -1])

        case 6:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, b])

        case 7:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, 1 if (a != b) else -1])
        
        case 8:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, -1 if (a == b == -1) else 1])

        case 9:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, 1 if (a == b == -1) else -1])

        case 10:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, 1 if (a == b) else -1])

        case 11:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, b*(-1)])

        case 12:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, 1 if (b == -1 or a == 1) else -1])

        case 13:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, a*(-1)])
        
        case 14:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, 1 if (b == 1 or a == -1) else -1])
        
        case 15:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, -1 if (a == b == 1) else 1])
        
        case 16:
            for a in [1, -1]:
                for b in [1, -1]:
                    tabela.append([a, b, 1])

        case _:
            print("Opção inválida")

    return tabela

def regra_hebb(entrada):
    saida = entrada.copy()
    #saida.insert(0,['_','_',0,0,0])
    b = 0
    w = [0, 0]
    t = 0
    for i in range(4):
        saida[i].append(saida[i][0]*saida[i][2])    #delta_w1 = x1 * t
        saida[i].append(saida[i][1]*saida[i][2])    #delta_w2 = x2 * t
        saida[i].append(saida[i][2])                #delta_b  = t

        w[0] += saida[i][3] # w1(new) = w1(old) + delta_w1
        w[1] += saida[i][4] # w2(new) = w2(old) + delta_w2
        b += saida[i][5]    # b(new) = b(old) + delta_b
        
        saida[i].append(w[0])
        saida[i].append(w[1])
        saida[i].append(b)
        
    return saida

def neuronio_MP(tabela_hebb):
    is_correct = True
    w1 = tabela_hebb[3][6]
    w2 = tabela_hebb[3][7]
    b = tabela_hebb[3][8]
    nova_tabela = []
    print("\nCalculando yin:")
    for i in range(4):
        x1 = tabela_hebb[i][0]
        x2 = tabela_hebb[i][1]
        nova_tabela.append([x1, x2])
        yin = w1*x1 + w2*x2 + b
        print(f"yin = {w1} * {x1} + {w2} * {x2} + {b} = {yin}")
        nova_tabela[i].append(1 if (yin >= 0) else -1)
        if nova_tabela[i][2] != tabela_hebb[i][2]:
            is_correct = False
    print(f"\nTabela encontrada com a rede neural:\n{tabulate(nova_tabela, ["x1", "x2", "target"], tablefmt="fancy_grid")}")
    return is_correct


if __name__ == "__main__":
    main()
