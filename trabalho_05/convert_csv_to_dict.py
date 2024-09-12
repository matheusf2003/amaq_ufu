import csv

dados = {}

with open('Basedados_B2.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)  # Usa DictReader para criar um dicionário
    print(reader)
    for linha in reader:
        for chave, valor in linha.items():
            if chave not in dados:
                dados[chave] = []
            if '.' in valor:
                dados[chave].append(float(valor))
            else:
                dados[chave].append(int(valor))

with open('base_dados.py', 'w') as arquivo:
    arquivo.write(f"base_dados = {dados}")