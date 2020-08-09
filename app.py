import numpy as np
import pandas as pd

# Importando arquivos csv da safra 2018-2019 e 2020 e transformando em DataFrames
df1 = pd.DataFrame(np.genfromtxt('Safra_2018-2019.csv', delimiter=',', skip_header=1, usecols=(2, 3, 4, 5, 6, 7, 8, 9,
                                                                                               10)))
df2 = pd.DataFrame(np.genfromtxt('Safra_2020.csv', delimiter=',', skip_header=1, usecols=(2, 3, 4, 5, 6, 7, 8, 9)))

# Preenchendo NaN com zeros
df1 = df1.fillna(0)
df2 = df2.fillna(0)

# Transformando em inteiros
df1 = df1.astype(int)
df2 = df2.astype(int)

# Transformando em safra anterior Dataframes menores e listas apenas para simplificar a compreensão do código
dfInsetosAnterior = pd.concat([df1[0], df1[8]], axis=1, join='inner')
dfTipoCultivoAnterior = pd.concat([df1[1], df1[8]], axis=1, join='inner')
dfTipoSoloAnterior = pd.concat([df1[2], df1[8]], axis=1, join='inner')
dfCategoriaPesticidaAnterior = pd.concat([df1[3], df1[8]], axis=1, join='inner')
dfDosesSemanaAnterior = pd.concat([df1[4], df1[8]], axis=1, join='inner')
dfSemanasUtilizandoAnterior = pd.concat([df1[5], df1[8]], axis=1, join='inner')
dfSemanasSemUsoAnterior = pd.concat([df1[6], df1[8]], axis=1, join='inner')
dfTemporadaAnterior = pd.concat([df1[7], df1[8]], axis=1, join='inner')
danosAnterior = list(df1[8])

# Transformando em safra atual listas menores e listas apenas para simplificar a compreensão do código
insetosAtual = list(df2[0])
tipoCultivoAtual = list(df2[1])
tipoSoloAtual = list(df2[2])
categoriaPesticidaAtual = list(df2[3])
dosesSemanaAtual = list(df2[4])
semanasUtilizandoAtual = list(df2[5])
semanasSemUsoAtual = list(df2[6])
temporadaAtual = list(df2[7])


# Função que compara os dados de cada coluna com o resultado da safra
def relacaoDanoAnterior(dfAnterior, colunaAnterior, tipoAnterior):
    # Declaração de variáveis
    aux = list(range(0, 99999))
    j = 0
    soma1 = 0
    soma2 = 0
    soma3 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    a = 0
    b = 0
    c = 0

    media1 = 0
    media2 = 0
    media3 = 0

    # Comparação entre colunas de dados / resultado da safra
    for i in dfAnterior[colunaAnterior]:
        aux[j] = int(dfAnterior[8][j])
        # Verificar se houve danos e por qual motivo (0 = Sem danos / 1 = Danos Externos / 2 = Danos Pesticida )
        if aux[j] == 0:
            soma1 += int(dfAnterior.at[i, colunaAnterior])
            total1 += int(dfAnterior.at[i, colunaAnterior])
            a += 1
        elif aux[j] == 1:
            soma2 += int(dfAnterior.at[i, colunaAnterior])
            total2 += int(dfAnterior.at[i, colunaAnterior])
            b += 1
        elif aux[j] == 2:
            soma3 += int(dfAnterior.at[i, colunaAnterior])
            total3 += int(dfAnterior.at[i, colunaAnterior])
            c += 1
        j += 1

    if a != 0:
        # Média sem gerar danos à safra
        media1 = soma1 / a
    if b != 0:
        # Média com danos externos à safra
        media2 = soma2 / b
    if c != 0:
        # Média com danos por pesticida à safra
        media3 = soma3 / c

    resultado = [media1, media2, media3, soma1, soma2, soma3, a, b, c]
    # Retorno de todas as informações que possam vir a ser necessárias
    return resultado


# Função de comparação entre as informações obtidas da safra anterior com as atuais
def danoSafraAtual(listaAtual, dfAnterior, colunaAnterior, tipoAnterior, categoria, inverso):
    # Declaração de variáveis
    soma = 0
    qnt = 0
    resultadoAnterior = relacaoDanoAnterior(dfAnterior, colunaAnterior, tipoAnterior)
    semDanos = 0
    comDanos = 0
    mediaAnterior = dfAnterior[colunaAnterior].mean()

    # Comparação entre colunas de dados / resultado da safra
    for i in listaAtual:
        soma += listaAtual[i]
        qnt += 1

        # Comparação entre os novos dados e a média sem causar danos à safra
        if listaAtual[i] < resultadoAnterior[0]:
            semDanos += 1

        # Comparação entre os novos dados e a média com danos por pesticida
        elif listaAtual[i] > resultadoAnterior[2]:
            comDanos += 1

    mediaAtual = soma / qnt

    percentual = (mediaAtual * 100 / mediaAnterior) - 100

    # Condicionais para a elaboração de texto com os resultados da verificação
    if mediaAtual < mediaAnterior:
        relacao = 'diminuição'
        consequencia = 'reduzindo'
        if inverso:
            consequencia = 'piorando'
    else:
        relacao = 'aumento'
        consequencia = 'piorando'
        if inverso:
            consequencia = 'melhorando'

    # Impressão dos resultados de comparação
    print('\nA média da safra atual de ', categoria, ' é ', mediaAtual, ', enquanto a média das safras anteriores '
          'era de', mediaAnterior ,'.\nIsso representa', relacao, ' de ', percentual, '%, ',
          consequencia, 'os danos relacionados ao mau uso de pesticidas na safra atual.\n' )

    # Retorno das informações para posterior comparação
    resultado = [mediaAtual, mediaAnterior, percentual, consequencia]
    return resultado


# Função principal
def comparacaoSafras():
    # Declaração de variáveis
    soma = 0
    qnt = 0
    semDanos = 0
    danosExternos = 0
    danosPesticida = 0
    aux = list(range(0, 99999))
    j = 0

    # Chamada das funções
    insetos = danoSafraAtual(insetosAtual, dfInsetosAnterior, 0, 1, 'insetos', False)
    usoPesticida = danoSafraAtual(categoriaPesticidaAtual, dfCategoriaPesticidaAnterior, 3, 1, 'uso de pesticida', True)
    dosesSemana = danoSafraAtual(dosesSemanaAtual, dfDosesSemanaAnterior, 4, 1, 'doses de pesticida por semana', False)

    # Cálculo da média anterior de danos da safra
    for i in danosAnterior:
        soma += danosAnterior[i]
        qnt += 1
        aux[j] = int(dfInsetosAnterior[8][j])
        if aux[j] == 0:
            semDanos += 1
        elif aux[j] == 1:
            danosExternos += 1
        elif aux[j] == 2:
            danosPesticida += 1
        j += 1

    percentualSemDanos = semDanos * 100 / 80000
    percentualDanosExternos = danosExternos * 100 / 80000
    percentualDanosPesticida = danosPesticida * 100 / 80000

    media = soma / qnt

    # Resultado da análise
    print('\nLevando em consideração que na safra anterior', percentualSemDanos, '% das safras foram saudáveis,',
          percentualDanosExternos, '% sofreram danos externos e \napenas',  percentualDanosPesticida, '% sofreram danos'
          ' devido ao mau uso de pesticidas, é possível concluir que as variações anteriores resultarão \nainda em uma'
          ' safra saudável, porém com uma maior quantidade de danos causados por pesticidas.')


# Chamada da função
comparacaoSafras()


# A RESOLVER
# relacaoDanoAnterior('dfSemanasSemUsoAnterior', dfSemanasSemUsoAnterior, 6, 2)
# danoSafraAtual(semanasSemUsoAtual, dfSemanasSemUsoAnterior, 5, 1, 'semanas sem uso', True)


