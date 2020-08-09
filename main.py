
from math import sqrt
from random import seed
from random import random
import numpy as numpy
import pandas as pd
import time

treino = pd.read_csv("Safra_2018-2019.csv")
#teste = pd.read_csv("Safra_2020.csv")
atributos = treino.columns[2:9]
index = treino.columns[1]
classe = treino.columns[10]
atributo_problema = treino.columns[7] #há valores nulos nesta coluna, a ideia vai ser rodar um knn e preencher os campos vazios com o valor dos vizinhos mais próximos
treino_ok = treino.loc[treino[atributo_problema] == treino[atributo_problema]]#testar se o valor é igual a ele mesmo valida que não é NaN
treino_nao_ok = treino.loc[treino[atributo_problema] != treino[atributo_problema]]#se ele for diferente dele mesmo, então é NaN StackOverflow Rules



def distancia_euclidiana(linha1, linha2):
    return sqrt(numpy.power(numpy.sum(linha1-linha2),2))
#Encontrar os vizinhos mais proximos
def get_vizinhos(train, test_linha, num_vizinhos): 
    #distancias = numpy.zeros((train.shape[0], 2))
    distancias = pd.DataFrame(0, index=train.index, columns=['index','distancia'])
    distancias.index = train.index
    vizinhos = numpy.zeros((num_vizinhos, 2))
    test_linha = test_linha.values
    print(train.index)
    print(distancias)

    for x in train.index:
        linha = train.loc[x].values
        print(x)
        distancias.iloc[x]['distancia'] = distancia_euclidiana(linha, test_linha)
        distancias.iloc[x]['index'] = x
    print(distancias)
    exit()
    ordenados = distancias[numpy.argsort(distancias[:, 1])]
    vizinhos =ordenados[0:num_vizinhos]
    return vizinhos



def classificar(train, test_linha, num_neighbors):
    #print(train.columns.difference([train.columns[len(train.columns)-1]]))
    #print(train[train.columns.difference([train.columns[len(train.columns)-1]])])
    #print(test_linha[train.columns.difference([train.columns[len(train.columns)-1]])])
    treino = train[train.columns.difference([train.columns[len(train.columns)-1]])]
    linha_teste = test_linha[train.columns.difference([train.columns[len(train.columns)-1]])]
    vizinhos = get_vizinhos(treino,linha_teste, num_neighbors)
    votos = numpy.zeros(3)
    exit()
    for x in vizinhos[:,0].astype(int):
        votos[train[x][3]] = votos[train[x][3]]+1 
    return numpy.argmax(votos)


treino_correcao = treino_ok[treino.columns[[2,3,4,5,6,8,9,10,7]]]
base_correcao = treino_nao_ok[treino.columns[[2,3,4,5,6,8,9,10,7]]]
#print(treino.columns[[2,3,4,5,6,8,9,10,7]])
print(atributo_problema)

#A ideia principal do que tentei desenvolver com este dataset era corregir o atributo problema(Semanas_Utilizando com valores nulos), através de uma classificação das instâncias inconsistentes
#atrav[es do KNN para gera uma média do valor entre os 5 vizinhos mais próximos para poder nao gerar valores absurdos para a instancia.
#O knn foi escolhido pois é o classificador que implementei por ultimo numa disciplina da minha licenciatura em computação e estava mais fresco.
# Porém poderia utilizar uma árvore de decisão com um equação de regressão em cada uma das folhas, ou entao uma rede neural artificial.
# De todo modo, o desempenho neste teste seletivo foi duramente influenciado pelo fato de eu estar cumprindo home office em horario comercial e também ter retomado algumas 
#atividades academicas(Faculdade ead e espanhol), então não tive o tempo que gostaria para explorar este dataset. 
# um outro ponto que influencio no resultado final deste teste foi o reencontro com a linguagem e suas libs, eu tinha em mente todo o pipeline necessário para resolver
#o problema, porém as incansáveis consultas no stackoverflow ou qualquer outro forum causaram lentidão na solução.



#Encontrando o valor minino e máximo dentro do dasaset
def minmax_dataset(data):
    minmax = numpy.zeros((len(data.columns),2))
    data = numpy.matrix(data)
    x=0
    for column in data.T:
        minmax[x][0] = numpy.min(column)
        minmax[x][1] = numpy.max(column)
        x=x+1

#Escalando o dataset entre 0 e 1
def normalizar_dataset(dataset, minmax):
	for linha in dataset:
		for i in range(len(linha)-1):
			linha[i] = (linha[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

