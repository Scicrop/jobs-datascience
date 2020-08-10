
# Processo seletivo para Estágio em Data Science - Solução para o problema

Neste repositório encontra-se a solução para o problema proposto no processo seletivo de estágio em data science da SciCrop. A proposta consiste em predizer se ocorre danos em plantios, baseado em uma série de fatores e em dados fornecidos dos anos anteriores.

Em cada arquivo `.csv` fornecido, encontram-se as seguintes relações

| Variável                 | Descrição                                                    |
| ------------------------ | ------------------------------------------------------------ |
| Identificador_Agricultor | IDENTIFICADOR DO CLIENTE                                     |
| Estimativa_de_Insetos    | Estimativa de insetos por M²                                 |
| Tipo_de_Cultivo          | Classificação do tipo de cultivo (0,1)                       |
| Tipo_de_Solo             | Classificação do tipo de solo (0,1)                          |
| Categoria_Pesticida      | Informação do uso de pesticidas (1- Nunca Usou, 2-Já Usou, 3-Esta usando) |
| Doses_Semana             | Número de doses por semana                                   |
| Semanas_Utilizando       | Número de semanas Utilizada                                  |
| Semanas_Sem_Uso          | Número de semanas sem utilizar                               |
| Temporada                | Temporada Climática (1,2,3)                                  |
| dano_na_plantacao        | Variável de Predição - Dano no Cultivo (0=Sem Danos, 1=Danos causados por outros motivos, 2=Danos gerados pelos pesticidas) |

Toda o desenvolvimento é discutido em três arquivos:
- **Análise**: a análise dos dados, como correlação e distribuição dos valores, é feita no arquivo notebook [`analise.ipynb`](analise.ipynb)
- **Preparação**: a preparação dos dados, tanto o detreinamento quanto o de predição, é feita no arquivo notebook [`preparacao.ipynb`](preparacao.ipynb)
- **Modelo**: o desenvolvimento do modelo de predição, bem como a análise dos resultados, são discutidos no arquivo notebook [`modelo.ipynb`](modelo.ipynb)

Para cada notebook, ainda, foram criados arquivos `.py` contendo as funções implementadas em cada notebook ([`analysis.py`](`analysis.py`), [`datahandle.py`](`datahandle.py`) e [`models.py`](`models.py`)). 

O resultado da previsão para a safra de 2020 se encontra no arquivo [`previsão_2020.csv`](previsão_2020.csv).