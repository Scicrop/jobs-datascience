Case SciCrop
==============================

Case analítico proposto como etapa do processo seletivo de estágio em data science da SciCrop

Para configurar esse projeto, precisamos seguir os passos abaixo:

```
mkdir <<nome projeto>>
cd <<nome projeto>>
pip3 install virtualenv (se não já tiver instalado)
virtualenv <<nome do env>>
git clone -b solucao-grazielly git@github.com:grazimelo/jobs-datascience.git
source <<nome do env>>/bin/activate
pip3 install jupyter notebook
cd jobs-datascience
pip3 install -r requirements.txt

```
Após isso, devemos instalar a pasta src como um pacote do virtualenv
```
pip3 install --editable .

```

Para utilizar o modelo, precisamos executar:

```
python predict.py <nome do modelo> <nome do arquivo de predição> <nome do arquivo de resultado>

```

Organização do Projeto
------------

    ├── LICENSE
    ├── README.md          <- Descrição do Projeto
    ├── data
    │   ├── results        <- Pasta com os arquivos de previsão gerados pelo modelo.
    │   ├── inter          <- Dados com processamento intermediário (pré-modelagem)
    │   ├── processed      <- Dados com processamento finalizado (pós-pipeline de transformação)
    │   └── raw            <- Dados originais utilizados na modelagem.
    |
    ├── models             <- Arquivos binários (.sav, .pkl) com os modelos e logs de previsão em .txt
    │
    ├── notebooks          <- Jupyter notebooks contendo o passo a passo da solução.
    ├── figures            <- Imagens geradas na análise exploratória e demais etapas do projeto.
    │
    ├── requirements.txt   <- Arquivo que contém todas as dependências do ambiente.
    │
    ├── src                <- Pacote contendo todos os módulos, submódulos, funções e classes do projeto.
    │   ├── __init__.py    <- Torna src um pacote.
    │   ├── data_cleaning.py <- Módulo de limpeza de dados.
    │   ├── selecao_de_features <- Módulo de seleção de features.
    │   │── modelagem_metricas.py <- Módulo de modelagem e avaliação.
    │
    └── predict.py            <- Script que roda o modelo e realiza a previsão.


--------

<p><small>Projeto baseado em <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
