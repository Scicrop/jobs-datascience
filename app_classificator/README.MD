#  Classificador de danos em plantações

No arquivo 'Classificacao_danos.ipynb' pode ser encontrado todo o processo de desenvolvimento, desde a analise dos dados até o modelo final

### Pasta Classificator

O intuito dessa parte do projeto é facilitar a execução, o deploy e a classificação de novos datasets

Para classificar os danos sofridos em um novo conjunto de dados, basta executar o seguinte comando no terminal

```bash
python classificator.py "endereco/do/arquivo/safra_20xx.csv"
```

Além disso, o classificador conta com o arquivo \_\_init\_\_.py o que possibilita sua execução como uma biblioteca, sendo possivel reaproveitar tudo na hora de implementar no backend de um processo.

Para usa-lo, basta:

```python
from Classificator.classificator import Classificador

df = Classificador().preprocessamento(data_frame)

Classificador().classify(df, "nome_do_arquivo_para_salvar")

```
