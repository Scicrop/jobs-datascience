#!/usr/bin/env python
# coding: utf-8

# ## Boa tarde! 
# ## Essa é a minha resolução para o problema. No final desse script, é gerado um arquivo .csv que contém os dados do dataset de 2020 com a coluna de predição feita pelo modelo. 

# In[ ]:


## Importando as bibliotecas necessárias

import pandas as pd
import requests
import numpy as np 


# ## Importando os arquivos 

# In[ ]:



## Dados de 2019

url2019 = 'https://raw.githubusercontent.com/Lucca21/jobs-datascience/master/Safra_2018-2019.csv'
df2019 = pd.read_csv(url2019) 
df2019.head()


# In[ ]:


## Dados de 2020

url2020 = 'https://raw.githubusercontent.com/Lucca21/jobs-datascience/master/Safra_2020.csv'
df2020= pd.read_csv(url2020) 
df2020.head()


# In[ ]:


## Definindo as features utilizadas 
## Estou definindo as features me baseando na instrução de remover variáveis de tempo e ambiente.

variaveis = ['Categoria_Pesticida','Doses_Semana','Semanas_Utilizando','Semanas_Sem_Uso']

x = df2019[variaveis]
y = df2019['dano_na_plantacao']


# ## Tratando os NaNs 
# 

# In[ ]:


print(x.isna().sum())


# In[ ]:


## Temos 8055 NaNs referentes a variável 'Semanas_Utilizando'. 
## Para lidar com os NaNs eu optei por substituir pela média. 

med = x['Semanas_Utilizando'].mean()
x['Semanas_Utilizando'] = x['Semanas_Utilizando'].fillna(med)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


## Agora, vou dividir o dataset de 2019 em Treino e Teste, para poder criar e testar a acurácia do meu modelo. 
np.random.seed(90)
X_train, X_valid, y_train,y_valid = train_test_split(x,y, test_size = 0.3)


# ## Criando um modelo de Random Forest.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier(n_estimators = 100,
                                    n_jobs = -1, 
                                    random_state = 0)


# In[ ]:


modelo.fit(X_train,y_train)


# In[ ]:


p = modelo.predict(X_valid)


# In[ ]:


##acuracia do modelo. 

acc = np.mean(y_valid == p)
print(acc)


# In[ ]:


## Acurácia assumindo que todas as plantações não obtiveram dano. 
acc2 = np.mean(y_valid == 0 )
print(acc2)


#    ## Temos acc2 > acc.                                                                                                                  Isso significa que se chutarmos que nenhuma plantação foi danificada, seremos mais precisos do que o modelo. Portanto, o modelo não é muito útil por enquanto. O próximo passo seria descobrir algúm modo de aumentar a precisão do modelo. Tentei aumentar a precisão de algumas formas, mas ainda não consegui, e por isso vou fazer a predição para 2020 com o modelo desse jeito mesmo

# In[ ]:


x_prev = df2020[variaveis]

print(x_prev.isna().sum())       
    


# In[ ]:


## Tratando os NaNs 
med2 = x_prev['Semanas_Utilizando'].mean()
x_prev['Semanas_Utilizando'] = x_prev['Semanas_Utilizando'].fillna(med2)
print(x_prev.isna().sum())                


# In[ ]:


p2020 = modelo.predict(x_prev)


# In[ ]:


p2020.value_counts()


# In[ ]:


df2020_final = df2020


# In[ ]:


df2020_final['dano_na_plantacao_predicted'] =p2020


# In[ ]:


df2020_final.head()


# ## Gerando Arquivo .CSV

# In[ ]:


df2020_final.to_csv("Previsão_2020.csv",header = True)

