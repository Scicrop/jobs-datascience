import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, power_transform, OneHotEncoder
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import random
import os
from joblib import load
import sys
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED']=str(42)

class Classificador:

    def __init__(self):
        try:
            self.nb = load(os.path.join(sys.path[0],"models","nb.joblib"))
            self.svc = load(os.path.join(sys.path[0],"models","svc.joblib"))
            self.rf = load(os.path.join(sys.path[0],"models","rf.joblib"))
            self.mlp = load(os.path.join(sys.path[0],"models","mlp.joblib"))
        except:
            self.nb = load(os.path.join(sys.path[0],"Classificator","models","nb.joblib"))
            self.svc = load(os.path.join(sys.path[0],"Classificator","models","svc.joblib"))
            self.rf = load(os.path.join(sys.path[0],"Classificator","models","rf.joblib"))
            self.mlp = load(os.path.join(sys.path[0],"Classificator","models","mlp.joblib"))
        self.model_dict = dict(zip(['nb', 'svc', "rf", "perc"],
                      [self.nb, self.svc, self.rf, self.mlp]))
        self.model_weights = {'nb': 0.01871953120627269,
        'perc': 0.7784968342738109,'rf': 0.1190089322158962,
        'svc': 0.03447975258835745}

    def __fillna_Semanas_Utilizando(self, df2):
        df2[df2['Doses_Semana'] == 0] = df2[df2['Doses_Semana'] == 0].fillna(0)
        df2['Semanas_Utilizando'] = df2.groupby(['Estimativa_de_Insetos'])['Semanas_Utilizando'].apply(lambda x: x.fillna(int(x.mean())))
        return df2

    def __clean_doses_semana(self, df2):
        df2['Doses_Semana'] = power_transform(np.array(df2['Doses_Semana']+3).reshape(-1,1), method = 'box-cox')
        return df2

    def __clean_semanas_utilizando(self, df2):
        df2['Semanas_Utilizando'] = power_transform(np.array(df2['Semanas_Utilizando']+1).reshape(-1,1), method = 'box-cox')
        return df2

    def __clean_semanas_Sem_Uso(self, df2):
        df2['Semanas_Sem_Uso'] = df2['Semanas_Sem_Uso']**(1/2)
        return df2

    def __clean_estimativa_de_Insetos(self, df2):
        df2['Estimativa_de_Insetos'] = df2['Estimativa_de_Insetos']**(1/3)
        return df2


    def __clean_categoria_pesticida(self, df2):
        cpohe = OneHotEncoder()
        encoded = cpohe.fit_transform(np.array(df2['Categoria_Pesticida']).reshape(-1,1)).todense()
        df2 = pd.concat([df2, pd.DataFrame(encoded)], axis=1).drop(['Categoria_Pesticida'], axis=1)
        return df2


    def __clean_temporada(self, df2):
        tempohe = OneHotEncoder()
        encoded = tempohe.fit_transform(np.array(df2['Temporada']).reshape(-1,1)).todense()
        df2 = pd.concat([df2, pd.DataFrame(encoded)], axis=1).drop(['Temporada'], axis=1)
        return df2

    def preprocessamento(self, df):
        df2 = df.copy()
        df2 = self.__fillna_Semanas_Utilizando(df2)
        df2 = self.__clean_doses_semana(df2)
        df2 = self.__clean_semanas_utilizando(df2)
        df2 = self.__clean_semanas_Sem_Uso(df2)
        df2 = self.__clean_estimativa_de_Insetos(df2)
        df2 = self.__clean_categoria_pesticida(df2)
        df2 = self.__clean_temporada(df2)
        return df2

    def classify(self, df2, filename="safra_classificada"):
        df = df2.copy()
        pred = 0

        for model_name, model in self.model_dict.items():
            print(model_name)
            pred += self.model_dict[model_name].predict_proba(df2.drop(["Identificador_Agricultor"], axis=1)) * self.model_weights[model_name]

        pred = (pred)/sum(self.model_weights.values())

        df['dano_na_plantacao'] = np.argmax(pred, axis=1)
        df.to_csv(os.path.join(sys.path[0],"out", filename+".csv"), index = False)


if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Insira o endereço do arquivo")
    else:
        df = pd.read_csv(sys.argv[1]).reset_index(drop=True)

        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)

        clf = Classificador()
        df = clf.preprocessamento(df)
        clf.classify(df, "TESTEEEEEE_ARQUIVO_FINAL")
