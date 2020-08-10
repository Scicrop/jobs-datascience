import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from imblearn.over_sampling import SMOTE

def impute_vals(df, target, strategy = 'median'):
    imp_median = SimpleImputer(missing_values=np.nan, strategy=strategy)
    vals = imp_median.fit_transform(df[target].values.reshape(-1, 1))
    
    df[target] = vals
    return df

def create_attr(df):
    ## 'Tipo_de_Solo_Cultivo'
    df.loc[((df.Tipo_de_Solo == 0) & (df.Tipo_de_Cultivo == 0)), 'Tipo_de_Solo_Cultivo'] = 0
    df.loc[((df.Tipo_de_Solo == 0) & (df.Tipo_de_Cultivo == 1)), 'Tipo_de_Solo_Cultivo'] = 1
    df.loc[((df.Tipo_de_Solo == 1) & (df.Tipo_de_Cultivo == 0)), 'Tipo_de_Solo_Cultivo'] = 2
    df.loc[((df.Tipo_de_Solo == 1) & (df.Tipo_de_Cultivo == 1)), 'Tipo_de_Solo_Cultivo'] = 3
    
    ## 'Dose_Total'
    df['Dose_Total'] = df['Doses_Semana']*df['Semanas_Utilizando']
    return df

def drop_feats(df, targets):
    return df.drop(targets, axis = 1)

def scaler(df):
    std_scaler = StandardScaler()
    df_scaled = std_scaler.fit_transform(df)
    return df_scaled

def ohe(df):
    ohe_enc = OneHotEncoder()
    df_encoded = ohe_enc.fit_transform(df).toarray()
    return df_encoded

def full_transform(df_in, train_set=True, SMOTE=False):
    ## Imputing
    df = impute_vals(df_in, 'Semanas_Utilizando')
    
    ## Criando atributos
    df = create_attr(df)
    
    ## Dropando atributos
    feats_to_drop = ['Doses_Semana', 'Tipo_de_Cultivo', 'Tipo_de_Solo', 'Temporada']
    df = drop_feats(df, feats_to_drop)
    
    ## Separando atributos numéricos e categóricos
    num_feats = ['Estimativa_de_Insetos', 'Semanas_Utilizando', 'Semanas_Sem_Uso', 'Dose_Total']
    cat_feats = ['Categoria_Pesticida', 'Tipo_de_Solo_Cultivo']
    id_feat = ['Identificador_Agricultor']
    if train_set:
        label_feat = ['dano_na_plantacao']
        df_label = df[label_feat]

    df_num = df[num_feats]
    df_cat = df[cat_feats]
    df_id = df[id_feat]
    
    ## Aplicando Scaling e Encoding nos atributos numéricos e categóricos, respectivamente
    df_num = scaler(df_num)
    df_cat = ohe(df_cat)
    
    ## Condensando em uma DataFrame
    df_out = np.concatenate([df_num, df_cat], axis = 1)
    df_out = pd.DataFrame(df_out, columns = None)
    
    cat_feats = ['Categoria_Pesticida_1',
             'Categoria_Pesticida_2',
             'Categoria_Pesticida_3',
             'Tipo_de_Solo_Cultivo_0',
             'Tipo_de_Solo_Cultivo_1',
             'Tipo_de_Solo_Cultivo_2',
             'Tipo_de_Solo_Cultivo_3',
            ]

    if train_set:
        cols = num_feats+cat_feats+label_feat
        df_out = pd.concat([df_out, df_label], axis = 1)
    else:
        cols = num_feats+cat_feats
    
    df_out.columns = cols
    df_out.index = df_id.iloc[:, 0]
    
    if SMOTE:
        X, y = df_out.drop('dano_na_plantacao', axis = 1), df_out['dano_na_plantacao']
        smote = SMOTE(sampling_strategy='not majority')
        X_sm, y_sm = smote.fit_sample(X, y)
        df_out = pd.concat([X_sm, y_sm], axis = 1)
    
    return df_out