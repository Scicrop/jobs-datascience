
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def naive_classifiers(df_train, df_test, target = None, method=None):

    """
    Essa função nos mostra a performance de classificadores naive, baseados em regras simples como:
    -prever a classe mais frequente,
    -prever de forma estratificada,
    - prever de forma uniforme e
    - prever a classe minoritária

    """
    if method == None:
        method = 'binary'
    else:
        method = method
    strategies = ['most_frequent', 'stratified', 'uniform', 'constant']
    metrics  = {}
    for s in strategies:
        if s =='constant':
            dclf = DummyClassifier(strategy = s, random_state = 0, constant ='1')
            dclf.fit(df_train.drop(target, axis =1), df_train[target])
            y_pred_train = dclf.predict(df_train)
            y_pred_test = dclf.predict(df_test.drop(target,axis=1))
            recall_test = recall_score(df_test[target], y_pred_test, average=method)
            precision_test = precision_score(df_test[target], y_pred_test, average=method)
            f1_test = f1_score(df_test[target], y_pred_test, average=method)
            recall_train = recall_score(df_train[target], y_pred_train, average=method)
            precision_train = precision_score(df_train[target], y_pred_train, average=method)
            f1_train = f1_score(df_train[target], y_pred_train, average=method)

            metrics[s] = {'recall_test':recall_test,
                          'recall_train':recall_train,
                          'precision_train':precision_train,
                          'precision_test': precision_test,
                          'f1_train':f1_train,
                          'f1_test': f1_test}
        else:
            dclf = DummyClassifier(strategy = s, random_state = 0)
            dclf.fit(df_train.drop(target, axis = 1), df_train[target])
            y_pred_train = dclf.predict(df_train)
            y_pred_test = dclf.predict(df_test.drop(target,axis=1))
            recall_test = recall_score(df_test[target], y_pred_test, average=method)
            precision_test = precision_score(df_test[target], y_pred_test, average=method)
            f1_test = f1_score(df_test[target], y_pred_test, average=method)
            recall_train = recall_score(df_train[target], y_pred_train, average=method)
            precision_train = precision_score(df_train[target], y_pred_train, average=method)
            f1_train = f1_score(df_train[target], y_pred_train, average=method)

            metrics[s] = {'recall_test':recall_test,
                        'recall_train':recall_train,
                        'precision_train':precision_train,
                        'precision_test': precision_test,
                        'f1_train':f1_train,
                        'f1_test': f1_test}
    metrics_df = pd.DataFrame.from_records(metrics)

    return metrics_df

