    # -*- coding: utf-8 -*-
    """
    Created on Fri Aug  7 13:50:02 2020
    
    @author: willian mayrink
    """
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    
    
    data= pd.read_csv('Safra_2018-2019.csv')
    dataprev=pd.read_csv('Safra_2020.csv')
    
    meanNan= (data.isna().sum()/len(data))*100  #Verificando a porcentagem de dados faltantes, como é 10% então aceitável.
    X= data.iloc[:,2:-1] #selecionando as variaveis independentes
    y=  data.iloc[:,-1]
    X_dataprev= dataprev.iloc[:,2:]
    
    #Fazendo OneHotEncoding de variaveis categoricas com mais de 2 caracteristicas
    X = pd.concat([X,pd.get_dummies(X['Categoria_Pesticida'], prefix='Categoria_Pesticida')],axis=1)
    X = pd.concat([X,pd.get_dummies(X['Temporada'], prefix='Temporada')],axis=1)
    X.drop(['Categoria_Pesticida','Temporada'],axis=1, inplace=True)
    
    #fazer o mesmo para os dados de previsao de 2020.
    X_dataprev= pd.concat([X_dataprev,pd.get_dummies(X_dataprev['Categoria_Pesticida'], prefix='Categoria_Pesticida')],axis=1)
    X_dataprev= pd.concat([X_dataprev,pd.get_dummies(X_dataprev['Temporada'], prefix='Temporada')],axis=1)
    X_dataprev.drop(['Categoria_Pesticida','Temporada'],axis=1, inplace=True)
    
    #Separar dados de treino e dados de teste.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    
    #tratando os dados faltantes:
    imputer=SimpleImputer()
    X_train= imputer.fit_transform(X_train)
    X_test= imputer.transform(X_test)
    
    
    #Com uma análise rapida é possível notar que os dados tem ordens de grandezas diferentes entre as variáveis independentes, por isso é necessário fazer uma normalização.
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #Selecionei 3 classificadores que podem testar bem os dados, considerando a previsao como sendo multiclasse
    
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
            
    #Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    nbclf = GaussianNB().fit(X_train, y_train)
    y_pred= nbclf.predict(X_test)
    print('===================== Naive Bayes =====================')
    
    print(classification_report(y_test, y_pred))
    print('Micro-averaged precision = {:.2f}'
          .format(precision_score(y_test, y_pred, average = 'micro')))
    print('Macro-averaged precision = {:.2f}'
          .format(precision_score(y_test, y_pred, average = 'macro')))
    
    
    #Random Forest
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = 100, max_features = 8,
                                random_state=0).fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    
    print('===================== Random Forest =====================')
    print(classification_report(y_test, y_pred))
    print('Micro-averaged precision = {:.2f}'
          .format(precision_score(y_test, y_pred, average = 'micro')))
    print('Macro-averaged precision = {:.2f}'
          .format(precision_score(y_test, y_pred, average = 'macro')))
    
        
    #Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    print('===================== Gradient Boosting =====================')
    
    GBclf = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2).fit(X_train, y_train)
    y_pred=  GBclf.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    print('Micro-averaged precision = {:.2f}'
          .format(precision_score(y_test, y_pred, average = 'micro')))
    print('Macro-averaged precision = {:.2f}'
          .format(precision_score(y_test, y_pred, average = 'macro')))
    
    #Dentre os valores aquele que deu resultados melhores foi o GradientBoosting, então usaremos esse classificador.
    X_dataprev= imputer.transform(X_dataprev)
    X_dataprev = sc.transform(X_dataprev)
    y_pred_2020= GBclf.predict(X_dataprev)

