import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.svm import SVC

def class_size(y,label):
    return len(y[y == label])

def resample(X, y):
    print('\nOriginal')
    for i in range( len(y.unique()) ): print( str(i) + ': ' + str(class_size(y,i)) )
    # under = RandomUnderSampler( sampling_strategy = 'majority' )
    under = RepeatedEditedNearestNeighbours( sampling_strategy = 'majority' )
    X_res, y_res = under.fit_resample(X, np.ravel(y, order='C'))
    over = ADASYN( sampling_strategy={0:class_size(y_res,0), 1:int(class_size(y_res,0)*0.7), 2:int(class_size(y_res,0)*0.7)} ) 
    # over = ADASYN( sampling_strategy='not majority' ) 
    X_res, y_res = over.fit_resample(X_res, np.ravel(y_res, order='C'))
    print('\nResampled')
    for i in range( len(y.unique()) ): print( str(i) + ': ' + str(class_size(y_res,i)) )
    return X_res, y_res

def print_metrics(pred, y_test):
    precision, recall, f_score, _ = metrics.precision_recall_fscore_support(y_test, pred, average='macro')
    print("precision: %.2f" % precision)
    print("recall: %.2f" % recall)
    print("f-score: %.2f" % f_score)
    print('confusion matrix: ')
    print(classification_report(y_test, pred))
    print( metrics.confusion_matrix(y_test, pred))

def rank_features(X, clf):
    feature_importances = clf.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    feature_names = X.columns.tolist()
    plt.bar(feature_names, feature_importances)
    plt.show()

def search_parameters(clf, distributions, X,y):
    grid = RandomizedSearchCV(clf, distributions)
    search = grid.fit(X,y)
    print(search.best_params_)

# remove rows with null data
def remove_null(data):
    for column in data.columns:
        data = data.dropna(subset=[ column ])
    return data

# expand categorical attributes into columns
def transform_categorical(df):
    data = df
    data['Cultivo_1'] = data['Tipo_de_Cultivo']
    data['Cultivo_2'] = 0
    data.loc[data['Cultivo_1'] == 0, 'Cultivo_2'] = 1
    data['Solo_1'] = data['Tipo_de_Solo']
    data['Solo_2'] = 0
    data.loc[data['Solo_1'] == 0, 'Solo_2'] = 1
    data['Pesticida_1']=0
    data['Pesticida_2']=0
    data['Pesticida_3']=0
    data.loc[data['Categoria_Pesticida'] == 1, 'Pesticida_1'] = 1
    data.loc[data['Categoria_Pesticida'] == 2, 'Pesticida_2'] = 1
    data.loc[data['Categoria_Pesticida'] == 3, 'Pesticida_3'] = 1
    data['Temporada_1']=0
    data['Temporada_2']=0
    data['Temporada_3']=0
    data.loc[data['Temporada'] == 1, 'Temporada_1'] = 1
    data.loc[data['Temporada'] == 2, 'Temporada_2'] = 1
    data.loc[data['Temporada'] == 3, 'Temporada_3'] = 1
    data = data.drop( ['Unnamed: 0', 'Identificador_Agricultor', 'Tipo_de_Cultivo', 'Tipo_de_Solo', 'Categoria_Pesticida', 'Temporada'], axis=1 )
    return data

# normalize sample values between 0-1
def rescale(data):
    min_max = preprocessing.MinMaxScaler()
    scaled = min_max.fit_transform(X)
    return pd.DataFrame(scaled)

data = pd.read_csv('Safra_2018-2019.csv')
data = remove_null(data)
data = transform_categorical(data)

# extract sample and target
y = data['dano_na_plantacao']
X = data.drop(['dano_na_plantacao'], axis=1)

# split data into train(66%) and test(33%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_train, y_train = resample(X_train, y_train)

# classifier
clf = RandomForestClassifier(class_weight='balanced', n_estimators=50, max_features='log2', max_depth=30, bootstrap=False)

clf.fit(X_train, y_train)

# prediction
data_2020 = pd.read_csv( 'Safra_2020.csv' )
data_2020 = remove_null(data_2020)
data_2020 = transform_categorical(data_2020)

pred = clf.predict(data_2020)

result = pd.read_csv( 'Safra_2020.csv' )
result = remove_null(result)
result['dano_na_plantacao'] = pred
result.to_csv( 'Predict_Safra_2020.csv' )