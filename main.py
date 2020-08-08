import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing

def class_size(y,label):
    return len(y[y == label])

def resample(X, y):
    print('\nOriginal')
    for i in range( len(y.unique()) ): print( str(i) + ': ' + str(class_size(y,i)) )

    # under = RandomUnderSampler( sampling_strategy = { 0:int(size_0/4), 1:size_1, 2:size_2 } )
    under = RepeatedEditedNearestNeighbours( sampling_strategy = 'majority' )
    X_under, y_under = under.fit_resample(X, np.ravel(y, order='C'))

    # over = ADASYN( sampling_strategy={0:class_size(y_under,0), 1:class_size(y_under,0), 2:class_size(y_under,2)*4} ) 
    # over = ADASYN( sampling_strategy='not majority' ) 
    over = SMOTEENN( sampling_strategy=0.7 ) 
    X_over, y_over = over.fit_resample(X_under, np.ravel(y_under, order='C'))

    print('\nResampled')
    for i in range( len(y.unique()) ): print( str(i) + ': ' + str(class_size(y_over,i)) )

    return X_over, y_over
    # return X_under, y_under

def print_metrics(pred, y_test):
    precision, recall, f_score, _ = metrics.precision_recall_fscore_support(y_test, pred, average='macro')
    print("precision: %.2f" % precision)
    print("recall: %.2f" % recall)
    print("f-score: %.2f" % f_score)
    print('confusion matrix: ')
    print( metrics.confusion_matrix(y_test, pred))

def rank_features(X, clf):
    feature_importances = clf.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    feature_names = X.columns.tolist()
    plt.bar(feature_names, feature_importances)
    plt.show()

def search_parameters(X,y):
    distributions = {
        'n_estimators':[50,100,300,500,1000],
        'bootstrap':[True,False],
        'max_features':['sqrt','log2', None],
        'max_depth':[10,30,None]
    }
    clf = RandomizedSearchCV(RandomForestClassifier(), distributions)
    search = clf.fit(X,y)
    print(search.best_params_)


data = pd.read_csv('Safra_2018-2019.csv')

# -----------------------------------Preprocessing------------------------------------------
# remove rows with null data
data = data.dropna(subset=['Semanas_Utilizando'])

# expand categorical attributes into columns
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

# merge target minority classes into one 
data['tem_dano']=0
data.loc[data['dano_na_plantacao'] == 0, 'tem_dano'] = 0 
data.loc[data['dano_na_plantacao'] == 1, 'tem_dano'] = 1
data.loc[data['dano_na_plantacao'] == 2, 'tem_dano'] = 1 

data['dano_outros']=0
data.loc[data['dano_na_plantacao'] == 1, 'dano_outros'] = 1 

data['dano_por_pesticida']=0
data.loc[data['dano_na_plantacao'] == 2, 'dano_por_pesticida'] = 1 


# extract sample and target
y = data['tem_dano']
X = data.drop(['Unnamed: 0', 'Identificador_Agricultor', 'dano_na_plantacao', 'Tipo_de_Cultivo', 'Tipo_de_Solo', 'Categoria_Pesticida', 'Temporada', 'tem_dano', 'dano_outros', 'dano_por_pesticida'], axis=1)

# normalize data
min_max = preprocessing.MinMaxScaler()
scaled = min_max.fit_transform(X)
X = pd.DataFrame(scaled)

# split data into train(66%) and test(33%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_train, y_train = resample(X_train, y_train)

# search_parameters(X_train, y_train)

forest = RandomForestClassifier(class_weight='balanced', n_estimators=50, max_features='log2', max_depth=30, bootstrap=False)
forest.fit(X_train, y_train)


pred = forest.predict(X_test)
print_metrics(pred, y_test)



