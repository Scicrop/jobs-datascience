import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def class_size(y,label):
    return len(y[y == label])

def resample(X, y):
    print('\nOriginal')
    for i in range(3): print( i + ': ' + class_size(y,i) )

    # under = RandomUnderSampler(random_state=42, sampling_strategy = { 0:int(size_0/4), 1:size_1, 2:size_2 } )
    under = RepeatedEditedNearestNeighbours(sampling_strategy = 'majority')
    X_under, y_under = under.fit_resample(X, np.ravel(y, order='C'))

    # over = ADASYN(random_state=42, sampling_strategy={0:class_size(y_under,0), 1:class_size(y_under,0), 2:class_size(y_under,2)*10}) 
    over = ADASYN(random_state=42, sampling_strategy='not majority') 
    X_over, y_over = over.fit_resample(X_under, np.ravel(y_under, order='C'))

    print('\nResampled')
    for i in range(3): print( i + ': ' + class_size(y_over,i) )

    return X_over, y_over

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

data = pd.read_csv('Safra_2018-2019.csv')

# remove rows with null data
data = data.dropna(subset=['Semanas_Utilizando'])

# extract sample and target
y = data['dano_na_plantacao']
X = data.drop(['Unnamed: 0', 'Identificador_Agricultor', 'dano_na_plantacao'], axis=1)

# split data into train(66%) and test(33%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_resampled, y_resampled = resample(X_train, y_train)
forest = RandomForestClassifier(random_state=42, n_estimators=100)
forest.fit(X_resampled, y_resampled)
# rank_features(X, forest)
pred = forest.predict(X_test)
print_metrics(pred, y_test)



