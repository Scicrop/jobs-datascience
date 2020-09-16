try:
  from . import data_processing
except:
  import data_processing
import pickle
import os
import sys
import numpy as np
from sklearn.model_selection import KFold

import xgboost as xgb
import lightgbm as lgb
import pandas as pd

import joblib

root = os.path.abspath(os.path.join(__file__ ,"../../.."))

class BinaryClassifier():
  """
    Binary Classifier Model
  """
  def __init__(self):
    # Load Best Parameters
    bi_config = os.path.join(root, 'models','binary_classfier_parameters.pkl')
    with open(bi_config,'rb') as file:
      model_paramters = pickle.load(file)
    # Classifier Paramters
    self.binary_clf_parameters = model_paramters['reg_paramters']
    # Decision Threshold
    self.threshold = model_paramters['threshold']
    # Save-Load Model Path
    self.model_path = os.path.join(root, 'models/binary_classifier.xgb')

  def fit(self,  X, y):
    """
    fit binary classifier model

    :type X: pandas DataFrame
    :param X: Training features

    :type y: pandas Series
    :param y: Training labels
    """
    X_train, _, y_train, _ = data_processing.train_test_sample(X, y, 0.0,
                                                        upsample_type = 'SMOTE',
                                                        over_sampling = 0.5,
                                                        under_sampling = 0.8)
    print('Training Binary Classifier Model')
    self.clf = xgb.XGBClassifier( **self.binary_clf_parameters,
                                  objective="binary:logistic",
                                  random_state=42)
    self.clf.fit(X_train, y_train)

  def dump(self, path = None):
    """
    export model to file

    :type path: string
    :param path: path to dump model, defaults to None
    """
    if path is None:
      path = self.model_path

    joblib.dump(self.clf, path)
    print(f'Binary Classifier saved at {path}')

  def load(self, path = None):
    """
    laod model from file

    :type path: string
    :param path: path to dump model, defaults to None
    """
    if path is None:
      path = self.model_path

    self.clf = joblib.load(path)
    print(f'Model Loaded from {path}')

  def predict(self, X):
    """
    predict on data

    :type X: pandas DataFrame
    :param X: data to predict

    :return: predictions
    :rtype: pandas Series
    """
    proba = self.clf.predict_proba(X)
    return (proba[:,1] > self.threshold).astype(int)

  def k_fold_prediction(self, X, y, n_splits = 5):
    """
    performs train-predict k fold

    :type X: pandas DataFrame
    :param X: Training Features

    :type y: pandas Series
    :param y: Training labels

    :type n_splits: int
    :param n_splits: number of k splits, defaults to 5

    :return: k fold predictions
    :rtype: pandas Series
    """
    cv_df = pd.concat([X, y], axis = 1)
    predictions = pd.Series(index = cv_df.index, dtype=np.float64)

    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(cv_df)

    for train_index, test_index in kf.split(cv_df):

      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train = y.iloc[train_index]

      X_train, y_train = data_processing.syntetic_sampling(X_train, y_train,
                                                     over_sampling = 0.5,
                                                     under_sampling = 0.8)

      # ML Model
      clf = xgb.XGBClassifier( **self.binary_clf_parameters,
                                  objective="binary:logistic",
                                  random_state=42)
      clf.fit(X_train, y_train, verbose=False)

      # Predict Probability
      proba = clf.predict_proba(X_test)
      predictions.iloc[test_index] = proba[:,1]

    return (predictions > self.threshold).astype(int)


class MultiClassifier():
  """
  Multiclass Classifier Model
  """
  def __init__(self,):
    multi_config = os.path.join(root, 'models',
                                'multiclass_classfier_parameters.pkl')
    with open(multi_config,'rb') as file:
      self.multi_clf_parameters = pickle.load(file)
    self.model_path = os.path.join(root, 'models/multiclass_classifier.lgb')

  def fit(self, X, y):
    """
    fit multiclass classifier model

    :type X: pandas DataFrame
    :param X: Training features

    :type y: pandas Series
    :param y: Training labels
    """

    X_train, _, y_train, _= data_processing.train_test_sample(X, y, 0)

    X_train_p = X_train[X_train['binary_predictions'] == 1]\
                        .drop(columns = ['binary_predictions'])
    y_train_p = y_train[X_train['binary_predictions'] == 1]

    w1 = self.multi_clf_parameters['w1']
    w2 = self.multi_clf_parameters['w2']
    w3 = self.multi_clf_parameters['w3']

    print('Training Binary Classifier Model')
    self.mclf = lgb.LGBMClassifier(class_weight = {0:w1, 1:w2, 2:w3},
                                   silent=True)
    self.mclf.fit(X_train_p, y_train_p)

  def dump(self, path = None):
    """
    :type path: string
    :param path: path to dump model, defaults to None
    """
    if path is None:
      path = self.model_path

    joblib.dump(self.mclf, path)
    print(f'Multiclass Classifier saved at {path}')

  def load(self, path = None):
    """
    laod model from file

    :type path: string
    :param path: path to dump model, defaults to None
    """
    if path is None:
      path = self.model_path

    self.mclf = joblib.load( path)
    print(f'Model Loaded from {path}')

  def predict(self, X):
    """
    predict on data

    :type X: pandas DataFrame
    :param X: data to predict

    :return: predictions
    :rtype: pandas Series
    """
    y_predictions = pd.Series(index = X.index, dtype = np.float64)

    mask = X['binary_predictions'] == 1
    X_p = X[mask].drop(columns = ['binary_predictions'])

    # Predict on positive Class
    y_predictions[ mask] = self.mclf.predict(X_p)
    # Negative class remains negative
    y_predictions[~mask] = 0

    return y_predictions
