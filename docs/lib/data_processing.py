
import pandas as pd
import logging

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold

import numpy as np

def read_data(path, remove_nans = True, apply_ohe = True):
  """
  read and transform data

  :type path: string
  :param path: path to csv file

  :type remove_nans: boolean
  :param remove_nans: remove nans from DataFrame

  :type apply_ohe: boolean
  :param apply_ohe: apply OneHotEncoder on categorical data

  :return: clean data
  :rtype: pandas DataFrame with transformations
  """
  df = pd.read_csv(path, index_col = 0 )

  categorical  = ['Tipo_de_Cultivo','Tipo_de_Solo', 'Categoria_Pesticida',
                  'Temporada']
  label = 'dano_na_plantacao'

  if label in df.columns:
    has_label = True
    # Group labels 1 and 2
    binary_label = 'dano_na_plantacao_binario'
    binary_y = pd.Series(df[label].map({0:0,1:1,2:1}), name = binary_label)
    # Split features and label
    y = df[label]
    df = df.drop(columns = [label])
  else:
    has_label = False

  id = df.iloc[:,0]
  X = df.iloc[:,1:]

  # Remove NaNs
  if remove_nans:
    nan_cols = X.columns[X.isna().any()]
    for col in nan_cols:
      print(f'Removing {X[col].isna().sum()} from {col} ')
      X = fill_missing_knn(X, col)

  if apply_ohe:
    print(f'Applying OneHotEncoder on categorical features')
    X = ohe(X, categorical)

  if has_label:
    df_clean = pd.concat([id, X, y, binary_y], axis = 1)
  else:
    df_clean = pd.concat([id, X], axis = 1)

  return df_clean

def syntetic_sampling(X, y, over_sampling, under_sampling):
  """
  Apply Synthetic Minority Oversampling Technique (SMOTE)
  to tn unbalanced class

  :type X: pandas DataFrame
  :param X: Training Features

  :type y: pandas Series
  :param y: Training Features

  :return: resampled data
  :rtype: tuple
  """

  over = SMOTE(sampling_strategy=over_sampling)
  under = RandomUnderSampler(sampling_strategy=under_sampling)
  steps = [('o', over), ('u', under)]
  pipeline = Pipeline(steps=steps)

  return pipeline.fit_resample(X, y)

def copy_upsample(X, y, over_sampling):
  """
  Apply upsampling on minority data

  :type X: pandas DataFrame
  :param X: Training Features

  :type y: pandas Series
  :param y: Training Features

  :return: resampled data
  :rtype: tuple
  """
  df = pd.concat([X, y], axis = 1)
  label_name = y.name

  df_minority = df[ y == 1]
  df_majority = df[ y == 0]

  n_sample = len(df_majority) * over_sampling
  df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=n_sample,
                                 random_state=33)

  # Combine majority class with upsampled minority class
  df_upsampled = pd.concat([df_majority, df_minority_upsampled])

  return df_upsampled.drop(columns = [label_name]), df_upsampled[label_name]

def train_test_sample(X, y, test_size, upsample_type = None,
                      over_sampling = None, under_sampling = None):
  """
  Splits into train and test samples and applies transformations
  to the train sample

  :type X: pandas DataFrame
  :param X: Training Features

  :type y: pandas Series
  :param y: Training Features

  :type test_size: float
  :param test_size: test size from split, defaults to None

  :type upsample_type: string - [None, 'SMOTE', 'SIMPLE' ]
  :param upsample_type: sampling method

  :type upsample_type: float
  :param upsample_type: oversample rate

  :type under_sampling: float
  :param under_sampling: undersample rate

  :return: train and test split
  :rtype: tuple
  """

  if test_size == 0:
    X_train, X_test, y_train, y_test = X, None, y, None
  else:
    X_train, X_test, y_train, y_test  =  train_test_split(X, y,
                                                        test_size = test_size)

  if upsample_type == 'SMOTE':
    print('SMOTE Upsample')
    X_train, y_train = syntetic_sampling( X_train, y_train, over_sampling,
                                          under_sampling )
  elif upsample_type == 'SIMPLE':
    print('Simple Upsample')
    X_train, y_train = copy_upsample( X_train, y_train, over_sampling )

  return X_train, X_test, y_train, y_test

def k_fold_prediction(X, y, n_splits, model, reg_params, fit_parameters, upsample_kwargs):
  """
  performs train-predict k fold

  :type X: pandas DataFrame
  :param X: Training Features

  :type y: pandas Series
  :param y: Training Labels

  :type n_splits: int
  :param n_splits: number of k splits

  :type model: object
  :param model: model with fit and predict methods

  :type reg_params: dictonary
  :param reg_params: model kwargs

  :type fit_parameters: dictonary
  :param fit_parameters: model fit kwargs

  :type upsample_kwargs: dictonary
  :param upsample_kwargs: syntetic_sampling kwargs

  :return: predictions
  :rtype: pandas Series
  """
  cv_df = pd.concat([X, y], axis = 1)
  predictions = pd.Series(index = cv_df.index, dtype=np.float64)

  kf = KFold(n_splits=n_splits)
  kf.get_n_splits(cv_df)

  for i, (train_index, test_index) in enumerate(kf.split(cv_df)):
    print(f'{i+1}/{kf.n_splits}: TRAIN: {len(train_index)} - TEST: {len(test_index)} ')

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train = y.iloc[train_index]

    X_train, y_train = syntetic_sampling(X_train, y_train, **upsample_kwargs)

    # ML Model
    print('Training Model ...')
    clf = model( **reg_params )
    clf.fit(X_train, y_train, verbose=False, **fit_parameters)

    print('Predicting ...\n')
    # Predict Probability
    proba = clf.predict_proba(X_test)[:,1]
    predictions.iloc[test_index] = proba

  print('Done!')
  return predictions

def fill_missing_knn(df, na_column, n_neighbors=10, algorithm='ball_tree'):
  """
  fill missing data using k Nearest Neighbors

  :type df: pandas DataFrame
  :param df: Data Frame

  :type na_column: list
  :param na_column: column with nans

  :type n_neighbors: int
  :param n_neighbors: number of neighbors

  :type algorithm: string
  :param algorithm: nearest neighbors algorithm

  :return: dataframe without nans
  :rtype: pandas DataFrame
  """
  # Split training features from missing data column
  prediction_df = df.loc[ df[na_column].isna(), : ].copy()
  training_df   = df.loc[~df[na_column].isna(), : ]

  nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)
  nbrs.fit(training_df.drop(columns = [na_column]))

  # Predict
  _, neighbors = nbrs.kneighbors(prediction_df.drop(columns = [na_column]))

  # column index
  col_idx = training_df.columns.get_loc(na_column)

  knn_input = []
  # For each sample we have n_neighbors
  for sample in range(neighbors.shape[0]):

    # Get mean for all neighbors
    sample_mean = np.mean([training_df.iloc[i, col_idx]
                           for i in neighbors[sample,:] ])
    knn_input.append(sample_mean)

  prediction_df.loc[:,na_column] = knn_input

  return pd.concat([training_df, prediction_df], axis = 0).sort_index().copy()

def ohe(df, columns, drop_first = True):
  """
  apply OneHotEncoder

  :type df: pandas DataFrame
  :param df: Data Frame

  :type columns: list
  :param columns: column with nans

  :type drop_first: boolean
  :param drop_first: drop first ohe columns

  :return: dataframe with ohe
  :rtype: pandas DataFrame
  """
  df_c = df.copy(deep = True)
  for col in columns:
    dummies = pd.get_dummies(df_c[col], prefix=col,
                 drop_first=drop_first)
    df_c = df_c.drop(columns = [col])
    df_c = pd.concat([df_c, dummies], axis = 1)

  return df_c
