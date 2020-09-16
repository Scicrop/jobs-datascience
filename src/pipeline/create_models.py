from lib import utils
from lib import data_processing
from lib import model
import sys
import logging
import os

if __name__ == "__main__":
  try:
    path = sys.argv[1]
  except Exception as e:
    raise ValueError('Excpected Path Argument')

  logging.basicConfig(level=logging.INFO,
                  format=f'# {__file__} - %(levelname)s: %(message)s ')

  base = os.path.basename(path).split('.')[0]

  # Load Training Data
  logging.info(f'Loading {base} for training')
  rd = data_processing.read_data(path)
  drop_columns = ['dano_na_plantacao_binario', 'dano_na_plantacao',
                  'Identificador_Agricultor']
  X_train = rd.drop(columns = drop_columns)
  y_train = rd['dano_na_plantacao_binario']
  # Create and Fit Binary Classifier Model
  binary_clf = model.BinaryClassifier()
  binary_clf.fit(X_train, y_train)
  binary_clf.dump()

  print('Performing k-fold prediction for binary classes')
  binary_train_predictions = binary_clf.k_fold_prediction(X_train, y_train)

  # Create and Fit Binary Classifier Model
  X_train['binary_predictions'] = binary_train_predictions
  y_train = rd['dano_na_plantacao']

  multi_clf = model.MultiClassifier()
  multi_clf.fit(X_train, y_train)
  multi_clf.dump()
