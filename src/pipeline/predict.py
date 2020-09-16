
from lib import model, data_processing
import logging
import sys
import pandas as pd
import os

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if  __name__ == "__main__":
  try:
    path = sys.argv[1]
  except Exception as e:
    raise ValueError('Excpected Path Argument')

  base = os.path.basename(path).split('.')[0]


  logging.basicConfig(level=logging.INFO,
                format=f'# {__file__} - %(levelname)s: %(message)s ')

  bclf = model.BinaryClassifier()
  bclf.load()
  logging.info('Binary Classfier loaded with success')

  mclf = model.MultiClassifier()
  mclf.load()
  logging.info('Multiclass Classfier loaded with success')

  logging.info(f'Loading {base} for prediction')
  df_original = pd.read_csv(path, index_col = 0)
  df = data_processing.read_data(path)
  id_ag = df['Identificador_Agricultor']

  X = df.drop(columns = ['Identificador_Agricultor'])

  binary_pred = bclf.predict(X)
  X['binary_predictions'] = binary_pred
  multiclass_pred = mclf.predict(X)

  df_pred = pd.concat([df_original, multiclass_pred], axis = 1)

  logging.info(f'Saving prediction in results/{base}-prediction.csv')
  create_dir('results')
  df_pred.to_csv(f'results/{base}-prediction.csv', index = False)


