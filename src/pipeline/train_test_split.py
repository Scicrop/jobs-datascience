import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                    format=f'# {__file__} - %(levelname)s: %(message)s ')

    logging.info('splitting dataset into training and validation subsets')

    df = pd.read_csv('data/Safra_2018-2019.csv',
                    index_col = 0)

    save_path = 'data/train-validation'
    create_dir(save_path)

    validation_size = 0.3
    logging.info(f'train-test size: {(1-validation_size)*100} %')
    logging.info(f'validation size: {(validation_size)*100} %')

    label = ['dano_na_plantacao']
    features = list(df.drop(columns = label).columns)

    X_train, X_test, y_train, y_test = train_test_split(
                                        df[features], df[label],
                                        test_size=validation_size,
                                        random_state=33)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_validation = pd.concat([X_test, y_test], axis=1)

    gb_train = df_train.groupby(label).count().iloc[:,0] / df_train.shape[0]
    prop_train = gb_train.to_dict()

    gb_validation = df_validation.groupby(label).count().iloc[:,0] / df_validation.shape[0]
    prop_validation = gb_validation.to_dict()

    logging.info(f'training class proportion   {str(prop_train)}')
    logging.info(f'validation class proportion {str(prop_validation)}')

    df_train.to_csv(f'{save_path}/training-data.csv')
    df_validation.to_csv(f'{save_path}/validation-data.csv')
    logging.info(f'saving data on {save_path}/')
