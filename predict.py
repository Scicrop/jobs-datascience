# Libs
import sys
import os
from datetime import datetime
import pandas as pd
import cloudpickle


variables = {'model_name':sys.argv[1],'predict_file': sys.argv[2],'results_file':sys.argv[3]}


# Constantes

META_DATA_PATH = os.path.join('data','raw', 'dicionario_tipos.pkl')
MODEL_PATH = os.path.join('models',variables['model_name'])
RESULTS_PATH = os.path.join('data', 'results', variables['results_file'])
TEST_FILE_PATH = os.path.join('data','raw',variables['predict_file'])
LOGS_PATH = os.path.join('models')
try: 
    print('Lendo dicionário de dados...')
    with open(META_DATA_PATH,'rb') as f:
        dicionario_tipo = cloudpickle.load(f)
except:
    print('Nenhum dicionário de tipos encontrado')
    exit()
# Modelo
try:
    print('Lendo Modelo...')
    with open(MODEL_PATH, 'rb') as model_file:
        model = cloudpickle.load(model_file)
        print('Modelo Lido')
except:
    print('O modelo especificado não existe')
    exit()

# Data
try:
    print('Lendo arquivo de predição...')
    df_predict = pd.read_csv(TEST_FILE_PATH, index_col=0, dtype=dicionario_tipo)
    print('Dados Lidos')
except:
    print('O arquivo de predição especificado não existe')
    exit()

try:
    print('Realizando previsões...')
    results = model.predict_proba(df_predict)
    print('Previsões realizadas')
    results = results*1000
    columns_predict = ['Score_Saudável','Score_Danos_Outros_Motivos','Score_Danos_Pesticida']
    df_results = pd.DataFrame(results, columns=columns_predict)
    df_results = df_results.round(2)
    df_results['Identificador_Agricultor'] = df_predict['Identificador_Agricultor'].values
    df_results.to_csv(RESULTS_PATH, index=False)
    print('Resultados Salvos')
except:
    print('Falhou :(')
    exit()

# Logs
try:
    with open(os.path.join(LOGS_PATH,'logs.txt'), "a") as log:
        log.write(f"Data-Hora: {datetime.now().strftime('%Y-%m-%d %H:%M')} -- Número de Registros: {df_results.shape[0]} -- Modelo Utilizado: {variables['model_name']} -- Arquivo de Predição: {variables['predict_file']} -- Arquivo Salvo: {variables['results_file']}\n")
except:
    with open(os.path.join(LOGS_PATH,'logs.txt'), 'w') as log:
        log.write(f"Data-Hora: {datetime.now().strftime('%Y-%m-%d %H:%M')} -- Número de Registros: {df_results.shape[0]} -- Modelo Utilizado: {variables['model_name']} -- Arquivo de Predição: {variables['predict_file']} -- Arquivo Salvo: {variables['results_file']}\n")
