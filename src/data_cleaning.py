

def remover_duplicatas(df):
    #Verificando duplicatas nas linhas
    print('Removendo...')
    df = df.drop_duplicates()
    #Verificando duplicatas colunas
    df_T = df.T
    print(f'Existem {df_T.duplicated().sum()} colunas duplicadas e {df.duplicated().sum()} linhas duplicadas')
    list_duplicated_columns = df_T[df_T.duplicated(keep=False)].index.tolist()
    df_T.drop_duplicates(inplace = True)
    print('Colunas duplicadas:')
    print(list_duplicated_columns)
    return  df_T.T, list_duplicated_columns

def remover_colunas_constantes(df):
    const_cols = []
    for i in df.columns:
        if len(df[i].unique()) == 1:
            df.drop(i, axis = 1, inplace= True)
            const_cols.append(i)
    return df, const_cols