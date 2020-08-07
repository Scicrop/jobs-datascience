#funcao preencher por mediana. 
def preencher_missing_mediana(df):
    list_to_null = ['Semanas_Utilizando']
    df.loc[:,list_to_null] = df.loc[:,list_to_null] .fillna('median')
    return df