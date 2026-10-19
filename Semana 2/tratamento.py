import pandas as pd
import numpy as np

'''
Acompanhamento pessoal de tarefas

[X] Faça o download do dataset e carregue-o usando a biblioteca Pandas;
[X] Remova as colunas: Utilities, Street, Condition2, RoofMatl, Heating, PoolQC, Exterior1st e SaleType.
[X] Trate os NA ("None") do dataset. No caso de ser um campo categorico trate isso como mais uma categoria, se for um campo númerico utilize a média da coluna.
[X] Calcule o z-score para todas as colunas numericas, não categoricas, e aplique essa transformação para todos os elementos da coluna.
[X] Transforme as colunas ordinais em números (Ex: {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}; )
[X] Calcule o one-hot encoding para as colunas categóricas.
[X] Para a coluna de SalePrice faça z-score(ln(...)) dos valores da coluna.
'''

# Carregamento do .csv
df = pd.read_csv('Semana 2/dados/train.csv')

# Remoção das colunas pedidas
df = df.drop(columns=['Utilities', 'Street', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC', 'Exterior1st', 'SaleType']) 

# Selecionei primeiro todas as colunas com dados numéricos, porém ainda não posso afirmar que todas são quantitativas
numericos = df.select_dtypes(include=[np.number]) 

# Removi colunas que não fazem sentido nas quantitativas, como "id", "MSSubClass" (numérica categórica) e "GarageYrBlt" (ao checar, vi que só há NA nas que não foram construídas)
numericos = numericos.drop(columns=['Id', 'MSSubClass', 'GarageYrBlt'])

# Criação do dataset de dados categóricos
categoricos = df.drop(columns=numericos.columns)

# Validação de quantos NA existem no dataframe antes da substituição
print(df.isna().sum().sum()) 

# Substituição dos NA (esse fiz usando loop e o z-score apenas com pandas)
for coluna in numericos:
    numericos[coluna] = numericos[coluna].fillna(numericos[coluna].mean())

for coluna in categoricos:
    categoricos[coluna] = categoricos[coluna].fillna("None")

# Validação de quantos NA existem no dataframe após a substituição
print(numericos.isna().sum().sum())
print(categoricos.isna().sum().sum())

# z-score = (df − mean()) / std()
numericos_zscore = (numericos - numericos.mean()) / numericos.std()

# Validação z-score
print(numericos_zscore.mean())
print(numericos_zscore.std())

# Escala compartilhada por várias colunas (e mostrada no exemplo)
escala_qualidade = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

# Colunas que compartilham a escala acima
colunas_qualitativas = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

# Mapping
mapeamentos = {coluna: escala_qualidade for coluna in colunas_qualitativas}

mapeamentos.update({
    'BsmtExposure':     {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    'BsmtFinType1':     {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    'BsmtFinType2':     {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    'GarageFinish':     {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    'PavedDrive':       {"N": 0, "P": 1, "Y": 2},
    'LotShape':         {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    'LandSlope':        {"Gtl": 0, "Mod": 1, "Sev": 2},
    'Functional':       {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    'CentralAir':       {"N": 0, "Y": 1},
})

# Criação do dataframe "ordinais" para armazenamento das colunas ⋆｡°✩ ordinais ⋆｡°✩
ordinais = pd.DataFrame()
for coluna, mapa in mapeamentos.items():
    ordinais[coluna] = categoricos[coluna].map(mapa)

# Validação ordinais/qualitativos
print(ordinais.isna().sum().sum())
print(ordinais.head())

# Categóricos - ordinais para calcular o one hot // PS: AO CHECAR O CÓDIGO FINAL COM A IA, FOI RECOMENDADO INSERIR " + ['Id', 'GarageYrBlt']", motivos:
# - Id: identificador, geraria 1460 colunas
# - GarageYrBlt: ~100 anos distintos (misturados com "None") gerariam ~100 colunas esparsas
nominais = categoricos.drop(columns=list(mapeamentos.keys()) + ['Id', 'GarageYrBlt'])

# Essa coluna em específico é nominal mas estava com dados em int64, apenas mudei o datatype para str
nominais['MSSubClass'] = nominais['MSSubClass'].astype(str)

# dtype=int para vir 0/1 em vez de True/False
one_hot = pd.get_dummies(nominais, dtype=int)

# Validação one hot
print(nominais.nunique().sum(), one_hot.shape[1])

# Parte do valor original (numericos), não do já padronizado: ln de valor negativo não existe
ln_preco = np.log(numericos['SalePrice'])
numericos_zscore['SalePrice'] = (ln_preco - ln_preco.mean()) / ln_preco.std()

# Validação z-score(ln())
print(numericos_zscore['SalePrice'].mean(), numericos_zscore['SalePrice'].std())

# Compilado final: numéricas (z-score) + ordinais (números) + one-hot
df_final = pd.concat([numericos_zscore, ordinais, one_hot], axis=1)

print(df_final.shape)
print(df_final.isna().sum().sum())
print(df_final.dtypes.value_counts())