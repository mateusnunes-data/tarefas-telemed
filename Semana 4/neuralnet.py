import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

'''
[X] Carregue o arquivo train.csv usado na penúltima atividade.
[X] Divida as linhas em dois conjuntos: treino (80%) e teste (20%).
[X] Faça o pré-processamento usando apenas estatísticas do conjunto de treino. Por exemplo, se você preencheu valores nulos de alguma coluna com a média,
    use a média do treino, e não a do conjunto completo. O mesmo vale para a média e o desvio padrão do z-score. Caso contrário,
    informações do teste vazam para o treino e seus resultados ficam contaminados.
[X] Concatene as colunas tratadas e use-as como entrada (input) da rede.
[X] Crie uma classe que herde de nn.Module. Essa classe será a sua rede neural.
[X] A rede deve ter pelo menos 1 ou 2 camadas, cada uma no formato Linear -> ReLU (função de ativação). Exemplo de arquitetura:
    Linear -> ReLU -> Linear -> ReLU -> Linear (saída).
[X] A saída da rede será o preço da casa. Para obter o valor real, reverta o z-score e depois o ln.
[X] Treine o modelo usando a função de perda MSE e o otimizador Adam.
[X] Treine a rede somente com os dados de treino e avalie-a somente com os dados de teste.
[ ] Faça os testes e ajustes que julgar necessários para melhorar os resultados e experimentar.
[X] Valide os resultados calculando métricas no conjunto de teste: RMSE, MAE e R²
'''

df = pd.read_csv('Semana 4/dados/train.csv')
df = df.drop(columns=['Utilities', 'Street', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC', 'Exterior1st', 'SaleType'])

# Separação dos dois datasets, treino e teste
df_treino = df.sample(frac=0.8, random_state=42)
df_teste = df.drop(df_treino.index)

# Colunas numéricas
colunas_numericas = df.select_dtypes(include=[np.number]).columns.drop(['Id', 'MSSubClass', 'GarageYrBlt', 'SalePrice'])

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

excluir = list(colunas_numericas) + list(mapeamentos.keys()) + ['Id', 'GarageYrBlt', 'SalePrice']
cols_nominais = [c for c in df.columns if c not in excluir]

# Numéricas: nulos e z-score com estatísticas do treino
medias_fill = df_treino[colunas_numericas].mean()
num_treino = df_treino[colunas_numericas].fillna(medias_fill)
num_teste = df_teste[colunas_numericas].fillna(medias_fill)

media_num = num_treino.mean()
std_num = num_treino.std()
X_num_treino = (num_treino - media_num) / std_num
X_num_teste = (num_teste - media_num) / std_num

# Ordinais
def tratar_ordinais(d):
    saida = pd.DataFrame(index=d.index)
    for coluna, mapa in mapeamentos.items():
        saida[coluna] = d[coluna].fillna("None").map(mapa)
    return saida

X_ord_treino = tratar_ordinais(df_treino)
X_ord_teste = tratar_ordinais(df_teste)

# Nominais: one-hot com as colunas do treino
def tratar_nominais(d):
    n = d[cols_nominais].fillna("None").copy()
    n['MSSubClass'] = n['MSSubClass'].astype(str)
    return n

X_cat_treino = pd.get_dummies(tratar_nominais(df_treino), dtype=int)
X_cat_teste = pd.get_dummies(tratar_nominais(df_teste), dtype=int)
X_cat_teste = X_cat_teste.reindex(columns=X_cat_treino.columns, fill_value=0)

# Alvo: ln + z-score com estatísticas do treino
ln_treino = np.log(df_treino['SalePrice'])
ln_teste = np.log(df_teste['SalePrice'])

media_y = ln_treino.mean()
std_y = ln_treino.std()
y_treino = (ln_treino - media_y) / std_y
y_teste = (ln_teste - media_y) / std_y

# Concatenação lado a lado
X_treino = pd.concat([X_num_treino, X_ord_treino, X_cat_treino], axis=1)
X_teste = pd.concat([X_num_teste, X_ord_teste, X_cat_teste], axis=1)

# Dataframe -> tensores para usar no PyTorch
X_treino_t = torch.tensor(X_treino.values, dtype=torch.float32)
X_teste_t = torch.tensor(X_teste.values, dtype=torch.float32)

y_treino_t = torch.tensor(y_treino.values, dtype=torch.float32).reshape(-1, 1)
y_teste_t = torch.tensor(y_teste.values, dtype=torch.float32).reshape(-1, 1)

# Recomenda-se guardar o número de entradas em variável para a primeira camada, (e failsafe em caso de mudança no futuro)
entradas = X_treino_t.shape[1]


class RedeNeural(nn.Module):
    def __init__(self, entradas):
        super().__init__()
        self.camada_1 = nn.Linear(entradas, 128)
        self.camada_2 = nn.Linear(128, 64)
        self.saida = nn.Linear(64, 1)

    def forward(self, output):
        output = torch.relu(self.camada_1(output))
        output = torch.relu(self.camada_2(output))
        return self.saida(output)


# Dados em lotes
dataset_treino = TensorDataset(X_treino_t, y_treino_t)
loader_treino = DataLoader(dataset_treino, batch_size=64, shuffle=True)

# Modelo, perda e otimizador
modelo = RedeNeural(entradas)
criterio = nn.MSELoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)


def treinar_uma_epoca(modelo, loader, criterio, otimizador):
    modelo.train()
    soma_loss = 0.0
    for xb, yb in loader:
        otimizador.zero_grad()
        loss = criterio(modelo(xb), yb)
        loss.backward()
        otimizador.step()
        soma_loss += loss.item() * len(xb)
    return soma_loss / len(loader.dataset)


def avaliar(modelo, X, y, criterio):
    modelo.eval()
    with torch.no_grad():
        return criterio(modelo(X), y).item()


# Reverte o alvo: z-score -> ln -> preço real
def reverter(y_z):
    ln = y_z * std_y + media_y   # desfaz o z-score (estatísticas do treino)
    return np.exp(ln)            # desfaz o ln


n_epocas = 100
hist_treino = []
hist_teste = []

for epoca in range(n_epocas):
    loss_treino = treinar_uma_epoca(modelo, loader_treino, criterio, otimizador)
    loss_teste = avaliar(modelo, X_teste_t, y_teste_t, criterio)

    hist_treino.append(loss_treino)
    hist_teste.append(loss_teste)

    if (epoca + 1) % 10 == 0:
        print(f"Época {epoca+1:3d} | treino: {loss_treino:.4f} | teste: {loss_teste:.4f}")


# Previsões em preço real
modelo.eval()
with torch.no_grad():
    pred_teste_z = modelo(X_teste_t).numpy().flatten()

preco_previsto = reverter(pred_teste_z)
preco_real = df_teste['SalePrice'].values

print("\nPrevisto:", preco_previsto[:5].round(0))
print("Real:    ", preco_real[:5])

# Métricas no conjunto de teste (em dólares)
erro = preco_previsto - preco_real

rmse = np.sqrt(np.mean(erro ** 2))
mae = np.mean(np.abs(erro))
r2 = 1 - np.sum(erro ** 2) / np.sum((preco_real - preco_real.mean()) ** 2)

print(f"\nRMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}")
print(f"R²:   {r2:.4f}")