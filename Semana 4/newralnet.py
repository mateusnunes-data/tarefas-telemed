import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizando dispositivo: {device}")
if device.type == 'cuda':
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

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
[X] Faça os testes e ajustes que julgar necessários para melhorar os resultados e experimentar.
[X] Valide os resultados calculando métricas no conjunto de teste: RMSE, MAE e R²
'''

df = pd.read_csv('tarefas-telemed/Semana 4/dados/train.csv')
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
    'LandSlope':        {"Sev": 0, "Mod": 1, "Gtl": 2},
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


class Bloco(nn.Module):
    # Linear -> Norm -> ReLU -> Dropout, com soma residual opcional.
    def __init__(self, d_in, d_out, norm=None, dropout=0.0, residual=False):
        super().__init__()
        self.residual = residual and d_in == d_out   # Só soma se as dimensões batem
        camadas = [nn.Linear(d_in, d_out)]
        if norm == 'bn': camadas.append(nn.BatchNorm1d(d_out))
        if norm == 'ln': camadas.append(nn.LayerNorm(d_out))
        camadas.append(nn.ReLU())
        if dropout > 0: camadas.append(nn.Dropout(dropout))
        self.f = nn.Sequential(*camadas)

    def forward(self, x):
        return x + self.f(x) if self.residual else self.f(x)


class RedeNeural(nn.Module):
    def __init__(self, entradas, largura=64, profundidade=2, norm=None, dropout=0.0, residual=False):
        super().__init__()
        blocos = [Bloco(entradas, largura, norm, dropout)]   # projeção de entrada, sem residual
        blocos += [Bloco(largura, largura, norm, dropout, residual) for _ in range(profundidade - 1)]
        self.corpo = nn.Sequential(*blocos)
        self.saida = nn.Linear(largura, 1)

    def forward(self, x):
        return self.saida(self.corpo(x))

# Dados em lotes
dataset_treino = TensorDataset(X_treino_t, y_treino_t)
loader_treino = DataLoader(dataset_treino, batch_size=64, shuffle=True, drop_last=True)

# Seed, modelo, perda e otimizador
torch.manual_seed(13)
modelo = RedeNeural(entradas, profundidade=2, dropout=0.2).to(device)
criterio = nn.MSELoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)


def treinar_uma_epoca(modelo, loader, criterio, otimizador, device):
    modelo.train()
    soma_loss = 0.0
    n_amostras = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        otimizador.zero_grad()
        loss = criterio(modelo(xb), yb)
        loss.backward()
        otimizador.step()
        soma_loss += loss.item() * len(xb)
        n_amostras += len(xb)
    return soma_loss / n_amostras


def avaliar(modelo, X, y, criterio, device):
    modelo.eval()
    with torch.no_grad():
        X_dev, y_dev = X.to(device), y.to(device)
        return criterio(modelo(X_dev), y_dev).item()


# Reverte o alvo: z-score -> ln -> preço real
def reverter(y_z):
    ln = y_z * std_y + media_y   # desfaz o z-score (estatísticas do treino)
    return np.exp(ln)            # desfaz o ln


n_epocas = 50
hist_treino = []
hist_teste = []

for epoca in range(n_epocas):
    loss_treino = treinar_uma_epoca(modelo, loader_treino, criterio, otimizador, device)
    loss_teste = avaliar(modelo, X_teste_t, y_teste_t, criterio, device)

    hist_treino.append(loss_treino)
    hist_teste.append(loss_teste)

    if (epoca + 1) % 10 == 0:
        print(f"Época {epoca+1:3d} | treino: {loss_treino:.4f} | teste: {loss_teste:.4f}")


# Previsões em preço real
modelo.eval()
with torch.no_grad():
    pred_teste_z = modelo(X_teste_t.to(device)).cpu().numpy().flatten()

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
print(f'Seed utilizada: {torch.initial_seed()}\n')


N_EPOCAS = 100   # defina uma vez, antes das funções, e use no laço principal também


# Valores padrão da rede: o rótulo sempre mostra TODOS os campos, mesmo os que você não passou
PADRAO = dict(largura=64, profundidade=2, norm=None, dropout=0.0, residual=False)
tabela = []   # guarda o resumo de cada experimento


def executar(seed, n_epocas=N_EPOCAS, **arq):
    torch.manual_seed(seed)
    modelo = RedeNeural(entradas, **arq).to(device)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    
    for _ in range(n_epocas):
        treinar_uma_epoca(modelo, loader_treino, criterio, otimizador, device)

    modelo.eval()
    with torch.no_grad():
        previsto = reverter(modelo(X_teste_t.to(device)).cpu().numpy().flatten())

    erro = previsto - preco_real
    rmse = np.sqrt(np.mean(erro ** 2))
    mae = np.mean(np.abs(erro))
    r2 = 1 - np.sum(erro ** 2) / np.sum((preco_real - preco_real.mean()) ** 2)
    return rmse, mae, r2


def rodar_seeds(n_seeds=5, n_epocas=N_EPOCAS, verbose=True, **arq):
    cfg = {**PADRAO, **arq}
    rotulo = (f"prof={cfg['profundidade']} | larg={cfg['largura']} | drop={cfg['dropout']} | "
              f"norm={cfg['norm']} | residual={cfg['residual']} | épocas={n_epocas}")

    resultados = np.array([executar(seed, n_epocas, **arq) for seed in range(n_seeds)])
    media, desvio = resultados.mean(axis=0), resultados.std(axis=0)

    if verbose:
        print(f"\n=== {rotulo} ({n_seeds} seeds) ===")
        print(f"RMSE: {media[0]:,.0f} ± {desvio[0]:,.0f}")
        print(f"MAE:  {media[1]:,.0f} ± {desvio[1]:,.0f}")
        print(f"R²:   {media[2]:.4f} ± {desvio[2]:.4f}")

    tabela.append((rotulo, media, desvio))


def mostrar_tabela(top=None):
    print("\n" + "=" * 118)
    print(f"{'Configuração':78s} {'RMSE':>12s} {'MAE':>10s} {'R²':>8s}")
    for rotulo, media, desvio in sorted(tabela, key=lambda linha: linha[1][0])[:top]:   # menor RMSE primeiro
        print(f"{rotulo:78s} {media[0]:>12,.0f} {media[1]:>10,.0f} {media[2]:>8.4f}")



import itertools
import time

# ---- Grade de combinações (edite as listas para aumentar ou reduzir) ----
GRADE = dict(
    profundidade=[2],
    dropout=[0.2, 0.4],
    norm=[None, 'ln'],
    residual=[False],
    n_epocas=[10, 20, 30, 40, 50, 100, 200, 500],
)
N_SEEDS = 5

chaves = list(GRADE.keys())
combinacoes = [dict(zip(chaves, valores)) for valores in itertools.product(*GRADE.values())]
total = len(combinacoes)
print(f"{total} combinações x {N_SEEDS} seeds")

linhas_csv = []
inicio = time.time()
for i, comb in enumerate(combinacoes, start=1):
    try:
        rodar_seeds(n_seeds=N_SEEDS, verbose=False, **comb)
    except Exception as e:
        print(f"[{i}/{total}] ERRO em {comb}: {e}")
        continue

    rotulo, media, desvio = tabela[-1]
    linhas_csv.append({**comb, 'norm': str(comb['norm']),
                       'rmse': media[0], 'rmse_std': desvio[0],
                       'mae': media[1], 'mae_std': desvio[1],
                       'r2': media[2], 'r2_std': desvio[2]})
    pd.DataFrame(linhas_csv).to_csv('resultados_grade.csv', index=False)   # salva a cada combinação

    decorrido = time.time() - inicio
    restante = decorrido / i * (total - i)
    print(f"[{i}/{total}] RMSE {media[0]:,.0f} | {comb} | decorrido {decorrido/60:.1f} min | restante ~{restante/60:.1f} min", flush=True)

mostrar_tabela()

# Removendo outliers (opcional)
outliers = (df_treino['GrLivArea'] > 4000) & (df_treino['SalePrice'] < 300000)
df_treino = df_treino[~outliers]