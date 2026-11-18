import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

# ======================= CONFIGURAÇÕES =======================
PASTA = Path(__file__).resolve().parent
DATA_DIR = PASTA / 'Dados'               # pasta com os arquivos .npz
ARQUIVO_PESOS = PASTA / 'resnet101.pt'   # onde os pesos da melhor época ficam salvos
SEED = 13
BATCH_SIZE = 64          # se der "CUDA out of memory", reduza (32, 16...)
LR = 0.0003
N_EPOCAS = 100
PACIENCIA = 10           # early stopping: épocas seguidas sem melhorar na validação
NUM_WORKERS = 4          # processos que preparam os lotes em paralelo
K = 5                    # número de clusters do K-Means (atividade 8)
# Nome de cada classe, na ordem dos rótulos 0, 1, 2, 3, 4. Confira a ordem no README do dataset.
NOMES = ['COVID-19', 'Opacidade pulmonar', 'Normal', 'Pneumonia viral', 'Tuberculose']

'''
Atividade 7:
[x] Baixar o dataset e pegar apenas o primeiro canal de cada imagem.
[x] Normalizar as imagens para que todos os valores fiquem entre 0 e 1.
[x] Z-score apenas com o conjunto de treinamento, usando v2.Compose.
[x] Treinar um modelo de classificação com a ResNet.
[x] Apresentar precisão, recall e F1 por classe.

Atividade 8:
[x] Remover a última camada da ResNet treinada na atividade 7.
[x] Clusterizar o conjunto de teste com K-Means (K=5).
'''


# ======================= DADOS =======================
def ler_primeiro_canal(caminho):
    """Lê as imagens do .npz em pedaços de 256, guardando só o 1º canal.
    As imagens estão em float64 com 3 canais: carregar tudo de uma vez ocuparia ~23 GB de RAM."""
    with zipfile.ZipFile(caminho) as z, z.open('image.npy') as f:
        versao = np.lib.format.read_magic(f)
        if versao == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        n, h, w, c = shape
        saida = np.empty((n, h, w), dtype=np.uint8)    # pixels de 0 a 255 cabem em 1 byte
        for i in range(0, n, 256):
            m = min(256, n - i)
            bloco = np.frombuffer(f.read(m * h * w * c * dtype.itemsize), dtype=dtype).reshape(m, h, w, c)
            saida[i:i + m] = bloco[..., 0]              # fica só com o 1º canal
    return saida


def carregar(nome):
    caminho = DATA_DIR / f'Dataset5_raw_{nome}.npz'
    imagens = ler_primeiro_canal(caminho)
    rotulos = np.load(caminho)['image_label'].ravel()  # ravel: (N, 1) -> (N,)
    return torch.from_numpy(imagens).unsqueeze(1), rotulos   # imagens: (N, 1, 224, 224)


class DatasetImagens(Dataset):
    def __init__(self, X, y, transform):
        self.X, self.y, self.transform = X, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].repeat(3, 1, 1)                   # 1 canal -> 3 cópias: a ResNet espera 3 canais
        return self.transform(x), self.y[i]


# ======================= REDE =======================
def treinar_uma_epoca(modelo, loader, criterio, otimizador, device):
    modelo.train()
    soma_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        otimizador.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):   # contas em 16 bits: mais rápido e usa menos memória
            saida = modelo(x)
        loss = criterio(saida.float(), y)
        loss.backward()
        otimizador.step()
        soma_loss += loss.item() * len(x)
        n += len(x)
    return soma_loss / n


def passar_pela_rede(modelo, loader, device):
    """Roda a rede em todas as imagens do loader, sem treinar, e junta as saídas numa matriz."""
    modelo.eval()
    saidas = []
    with torch.no_grad():
        for x, _ in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                saida = modelo(x.to(device))
            saidas.append(saida.float().cpu())
    return torch.cat(saidas)


def main():
    device = 'cuda'
    torch.manual_seed(SEED)

    # ======================= ATIVIDADE 7 =======================
    # ---------- Dados (já vêm separados em treino / validação / teste) ----------
    X_treino, rot_treino = carregar('train')
    X_val, rot_val = carregar('val')
    X_teste, rot_teste = carregar('test')

    # Rótulos viram 0, 1, 2, ... (a CrossEntropyLoss pede inteiros começando em 0)
    classes = np.unique(rot_treino)
    y_treino = torch.from_numpy(np.searchsorted(classes, rot_treino))
    y_val = torch.from_numpy(np.searchsorted(classes, rot_val))
    y_teste = torch.from_numpy(np.searchsorted(classes, rot_teste))
    print(f'Classes: {classes.tolist()}')
    print(f'Treino: {tuple(X_treino.shape)} | Validação: {tuple(X_val.shape)} | Teste: {tuple(X_teste.shape)}')

    # ---------- Z-score com média e desvio SÓ do treino (na escala 0-1) ----------
    X01 = X_treino.float().div_(255)                    # div_ divide no próprio tensor: economiza memória
    media, desvio = X01.mean().item(), X01.std().item()
    del X01
    print(f'Média do treino: {media:.4f} | desvio do treino: {desvio:.4f}')

    tf_treino = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),          # 0-255 -> 0-1
        v2.Normalize(mean=[media] * 3, std=[desvio] * 3),   # z-score (mesmo valor nos 3 canais)
    ])
    tf_eval = v2.Compose([                              # validação e teste: sem flip
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[media] * 3, std=[desvio] * 3),
    ])

    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True)
    loader_treino = DataLoader(DatasetImagens(X_treino, y_treino, tf_treino), shuffle=True, drop_last=True, **kw)
    loader_val = DataLoader(DatasetImagens(X_val, y_val, tf_eval), **kw)
    loader_teste = DataLoader(DatasetImagens(X_teste, y_teste, tf_eval), **kw)

    # ---------- Modelo: ResNet101 do zero ----------
    modelo = models.resnet101(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, len(classes))   # última camada: 2048 -> um valor por classe
    modelo = modelo.to(device)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

    # ---------- Treino com early stopping (ou carrega o que já foi treinado) ----------
    if ARQUIVO_PESOS.exists():
        modelo.load_state_dict(torch.load(ARQUIVO_PESOS))
        print(f'Pesos carregados de {ARQUIVO_PESOS.name}: treino pulado (apague o arquivo para treinar de novo).')
    else:
        melhor_loss, sem_melhora = float('inf'), 0
        for epoca in range(1, N_EPOCAS + 1):
            loss_treino = treinar_uma_epoca(modelo, loader_treino, criterio, otimizador, device)
            saidas_val = passar_pela_rede(modelo, loader_val, device)
            loss_val = criterio(saidas_val, y_val).item()
            acc_val = (saidas_val.argmax(1) == y_val).float().mean().item()
            print(f'Época {epoca:3d} | loss treino {loss_treino:.4f} | loss val {loss_val:.4f} | acc val {acc_val:.4f}')

            if loss_val < melhor_loss:                  # melhorou: salva os pesos desta época
                melhor_loss, sem_melhora = loss_val, 0
                torch.save(modelo.state_dict(), ARQUIVO_PESOS)
            else:
                sem_melhora += 1
                if sem_melhora >= PACIENCIA:
                    print(f'Early stopping na época {epoca}.')
                    break
        modelo.load_state_dict(torch.load(ARQUIVO_PESOS))   # volta para a melhor época

    # ---------- Teste: precisão, recall e F1 por classe ----------
    pred = passar_pela_rede(modelo, loader_teste, device).argmax(1)
    print(classification_report(y_teste.numpy(), pred.numpy(), target_names=NOMES, digits=4))

    # ======================= ATIVIDADE 8 =======================
    # 1) Remove a última camada. A fc recebia um vetor de 2048 números e devolvia um valor por classe.
    #    Trocando por Identity (uma camada que devolve a entrada sem mudar nada),
    #    a rede passa a devolver o próprio vetor de 2048 números: o embedding da imagem.
    modelo.fc = nn.Identity()

    # 2) Embeddings do teste: uma linha por imagem, 2048 colunas
    embeddings = passar_pela_rede(modelo, loader_teste, device).numpy()
    print(f'Embeddings do teste: {embeddings.shape}')

    # 3) K-Means com K=5. Ele só vê os vetores, não os rótulos.
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEED)   # n_init=10: roda 10 vezes e fica com a melhor
    clusters = kmeans.fit_predict(embeddings)

    # 4) Resultado: quantas imagens de cada classe caíram em cada cluster
    tabela = pd.crosstab(clusters, y_teste.numpy(), rownames=['cluster'], colnames=['classe'])
    tabela.columns = [NOMES[c] for c in tabela.columns]       # troca 0, 1, 2... pelo nome da doença
    print(tabela)


if __name__ == '__main__':      # obrigatório no Windows quando NUM_WORKERS > 0
    main()