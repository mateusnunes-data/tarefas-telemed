import copy
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from sklearn.metrics import classification_report

# ======================= CONFIGURAÇÕES =======================
DATA_DIR = Path(__file__).resolve().parent / 'Dados'   # pasta "Dados" ao lado deste arquivo
SEED = 13               # 
BATCH_SIZE = 64         # se der "CUDA out of memory", reduza (64, 32...)
LR = 0.0003
N_EPOCAS = 100
PACIENCIA = 10          # early stopping: épocas seguidas sem melhora na validação

NUM_WORKERS = 4         # processos que preparam os lotes em paralelo (0 = desliga)
USAR_AMP = True         # precisão mista (bfloat16): mais rápido e usa menos VRAM na RTX 4080

'''
[ ] Baixar o dataset e pegar apenas o primeiro canal de cada imagem.
[ ] Normalizar as imagens para que todos os valores fiquem entre 0 e 1.
[ ] Z-score apenas com o conjunto de treinamento, usando v2.Compose.
[ ] Treinar um modelo de classificação com a ResNet.
[ ] Apresentar precisão, recall e F1 por classe.
'''


# ======================= LEITURA DOS .NPZ (fora do main por causa do Windows) =======================
def _ler_exato(f, n_bytes):
    buf = bytearray()
    while len(buf) < n_bytes:
        pedaco = f.read(n_bytes - len(buf))
        if not pedaco:
            raise EOFError("Arquivo terminou antes do esperado.")
        buf += pedaco
    return bytes(buf)


def ler_primeiro_canal(caminho, chave='image', bloco=256):
    """Lê o array `chave` do .npz em blocos, guardando só o 1º canal em uint8.
    Evita carregar o array inteiro (float64 com 3 canais ocuparia ~23 GB no treino)."""
    with zipfile.ZipFile(caminho) as z:
        with z.open(f'{chave}.npy') as f:
            versao = np.lib.format.read_magic(f)
            if versao == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            if fortran or dtype.hasobject or len(shape) != 4 or shape[-1] < 1:
                raise ValueError(f"Formato inesperado em {caminho.name}: shape={shape}, dtype={dtype}. "
                                 f"Esperado (N, H, W, C) numérico.")

            n, h, w, c = shape
            bytes_por_imagem = h * w * c * dtype.itemsize
            saida = np.empty((n, h, w), dtype=np.uint8)

            for i in range(0, n, bloco):
                m = min(bloco, n - i)
                dados = np.frombuffer(_ler_exato(f, m * bytes_por_imagem), dtype=dtype).reshape(m, h, w, c)
                if i == 0:
                    iguais = c == 1 or all(np.array_equal(dados[..., 0], dados[..., k]) for k in range(1, c))
                    inteiros = bool(np.all(dados[..., 0] == np.rint(dados[..., 0])))
                    print(f"  [{caminho.name}] canais idênticos (1º bloco): {iguais} | "
                          f"valores inteiros: {inteiros} | min={dados.min():.1f} max={dados.max():.1f}")
                saida[i:i + m] = np.clip(np.rint(dados[..., 0]), 0, 255).astype(np.uint8)   # 1º canal
    return saida


def carregar_split(nome):
    caminho = DATA_DIR / f'Dataset5_raw_{nome}.npz'
    imagens = ler_primeiro_canal(caminho)
    with np.load(caminho) as d:
        rotulos = d['image_label']
        nomes = d['image_name']
    # Os arrays podem vir como coluna (N, 1): achata para (N,)
    if rotulos.ndim == 2 and rotulos.shape[1] == 1:
        rotulos = rotulos.ravel()
    if nomes.size == len(imagens):
        nomes = nomes.reshape(-1)
    if rotulos.ndim != 1 or len(rotulos) != len(imagens):
        raise ValueError(f"image_label em {nome} tem shape {rotulos.shape} (esperado 1 rótulo por imagem). "
                         f"Se for one-hot, avise para ajustar.")
    return imagens, rotulos, nomes


class DatasetImagens(Dataset):
    def __init__(self, X, y, transform):
        self.X, self.y, self.transform = X, y, transform   # X: uint8 (N, 1, H, W)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.transform(self.X[i]), self.y[i]         # 1 canal; a repetição 3x é feita na GPU


class Repetir3Canais(nn.Module):
    # Repete o canal 3 vezes dentro da GPU (transfere 1/3 dos dados pelo PCIe)
    def forward(self, x):
        return x.repeat(1, 3, 1, 1).contiguous(memory_format=torch.channels_last)


def treinar_uma_epoca(modelo, loader, criterio, otimizador, device):
    modelo.train()
    soma_loss = torch.zeros((), device=device)              # acumula na GPU: evita sincronizar a cada lote
    n_amostras = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        otimizador.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USAR_AMP):
            logits = modelo(xb)
        loss = criterio(logits.float(), yb)                 # perda sempre em float32
        loss.backward()
        otimizador.step()
        soma_loss += loss.detach() * len(xb)
        n_amostras += len(xb)
    return soma_loss.item() / n_amostras


def avaliar(modelo, loader, criterio, device):
    modelo.eval()
    soma_loss = torch.zeros((), device=device)
    n_amostras = 0
    previsoes, reais = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USAR_AMP):
                logits = modelo(xb)
            logits = logits.float()
            soma_loss += criterio(logits, yb) * len(xb)
            n_amostras += len(xb)
            previsoes.append(logits.argmax(dim=1).cpu())
            reais.append(yb.cpu())
    return soma_loss.item() / n_amostras, torch.cat(previsoes).numpy(), torch.cat(reais).numpy()


def main():
    # ======================= GPU =======================
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA não está disponível: o PyTorch instalado provavelmente é a versão só de CPU.\n"
            "Reinstale com o comando de https://pytorch.org/get-started/locally (Windows, Pip, CUDA)."
        )
    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB | "
          f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")

    torch.backends.cudnn.benchmark = True    # escolhe os kernels mais rápidos (tamanho de entrada é fixo)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # ======================= CARREGAMENTO (já vem dividido: treino / validação / teste) =======================
    print("Lendo os arquivos (só o 1º canal de cada imagem)...")
    img, rot, nom = {}, {}, {}
    for nome in ['train', 'val', 'test']:
        img[nome], rot[nome], nom[nome] = carregar_split(nome)

    # Aviso de vazamento: mesma imagem em conjuntos diferentes
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        comum = set(nom[a].tolist()) & set(nom[b].tolist())
        if comum:
            print(f"ATENÇÃO: {len(comum)} imagens com o mesmo nome em '{a}' e '{b}'.")

    # Classes definidas pelo treino; val/test usam o mesmo mapeamento
    classes = np.unique(rot['train'])
    n_classes = len(classes)

    def codificar(r):
        pos = np.clip(np.searchsorted(classes, r), 0, n_classes - 1)
        if not np.array_equal(classes[pos], r):
            raise ValueError("Há rótulos em val/test que não existem no treino.")
        return torch.from_numpy(pos).long()

    X_treino = torch.from_numpy(img['train']).unsqueeze(1)   # (N, 1, 224, 224) uint8
    X_val = torch.from_numpy(img['val']).unsqueeze(1)
    X_teste = torch.from_numpy(img['test']).unsqueeze(1)
    y_treino, y_val, y_teste = codificar(rot['train']), codificar(rot['val']), codificar(rot['test'])

    print(f"\nClasses ({n_classes}): {classes.tolist()}")
    print(f"Treino: {tuple(X_treino.shape)} | Validação: {tuple(X_val.shape)} | Teste: {tuple(X_teste.shape)}")
    for nome, yy in [('treino', y_treino), ('val', y_val), ('teste', y_teste)]:
        print(f"  {nome:7s} imagens por classe: {torch.bincount(yy, minlength=n_classes).tolist()}")

    # ======================= Z-SCORE SÓ COM O TREINO (escala 0-1) =======================
    Xf = X_treino.float().div_(255)                         # ~4 GB temporários
    media = Xf.mean().item()
    desvio = Xf.std().item()
    del Xf
    print(f"Média treino: {media:.4f} | desvio treino: {desvio:.4f}")

    # uint8 -> float32 em [0, 1] (scale=True) -> z-score com média/desvio do treino.
    # Os 3 canais seriam cópias do mesmo canal: normalizar 1 canal e repetir depois é equivalente.
    tf_treino = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[media], std=[desvio]),
    ])
    tf_eval = v2.Compose([                                  # validação e teste: sem flip
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[media], std=[desvio]),
    ])

    kw = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    if NUM_WORKERS > 0:
        kw['prefetch_factor'] = 4

    loader_treino = DataLoader(DatasetImagens(X_treino, y_treino, tf_treino),
                               batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                               generator=torch.Generator().manual_seed(SEED), **kw)
    loader_val = DataLoader(DatasetImagens(X_val, y_val, tf_eval), batch_size=BATCH_SIZE, shuffle=False, **kw)
    loader_teste = DataLoader(DatasetImagens(X_teste, y_teste, tf_eval), batch_size=BATCH_SIZE, shuffle=False, **kw)

    # ======================= MODELO (ResNet101 do zero) =======================
    resnet = models.resnet101(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)  # saída: um logit por classe
    modelo = nn.Sequential(Repetir3Canais(), resnet)
    modelo = modelo.to(device, memory_format=torch.channels_last)

    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

    # ======================= LOOP COM EARLY STOPPING (usa a VALIDAÇÃO) =======================
    hist_treino, hist_val = [], []
    melhor_loss, melhor_epoca, melhores_pesos, sem_melhora = float('inf'), 0, None, 0
    torch.cuda.reset_peak_memory_stats()

    for epoca in range(N_EPOCAS):
        t0 = time.time()
        loss_treino = treinar_uma_epoca(modelo, loader_treino, criterio, otimizador, device)
        loss_val, pred_val, real_val = avaliar(modelo, loader_val, criterio, device)
        acc_val = (pred_val == real_val).mean()

        hist_treino.append(loss_treino)
        hist_val.append(loss_val)

        if loss_val < melhor_loss:
            melhor_loss, melhor_epoca, sem_melhora = loss_val, epoca + 1, 0
            melhores_pesos = copy.deepcopy(modelo.state_dict())
            marca = "*"
        else:
            sem_melhora += 1
            marca = ""

        print(f"Época {epoca+1:3d} | treino: {loss_treino:.4f} | val: {loss_val:.4f} | "
              f"acc val: {acc_val:.4f} | {time.time() - t0:.1f}s {marca}")

        if epoca == 0:
            pico = torch.cuda.max_memory_allocated() / 1024**3
            print(f"      Pico de VRAM na 1ª época: {pico:.1f} GB de {props.total_memory / 1024**3:.1f} GB")

        if sem_melhora >= PACIENCIA:
            print(f"Early stopping na época {epoca+1}.")
            break

    modelo.load_state_dict(melhores_pesos)                  # restaura a melhor época
    print(f"\nMelhor época: {melhor_epoca} (loss val: {melhor_loss:.4f})")

    # ======================= TESTE (só aqui, uma vez) =======================
    _, pred, real = avaliar(modelo, loader_teste, criterio, device)

    print("\n" + classification_report(real, pred, labels=list(range(n_classes)),
                                       target_names=[str(c) for c in classes], digits=4, zero_division=0))
    print(f"Seed utilizada: {SEED}")


if __name__ == '__main__':      # obrigatório no Windows quando NUM_WORKERS > 0
    main()