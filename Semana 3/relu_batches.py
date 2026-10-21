import torch



'''
Bom dia, pessoal! Aqui está a próxima tarefa para quarta-feira:
[x] O que é learning rate?
[x] O que é um otimizador? Como ele afeta a learning rate?
[x] O que diferencia o Adam do SGD?
[x] O que é um batch? Por que ele é utilizado?
[x] O que é uma função de ativação?
[x] Implemente uma função que recebe um tensor e seu output é a função de ativação ReLU com suporte a batches. Não utilize a função ReLU pronta!
[x] Adicione suporte a batches para uma loss function da penúltima atividade (ex: MSE, etc...).
'''

def relu(x):
    # Onde x for maior que zero, manter o x. Onde não for maior que zero, usaremos 0
    return torch.where(x > 0, x, 0)

tensor_teste = torch.tensor([2.0, 5.0, 8.0, -1.0])
print(relu(tensor_teste))

def mse_batch(pred, real):
    erro = (real - pred) ** 2

    # dim = -1: última dimensão. A função em teoria já tinha suporte à batches, mas fiz desta forma para mostrar o que a atividade pedia
    loss_por_amostra = erro.mean(dim = -1)

    return loss_por_amostra.mean()



tensor_teste = torch.tensor([2.0, 5.0, 8.0, -1.0])
pred = torch.tensor([2.0, 5.0, 8.0])
real = torch.tensor([3.0, 4.0, 10.0])



print(relu(tensor_teste))
print(mse_batch(pred, real))