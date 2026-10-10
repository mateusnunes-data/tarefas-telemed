import torch
import torch.nn as nn


def loss_function_MSE(pred, real):
    return ((real - pred) ** 2).mean()

def loss_function_RMSE(pred, real):
    return torch.sqrt(((real - pred) ** 2).mean())

def loss_function_MAE(pred, real):
    return torch.abs(real - pred).mean()

def loss_function_BCE(pred, real):
    return -((real * torch.log(pred)) + (1 - real) * torch.log(1 - pred)).mean()
'''
Adicionei as fórmulas como funções criadas no código,
porém utilizei também as funções nativas do PyTorch.
Deixei os números no formato abaixo apenas para
reaproveitamento em todas as funções,
já que BCE utiliza probabilidade.
'''
pred = torch.tensor([0.05, 0.1, 0.16])
real = torch.tensor([0.06, 0.12, 0.15])

#MSE =========================================
print(loss_function_MSE(pred, real))

mse_loss = torch.nn.MSELoss()

print(mse_loss(pred, real))

#RMSE ========================================
print(loss_function_RMSE(pred, real))

mse_loss = nn.MSELoss()

print(torch.sqrt(mse_loss(pred, real)))

#MAE =========================================
print(loss_function_MAE(pred, real))

mae_loss = nn.L1Loss()

print(mae_loss(pred, real))

#BCE =========================================
print(loss_function_BCE(pred, real))

bce_loss = nn.BCELoss()

print(bce_loss(pred, real))

# ============================================
