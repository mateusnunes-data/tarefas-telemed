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

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
bce_loss = nn.BCELoss()

'''
Adicionei as fórmulas como funções criadas no código,
porém utilizei também as funções nativas do PyTorch.
'''

pred = torch.tensor([2.0, 5.0, 8.0])
real = torch.tensor([3.0, 4.0, 10.0])

pred_bce = torch.tensor([0.03, 0.9, 0.07])
real_bce = torch.tensor([0.0, 1.0, 0.0])

#MSE =========================================
print(loss_function_MSE(pred, real))
print(mse_loss(pred, real))

#RMSE ========================================
print(loss_function_RMSE(pred, real))
print(torch.sqrt(mse_loss(pred, real)))

#MAE =========================================
print(loss_function_MAE(pred, real))
print(mae_loss(pred, real))

#BCE =========================================
print(loss_function_BCE(pred_bce, real_bce))
print(bce_loss(pred_bce, real_bce))