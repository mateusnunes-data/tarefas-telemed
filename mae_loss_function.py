import torch


def loss_function_MAE(pred, real):
    return torch.abs(real - pred).mean()

    
pred = torch.tensor([5.0, 10.0, 16.0])
real = torch.tensor([6.0, 12.0, 15.0])

print(loss_function_MAE(pred, real))