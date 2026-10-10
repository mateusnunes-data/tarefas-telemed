import torch


def loss_function_BCE(pred, real):
    return ((real * torch.log(pred)) + (1 - real) * torch.log(1 - pred)).mean()


pred = torch.tensor([5.0, 10.0, 16.0])
real = torch.tensor([6.0, 12.0, 15.0])

print(loss_function_BCE(pred, real))