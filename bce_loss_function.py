import torch


def loss_function_BCE(pred, real):
    return ((real * torch.log(pred)) + (1 - real) * torch.log(1 - pred)).mean()


pred = torch.tensor([0.9, 0.2, 0.7])
real = torch.tensor([1.0, 0.0, 1.0])

print(loss_function_BCE(pred, real))