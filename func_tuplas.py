import torch

def tuple_tensor_shape(shape1, shape2):
    tensor1 = torch.rand(shape1)
    print(tensor1)
    tensor2 = torch.rand(shape2)
    print(tensor2)
    return tensor1 @ tensor2 #outra maneira de multiplicar matrizes: "@"

resultado = tuple_tensor_shape((2,3), (3,4))
print('\n\n','=' * 100)
print(f'\n\nA função gera {resultado}.\n\n\nO shape é {resultado.shape}.\n\n\n')