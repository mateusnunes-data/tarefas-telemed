import torch

matriz_a = torch.rand(2, 3) #matriz 2x3 com valores aleatórios (por padrão do rand, os valores serão entre 0 e 1)
matriz_b = torch.rand(3, 4) #matriz 3x4 (mesma coisa)

matriz_c = torch.mm(matriz_a, matriz_b) #torch.mm = multiplicação das matrizes

print(matriz_a.shape, matriz_b.shape, matriz_c.shape)
'''
(2,3), (3,4) -> o 3 de A e B são "consumidos"
2 linhas de A + 4 colunas de B geram o output (2,4)
'''
print(matriz_c)