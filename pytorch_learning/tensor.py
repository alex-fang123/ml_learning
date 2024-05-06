"""
https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
"""
import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.rand(shape)

# %% tensor attributes
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %% tensor operations

if torch.cuda.is_available():
    tensor = tensor.to('cuda')

tensor = torch.ones(4, 4)
tensor[:, 1] = 0

t1 = torch.cat([tensor,]*3, dim=1)

tensor.mul(tensor)
tensor * tensor  # same as above

tensor - 1

tensor.matmul(tensor.T)
tensor @ tensor.T  # same as above

tensor.add_(5)
