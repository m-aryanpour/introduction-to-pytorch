# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:38:55 2021

@author: PC
"""

import torch
import numpy as np

# initialize a tensor
data = [[1, 3], [2, 4]]
x_data = torch.tensor(data)

# from a numpy array
data_np = np.array(data)
x_np = torch.from_numpy(data_np)

# from random or constant values

shape= (3,2,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor= torch.zeros(shape)

print(f" random: {rand_tensor}")
print(f" ones  : {ones_tensor}")
print(f" zeros : {zeros_tensor}")

# attributes of a tensor
print(f" shape    of x_data: {x_data.shape}")
print(f" datatype of x_data: {x_data.dtype}")
print(f" device   of x_data: {x_data.device}")

# move tensor to GPU if avaiable
if torch.cuda.is_available():
   x_data_gpu = x_data.to('cuda')
   
# indexing and slicing
tensor1 = torch.ones(3,4)
print(f" first row: {tensor1[0]}")
print(f" first column: {tensor1[:,0]}")
print(f" last column: {tensor1[...,-1]}")
tensor1[:,1]= -2
print(tensor1)

# joining tensors
t2 = torch.cat([tensor1, tensor1, tensor1], dim=1)
print(f" concat tensor for 3 times: ", t2)

# matrix multiplication
# three equivalent methods: y1=y2=y3
tensor2 = torch.rand([3,3])
y1 = tensor2 @ tensor2.T
y2 = tensor2.matmul(tensor2.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor2, tensor2.T, out= y3)

# element-wise muliplication
# three equivalent methods: z1=z2=z3
z1 = tensor2 * tensor2
z2 = tensor2.mul(tensor2)
z3 = 0*z1
torch.mul(tensor2, tensor2, out=z3)

# single-element tensors: aggregate values in a tensor
agg2 = tensor2.sum()
agg2_item = agg2.item()
print(agg2_item, type(agg2_item))

# in-place operations
print(f" tensor2= ",tensor2)
tensor2.add_(2)
print(f" tensor2= ",tensor2)

# bridge with a numpy array
t1 = 2*torch.ones([2,3])
print(f"t1: {t1}")
t1_np = t1.numpy()
print(f"t1_np: {t1_np}")
print(" now modify t1 by adding 3. -> t1.add_(3)")
t1.add_(3)
print(f"t1: {t1}")
print(" its numpy array has changed too!")
print(f"t1_np: {t1_np}")
print(" the reverse is also true -> t1_np /= 2.4")
t1_np /= 2.4
print(f"t1_np: {t1_np}")
print(f"t1: {t1}")









