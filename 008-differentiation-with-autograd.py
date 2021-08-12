# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:52:09 2021

@author: PC
"""

import torch

# define a simple one-layer NN

x = torch.ones(6) # input
y = torch.zeros(4) # expected output
w = torch.randn(6,4, requires_grad= True) # initialize weights
b = torch.randn(4, requires_grad= True)  # initialize biases
z1 = torch.matmul(x, w) + b             # predicted output
z = torch.nn.Softmax(dim=0)(z1)
loss = torch.nn.functional.binary_cross_entropy(z, y)
print(f" loss: {loss}")

# compute gradients
loss.backward()  # compute 
print(w.grad)
print(b.grad)

# tensor gradients and jacobian products
inp = torch.eye(4, requires_grad= True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph= True)
print(f" first call: {inp.grad}")

out.backward(torch.ones_like(inp), retain_graph= True)
print(f" second call [accumulated gradients]: {inp.grad}")

inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph= True)
print(f" after zeroing gradients: {inp.grad}")