# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:15:52 2021

@author: PC
"""

import os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# if CUDA is available, set device to cuda
device= 'cpu'

# USE class NN as defined in <create-class-model>
model = NN().to(device)
print(model)


# pass data to model and predict probabilities using Softmax
X = torch.rand((1,28,28,), device= device)
logit = model(X)
pred_prob = nn.Softmax(dim=1)(logit)
y_pred = pred_prob.argmax(1)
print(f"predicted class:{y_pred}")
# OUTPUT= predicted class:tensor([4])

# Investigate model layers
input_image = torch.rand(3,28,28)
print(input_image.size())

# flatten images
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# apply linear transformation using the stored weights and biases
layer1  = nn.Linear(in_features=28*28, out_features= 20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# apply ReLU activation
print(f"before ReLU:{hidden1}")
hidden1 = nn.ReLU()(hidden1)
print(f"after ReLU:{hidden1}")

# make a sequential container
seq_modules = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20,10))
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# map logits onto range [0,1] using Softmax
softmax = nn.Softmax(dim=1)
pred_prob = softmax(logits)

# access model structure and parameters
print(f"model structure: ", model,"\n")

for name, param in model.named_parameters():
   print(f" layer: {name} | size: {param.size()} | values: {param[:2]}\n")














