# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:14:11 2021

@author: PC
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:38:55 2021

@author: PC
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


#
# LOAD DATA USING A PREVIOUS SCRIPT
#


# Get cpu or gpu 
device= "cpu" if not torch.cuda.is_available() else "cuda"
print(" set device to {}".format(device))

# Define NN model
class NN(nn.Module):
   def __init__(self):
      super(NN,self).__init__()
      self.flatten = nn.Flatten()
      self.linear_relu_stack = nn.Sequential(
         nn.Linear(28*28, 512),
         nn.ReLU(),
         nn.Linear(512,512),
         nn.ReLU(),
         nn.Linear(512, 10),
         nn.ReLU()
         )
   
   def forward(self,x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits

model = NN().to(device)
print(model)
