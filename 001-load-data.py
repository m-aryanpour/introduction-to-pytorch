# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:58:53 2021

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
## download data and store in the current directory <data/FashionMNIST>
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz 
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

training_data=datasets.FashionMNIST(root="data",train=True, download=True,transform=ToTensor())
test_data=datasets.FashionMNIST(root="data",train=False, download=True,transform=ToTensor())

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
 