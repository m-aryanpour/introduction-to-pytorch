# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 18:12:41 2021

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

#
# DEFINE CLASS MODEL AND ASSIGN ONE INSTANCE
#

#
# OPTIMIZE THE MODEL
#

# save the model
torch.save(model.state_dict(), "model1.pth")
print(f"model saved as model1.pth")


   
# load model1 as model2
model2 = NN()
model2.load_state_dict(torch.load("model1.pth"))

print(f"to see the structure of state_dict in model2, type <model2.state_dict")
print(f"to access state_dict of model2, type <model2.state_dict()")
   
# usage of loaded model
classes= [
"T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot",
]
   
model2.eval()
x,y = test_data[0][0], test_data[0][1]
with torch.no_grad():
   pred = model(x)
   predicted, actual = classes[pred[0].argmax(0)], classes[y]
   print(f' predicted: "{predicted}", actual: " {actual}"')
   
# result
#  predicted: "Sneaker", actual: " Ankle boot" 
# !! OMG
   
              