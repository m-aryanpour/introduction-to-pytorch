# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:47:57 2021

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

#
# DEFINE CLASS MODEL AND ASSIGN ONE INSTANCE
#

# optimizing the model

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# training function
def train(dataloader, model, loss_func, optimizer):
   size= len(dataloader.dataset)
   for batch, (X,y) in enumerate(dataloader):
      X,y = X.to(device), y.to(device)
      
      # compute prediction error
      predict = model(X)
      loss    = loss_func(predict, y)
      
      # backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      if batch%100 == 0:
         loss, current = loss.item(), batch*len(X)
         print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
         
# test function
def test(dataloader, model, loss_func):
   size = len(dataloader.dataset)
   num_batches = len(dataloader)
   model.eval()
   test_loss, correct = 0, 0
   with torch.no_grad():
      for X,y in dataloader:
         X,y = X.to(device), y.to(device)
         prediction = model(X)
         test_loss += loss_func(prediction, y).item()
         correct   += (prediction.argmax(1)==y).type(torch.float).sum().item()
         test_loss /= num_batches
         correct /= size
         print(f"test error: accuracy: {(100*correct):>0.1f}%, avg. loss:{test_loss:>8f}\n")
         
   
# execute train and test processes
epochs = 5
for t in range(epochs):
   print(f"epoch {t+1}\n-----------------------------------")
   train(train_dataloader, model, loss_func, optimizer)
   test(test_dataloader, model, loss_func)
print(" DONE with train and test ")
   
# last line of results
# test error:  accuracy: 0.1%, avg. loss:0.010044
   
   
   
   
   
   
   
   
              