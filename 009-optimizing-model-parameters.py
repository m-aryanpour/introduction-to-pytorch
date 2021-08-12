# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:49:15 2021

@author: PC
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# uncomment and execute if data not already loaded
# reload data using a previous script
# exec(open("001-load-data.py").readline())

model = NN()

# adjust hyperparameters
learning_rate = 1e-3
batch_size    = 64
epochs        = 5

# define the loss function
loss_func = nn.CrossEntropyLoss()

# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# train and test loops
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_func, optimizer)
    test_loop(test_dataloader, model, loss_func)
print("train and test Done!")    

# epoch 1
# -------------------------------
# loss: 2.299718  [    0/60000]
# loss: 2.288355  [ 6400/60000]
# loss: 2.276624  [12800/60000]
# loss: 2.279927  [19200/60000]
# loss: 2.259863  [25600/60000]
# loss: 2.236958  [32000/60000]
# loss: 2.256164  [38400/60000]
# loss: 2.226077  [44800/60000]
# loss: 2.219879  [51200/60000]
# loss: 2.193057  [57600/60000]
# Test Error: 
#  Accuracy: 39.4%, Avg loss: 2.204182 

# epoch 2
# -------------------------------
# loss: 2.202354  [    0/60000]
# loss: 2.191863  [ 6400/60000]
# loss: 2.164001  [12800/60000]
# loss: 2.190173  [19200/60000]
# loss: 2.129886  [25600/60000]
# loss: 2.104414  [32000/60000]
# loss: 2.136825  [38400/60000]
# loss: 2.084475  [44800/60000]
# loss: 2.084348  [51200/60000]
# loss: 2.026994  [57600/60000]
# Test Error: 
#  Accuracy: 49.2%, Avg loss: 2.059734 

# epoch 3
# -------------------------------
# loss: 2.061697  [    0/60000]
# loss: 2.041403  [ 6400/60000]
# loss: 1.992759  [12800/60000]
# loss: 2.049079  [19200/60000]
# loss: 1.930823  [25600/60000]
# loss: 1.915289  [32000/60000]
# loss: 1.958412  [38400/60000]
# loss: 1.888721  [44800/60000]
# loss: 1.902629  [51200/60000]
# loss: 1.807140  [57600/60000]
# Test Error: 
#  Accuracy: 51.1%, Avg loss: 1.875505 

# epoch 4
# -------------------------------
# loss: 1.884750  [    0/60000]
# loss: 1.866392  [ 6400/60000]
# loss: 1.798832  [12800/60000]
# loss: 1.877814  [19200/60000]
# loss: 1.717154  [25600/60000]
# loss: 1.734660  [32000/60000]
# loss: 1.768978  [38400/60000]
# loss: 1.706983  [44800/60000]
# loss: 1.703207  [51200/60000]
# loss: 1.600674  [57600/60000]
# Test Error: 
#  Accuracy: 51.3%, Avg loss: 1.694199 

# epoch 5
# -------------------------------
# loss: 1.698234  [    0/60000]
# loss: 1.701260  [ 6400/60000]
# loss: 1.616141  [12800/60000]
# loss: 1.725434  [19200/60000]
# loss: 1.524656  [25600/60000]
# loss: 1.589858  [32000/60000]
# loss: 1.604527  [38400/60000]
# loss: 1.564895  [44800/60000]
# loss: 1.541871  [51200/60000]
# loss: 1.450118  [57600/60000]
# Test Error: 
#  Accuracy: 52.8%, Avg loss: 1.553975 

# epoch 6
# -------------------------------
# loss: 1.545074  [    0/60000]
# loss: 1.570775  [ 6400/60000]
# loss: 1.473647  [12800/60000]
# loss: 1.614383  [19200/60000]
# loss: 1.381250  [25600/60000]
# loss: 1.481769  [32000/60000]
# loss: 1.480402  [38400/60000]
# loss: 1.458290  [44800/60000]
# loss: 1.427172  [51200/60000]
# loss: 1.346671  [57600/60000]
# Test Error: 
#  Accuracy: 53.7%, Avg loss: 1.450696 

# epoch 7
# -------------------------------
# loss: 1.430737  [    0/60000]
# loss: 1.472506  [ 6400/60000]
# loss: 1.363011  [12800/60000]
# loss: 1.530911  [19200/60000]
# loss: 1.279935  [25600/60000]
# loss: 1.400105  [32000/60000]
# loss: 1.392996  [38400/60000]
# loss: 1.378281  [44800/60000]
# loss: 1.344460  [51200/60000]
# loss: 1.273410  [57600/60000]
# Test Error: 
#  Accuracy: 54.4%, Avg loss: 1.374381 

# epoch 8
# -------------------------------
# loss: 1.346247  [    0/60000]
# loss: 1.398318  [ 6400/60000]
# loss: 1.275534  [12800/60000]
# loss: 1.466210  [19200/60000]
# loss: 1.209320  [25600/60000]
# loss: 1.339374  [32000/60000]
# loss: 1.331726  [38400/60000]
# loss: 1.319075  [44800/60000]
# loss: 1.284207  [51200/60000]
# loss: 1.217707  [57600/60000]
# Test Error: 
#  Accuracy: 55.0%, Avg loss: 1.316946 

# epoch 9
# -------------------------------
# loss: 1.282145  [    0/60000]
# loss: 1.340569  [ 6400/60000]
# loss: 1.205162  [12800/60000]
# loss: 1.413193  [19200/60000]
# loss: 1.160208  [25600/60000]
# loss: 1.293292  [32000/60000]
# loss: 1.287339  [38400/60000]
# loss: 1.274274  [44800/60000]
# loss: 1.238719  [51200/60000]
# loss: 1.174384  [57600/60000]
# Test Error: 
#  Accuracy: 55.9%, Avg loss: 1.272592 

# epoch 10
# -------------------------------
# loss: 1.231080  [    0/60000]
# loss: 1.293622  [ 6400/60000]
# loss: 1.148044  [12800/60000]
# loss: 1.370873  [19200/60000]
# loss: 1.123674  [25600/60000]
# loss: 1.258321  [32000/60000]
# loss: 1.253301  [38400/60000]
# loss: 1.239414  [44800/60000]
# loss: 1.203053  [51200/60000]
# loss: 1.139846  [57600/60000]
# Test Error: 
#  Accuracy: 56.7%, Avg loss: 1.237301 

# train and test Done!
