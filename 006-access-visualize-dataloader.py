# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:38:55 2021

@author: PC
"""

# reload data using a previous script
exec(open("001-load-data.py").readline())

# visualize sample data from a dataset
# actually, there are only two key lines: 
   # 1: img, lable= trainin_data[..]
   # 2: plt.imshow(..)
   
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="brg")
plt.show()

# access all features and labels by iterating through dataloader
train_features, train_labels = next(iter(train_dataloader))
img1 = train_features[1]
lab1 = train_labels[1]
plt.imshow(img1.squeeze(), cmap='gray')




