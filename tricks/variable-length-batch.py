import os
import torch
import cv2
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle


data = list()
data.append(np.ones((3, 300, 390)))
data.append(np.ones((3, 300, 400)))
data.append(np.ones((3, 300, 350)))
data.append(np.ones((3, 300, 360)))
data.append(np.ones((3, 300, 490)))
data.append(np.ones((3, 300, 500)))
data.append(np.ones((3, 300, 450)))
data.append(np.ones((3, 300, 460)))

batch_size = 4

class PadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        img = self.data[index]
        img = np.pad(img, ((0, 0), (0, 0), (0, max_lengths[index] - img.shape[2])), 'constant')
        img = torch.Tensor(img)
        
        return img, max_lengths[index]

    def __len__(self):
        return len(self.data)

train_loader = DataLoader(PadDataset(data=data),  batch_size=batch_size, shuffle=False)


lengths = list()
for img in data:
    lengths.append(img.shape[2])

# get group max value
if len(lengths) % batch_size != 0:
    print('batch_size must be chosen carefully')
    exit(1)

max_lengths = list()
for i in range(len(lengths)):
    max_lengths.append(max(lengths[i//batch_size*batch_size : i//batch_size*batch_size+batch_size]))
print(max_lengths)


for step, (img, length) in enumerate(train_loader):
    print(img.shape)

