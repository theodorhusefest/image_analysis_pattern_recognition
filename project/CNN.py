import os
import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class OperatorsDataset(Dataset):
    """ Custom Dataset for Operators"""
    
    def __init__(self, root_dir, csv_file, transform = None, grayscale = True):
        
        self.root_dir = root_dir
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform
        self.grayscale = grayscale
        
    def __len__(self):
        return self.dataframe.shape[0]
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.dataframe.loc[idx].path)
        image = Image.open(img_name)

        label = self.dataframe.loc[idx].label
        
        if self.transform:
            sample = self.transform(image)
        
        image = np.asarray(image)        
        if self.grayscale:
            image = rgb2gray(image)
            
        image = image.astype(np.float32)

        return (image, label)


def get_dataloaders(dataset,
        batch_size = 64,
        valid_size = 0.1,
        shuffle = True,
        random_seed = 1,
        num_workers = 10,
        pin_memory = 1):

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers = num_workers, pin_memory = pin_memory
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers = num_workers, pin_memory = pin_memory
    )
    return train_loader, valid_loader


class Classifier(nn.Module):
    def __init__(self):
        """ Init """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels = 32, kernel_size= 5, padding= 2)
        self.conv1_1 = nn.Conv2d(in_channels= 32, out_channels = 32, kernel_size= 5, padding= 2)
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels = 64, kernel_size= 3, padding= 2)
        self.conv3 = nn.Conv2d(in_channels= 64, out_channels = 64, kernel_size= 3, padding= 2)
        self.fc1 = nn.Linear(64*13*13, 1024)
        self.fc2 = nn.Linear(1024,128)
        self.fc3 = nn.Linear(128,14)
        
    def forward(self, x):
        """ Forward pass """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(-1, 64*13*13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def predict(self, x):
        """ Returns softmaxed labels"""
        
        x = self.forward(x)
        return F.softmax(x)

