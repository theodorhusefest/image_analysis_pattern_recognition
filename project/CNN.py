import os
import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torchvision import transforms, utils


class OperatorNet(nn.Module):
    def __init__(self):
        """ Init """
        super().__init__()
        self.fc1 = nn.Linear(32*32, 100)
        self.fc2 = nn.Linear(100, 5)
        
    def forward(self, x):
        """ Forward pass """
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def predict(self, x):
        """ Returns softmaxed labels"""
        
        x = self.forward(x)
        return F.softmax(x)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

    
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
        image = io.imread(img_name)

        label = self.dataframe.loc[idx].label
        
        if self.transform:
            sample = self.transform(image)
        
        if self.grayscale:
            image = rgb2gray(image)
        
        image = image.astype(np.float32)
        return (image, label)


def get_dataloaders(dataset,
        batch_size = 4,
        valid_size = 0.2,
        shuffle = True,
        random_seed = 1):

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
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
    )
    return train_loader, valid_loader