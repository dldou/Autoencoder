#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:21:53 2022

@author: dldou
"""

import torch
from torchsummary import summary

#Import MNIST of handwritten digits dataset
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import *
from utils import *
from data import *


if __name__=="__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Instanciation of the model
    AE = AutoEncoder()
    AE.to(device)
    summary(AE, (1, 28, 28))

    #Loading the data
    transform = transforms.Compose([transforms.CenterCrop(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)
                                    ])
    
    #Train dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #Dataloader used to shuffle and create batch
    train_loader   = torch.utils.data.DataLoader(mnist_trainset, batch_size=256, shuffle=True)
    #Test dataset
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader   = torch.utils.data.DataLoader(mnist_testset, batch_size=256, shuffle=True)
    
    #Check data
    print_data_info(train_loader, test_loader)

    #Training
    model = AE
    train_loader = train_loader
    test_loader  = test_loader
    nof_epochs = 5
    learning_rate = 0.001
    optimizer = torch.optim.Adam(AE.parameters(), lr = 0.001)
    criterion = torch.nn.MSELoss()
    file_path_save_model = '/content/checkpoint.pth'

    train_model(model, train_loader, test_loader, 
                nof_epochs, optimizer, learning_rate, criterion, 
                file_path_save_model)
    
    #Display results
    plot_results(AE, test_loader, device)