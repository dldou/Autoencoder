#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:21:53 2022

@author: dldou
"""

import torch
from torchsummary import summary




if __name__=="__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Instanciation of the model
    AE = AutoEncoder()
    AE.to(device)
    summary(AE, (1, 28, 28))

    #Training
    model = AE
    train_loader = train_loader
    test_loader  = test_loader
    nof_epochs = 5
    learning_rate = 0.001
    optimizer = torch.optim.Adam(AE.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    file_path_save_model = '/content/checkpoint.pth'

    train_model(model, train_loader, test_loader, 
                nof_epochs, optimizer, learning_rate, criterion, 
                file_path_save_model)
    
    #Inference
    
    #Display results