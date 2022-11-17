#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:21:52 2022

@author: dldou
"""

import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable


def saveModel(model, file_path):
    """
        Function to save model's parameters
    """
    torch.save(model.state_dict(), file_path)


def loadModel(model, file_path, device):
    """
        Function to load function when only the params have been saved
    """
    params = torch.load(file_path)
    model.load_state_dict(params)


def checkPoint_model(model, 
                     optimizer, loss, epoch,
                     file_path):
    """
        Function to save model's checkpoints
    """
    
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 
                file_path)

def load_checkPoint_model(model, optimizer, file_path, device):

    checkpoint = torch.load(file_path)

    #Loading
    model.load_state_dict(checkpoint['model_state_dict'], map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'], map_location=device)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss

def train_epoch(model, optimizer, criterion, 
                train_loader, 
                device):

    loss     = 0.0

    for i, (images, labels) in enumerate(train_loader, 0):
        
            #Data send to device + requires_grad=True
            images, labels = Variable(images.to(device)), Variable(labels.to(device))
            #Zero the gradient 
            optimizer.zero_grad()
            #Predictions 
            images_hat = model(images)
            images_hat.to(device)
            #Loss (CrossEntropy mustn't have the channel param)
            loss = criterion(images_hat.squeeze(), images.squeeze())
            #Upgrade the gradients (backpropagate) and the optimizer
            loss.backward()
            optimizer.step()

    return loss


def validate_epoch(model, optimizer, criterion, 
                   test_loader, epoch,
                   device):
    
    losses          = []
    loss            = 0.0
    accuracy        = 0.0
    nof_predictions = 0.0

    #Fasten the inference by setting every requires_grad to False
    with torch.no_grad():
        for data in test_loader:
            #Get data and send them to the device
            images, _ = data
            images    = images.to(device)
            #Run the model on the test set
            outputs = model(images)
            outputs.to(device)
            #Compute the loss on the batch
            loss = criterion(images.squeeze(), outputs.squeeze())
            losses.append(loss)
            #Update 
            nof_predictions += images.size(0)

    print("losses", losses)
    #Accuracy is the mean of loss on each batch
    accuracy = (sum(losses)/nof_predictions).item()

    return accuracy



def train_model(model, train_loader, test_loader, 
                nof_epochs, optimizer, learning_rate, criterion, 
                file_path_save_model):
    
    #Which device + sending model to its memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    model.to(device)

    best_accuracy = 0.0

    for epoch in range(nof_epochs):

        epoch_accuracy = 0.0
        epoch_loss     = 0.0

        #Training
        model.train()
        epoch_loss = train_epoch(model, optimizer, criterion, 
                                 train_loader, 
                                 device)
        
        #Validation
        model.eval()
        epoch_accuracy = validate_epoch(model, optimizer, criterion, 
                                        test_loader, epoch,
                                        device)
        
        print('Epoch', epoch+1,', accuracy: {:.4f} % \n'.format(epoch_accuracy))
        
        #Save model when best accuracy is beaten
        if epoch_accuracy > best_accuracy:
            #load_checkPoint_model(model, optimizer, file_path_save_model, device)
            saveModel(model, file_path_save_model)
            best_accuracy = epoch_accuracy

    return model