#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:21:52 2022

@author: dldou
"""

def print_data_info(train_loader, test_loader):
    """
        Function to print useful info about data
    """
    
    print("Nof train samples: ", len(train_loader.dataset))
    print("- samples' size: ", train_loader.dataset[0][0].shape)
    print("- batch size: ", train_loader.batch_size)
    print("\n")
    print("Nof test samples: ", len(test_loader.dataset))
    print("- samples' size: ", test_loader.dataset[0][0].shape)
    print("- batch size: ", test_loader.batch_size)