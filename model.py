#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:21:44 2022

@author: dldou
"""

import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        #Encoder part
        self.encoder_part = nn.Sequential(
            #Conv2d
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, bias=True),
            nn.ReLU(),
            #MaxPooling
            nn.MaxPool2d(kernel_size=2, stride=2),
            #Conv2d
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, bias=True),
            nn.ReLU(),
        )

        #Decoder part
        self.decoder_part = nn.Sequential(
            #DeConv2d
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=2, bias=True),
            nn.ReLU(),
            #Upsample
            nn.Upsample(size=(26,26)),
            #Conv2d
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=2),
        )

    def forward(self,x):

        #Encoder
        x = self.encoder_part(x)
        #Decoder
        x = self.decoder_part(x)

        return x