#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 18:27:08 2022

@author: Sourav
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class weights(nn.Module):
    def __init__(self,input_size,output_size):
        super(weights,self).__init__()
        self.input_size = input_size;
        self.output_size = output_size;
        self.linear1 = nn.Linear(self.input_size, self.output_size, bias=False)
    def forward(self,state):
        s = np.zeros(self.input_size);
        s[state] = 1;
        state = torch.FloatTensor(s).to(device)
        output = torch.exp(self.linear1(state)) #To ensure that the outputs are always positive. giving Relu will cause problems.
        return output
'''l_r=0.01
obj = weights(4,1).to(device);
optimizerW = optim.Adam(obj.parameters(),lr=l_r);'''

