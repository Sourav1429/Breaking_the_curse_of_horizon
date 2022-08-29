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
'''
    The class weights is created to define the neural network structure.
    Inputs:
    -------
    input_size  : number of input head perceptrons. Basically is equal to number of states nS.
    output_size : number of perceptrons in the output layer.
'''
class weights(nn.Module):
    def __init__(self,input_size,output_size):
        super(weights,self).__init__()
        self.input_size = input_size;
        self.output_size = output_size;
        self.linear1 = nn.Linear(self.input_size, self.output_size, bias=False)
    '''
        forward(): We accept a state 's' as input. Then we convert this into one hot encoding which is accomplished by first two lines.
        Further we convert this one_hot vector 's' into pytorch tensor and then pass it through the network to obtain a output which is returned 
    '''
    def forward(self,state):
        s = np.zeros(self.input_size);
        #print(state,end='===>');
        s[state] = 1;
        state = torch.FloatTensor(s).to(device)
        #print(state);
        output = torch.exp(self.linear1(state)) #To ensure that the outputs are always positive. giving Relu will cause problems.
        return output

