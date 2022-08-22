#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:43:00 2022

@author: Sourav
"""
import numpy as np
class Machine_Replacement:
    def __init__(self,rep_cost,nS=6,nA=2):
        self.nS = nS;
        self.nA = nA;
        self.cost = np.linspace(0.1, 0.99,nS);
        self.rep_cost = rep_cost;
    def gen_probability(self):
        self.P = np.zeros((self.nA,self.nS,self.nS));
        for i in range(self.nS):
            for j in range(self.nS):
                if(i<=j):
                    self.P[0,i,j]=(i+1)*(j+1);
                else:
                    continue;
            self.P[0,i,:]=self.P[0,i,:]/np.sum(self.P[0,i,:])
            self.P[1,i,0]=1;
        return self.P;
    def gen_reward(self):
        self.R=np.zeros((self.nA,self.nS,self.nS));
        for i in range(self.nS):
            self.R[0,i,:] = self.cost[i];
            self.R[1,i,0] = self.rep_cost+self.cost[0];
        return self.R;
'''nS = 4;
nA=2;
cost=np.linspace(0.1, 0.99,nS);
rep_cost=0.7
obj = Machine_Replacement(rep_cost,nS,nA);
print("Probability matrix");
print(obj.gen_probability());
print("Reward Matrix");
print(obj.gen_reward());          '''
