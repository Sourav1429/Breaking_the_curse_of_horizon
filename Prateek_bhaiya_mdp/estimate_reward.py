#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:33:54 2022

@author: Sourav
"""
import numpy as np
import weights_parameterization
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
A class created named as est_rew. The object of est_rew takes the following as input
Inputs:
--------
state_distribution: This is a 'list' containing the probability distribution of each state.
target_policy    : We accept a target_policy for our MDP. The target_policy is accepted inh the form of [0,0,1,1] where 0 stands for action '0' 
and 1 stands for action '1'. Next we convert it into probability distribution by converting it into one_hot vector. 0 - gets converted as [1,0] 
and 1 - gets converted as [0,1].
behaviour_policy: We accept a behaviour_policy. The behaviour policy is given in the format as [[0.6,0.4],[0.2,0.8]]. So each row consists of
probability distribution of taking an action in a state. sum(behaviour[s][:])=1 for all 'a'.
data           : This is dictionary consisting of the simulated data. The dictionary looks like{'state':[],'action':[],'next_state':[],'reward':[]}
such that the keys of the dictionary is 'state', 'action', 'next_state' and 'reward'. Each key stores a list of episodic data.
nS            : stores the number of states.
nA           : stores the number of actions.
'''

class est_rew:
    def __init__(self,state_distribution,target_policy,behaviour_policy,data,nS=4,nA=2):
        self.state_distribution = state_distribution  #list of distribution of all states in state space
        self.target_policy = self.one_hot(target_policy,nS,nA); #to store target_policy that we try to achieve
        self.behaviour_policy = behaviour_policy; #to store the behaviour policy that we follow
        self.data = data;# dictionary {'state':[list of current state] ,'action':[list of actions taken],'next_state':[list of the next states],'reward':[reward observed for taking such transition and performing this action]}
        self.n = len(data['state'])#total number of samples
        self.weight_obj = weights_parameterization.weights(nS, 1)
    # one_hot(): converts a number 'i' into a vector of nS dimension such that its ith element is 1 and rest elements are 0.
    def one_hot(self,policy,nS,nA):
        target_policy=np.zeros((nS,nA));
        #print(policy);
        policy = policy.astype(int)
        print(policy);
        for i in range(nS):
            #print(policy[int(i)]);
            target_policy[i][policy[int(i)]]=1;
        return target_policy
    # find_reward(): this fuction finds the reward as for all states 's' (state_distribution[s] * beta(s,a) *r)/(state_distribution[s] * beta[s,a])
    def find_reward(self):
        num,den = 0,0
        for i in range(self.n):
            s,a,s_next,r = self.data['state'][i],self.data['action'][i],self.data['next_state'][i],self.data['reward'][i]
            num = num + self.state_distribution[s] * (self.target_policy[s,a]/self.behaviour_policy[s,a]) * r;
            den = den + self.state_distribution[s] * (self.target_policy[s,a]/self.behaviour_policy[s,a]);
        return num/den;