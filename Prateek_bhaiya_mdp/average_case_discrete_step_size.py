#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 18:27:08 2022

@author: Sourav
"""
import weights_parameterization #user made module
from transition_simulator import Simulate_episode #user made module and class
import numpy as np
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from estimate_reward import est_rew #user defined module and class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
  get_batch(): forms a batch. when random value less than equal to 0.5
  Input:
  ------
  batch_size : the size of the batch formed
  T         :  total size of the
'''
def get_batch(data,batch_size,T):
    i=1;
    j = np.random.choice(T);
    batch,reward = [],[];
    while(i<=batch_size):
        if np.random.random()<=0.5:
            batch.append([data['state'][j],data['action'][j],data['next_state'][j]])
            reward.append(data['reward'][j]);
            j=(j+1)%T;
            i+=1;
    return batch,reward
'''
    get_w(): 
'''
def get_w(data,weight_obj,behaviour_policy,target_policy,pair=0):
    if(pair==1):
        Z_wstate = 0;
        for i in range(len(data)):
            val=weight_obj(data[i][0]);
            Z_wstate+=val;
        return Z_wstate/len(batch);
    else:
        state1,state2,w_state1,w_state2,w_next_state1,w_next_state2,beta1,beta2 = list(),list(),list(),list(),list(),list(),list(),list();
        K=list();
        for i in range(len(data)):
            sample1 = data[i][0];
            sample2 = data[i][1];
            state1.append(sample1[0]);
            w_state1.append(weight_obj(sample1[0]));
            w_next_state1.append(weight_obj(sample1[2]));
            state2.append(sample2[0]);
            w_state2.append(weight_obj(sample2[0]));
            w_next_state2.append(weight_obj(sample2[2]));
            beta1.append((target_policy[sample1[0]]==sample1[1])/behaviour_policy[sample1[0],sample1[1]]);
            beta2.append((target_policy[sample2[0]]==sample2[1])/behaviour_policy[sample2[0],sample2[1]]);
            K.append(sample1[2]==sample2[2]);
        return (state1,state2,w_state1,w_state2,w_next_state1,w_next_state2,beta1,beta2,K);
            
T = 1000 # number of times episode is run
nS = 2 #enter number of states
nA = 2 #let this be 2 only
rep_cost = 0.7 #replacement cost
#behaviour_policy = np.array([[0.6,0.4],[0.3,0.7],[0.4,0.6],[0.1,0.9]]);
behaviour_policy = np.array([[0.5,0.5],[1,0]]);
state = 0
batch_size = 50
l_rate=0.1
weight_obj = weights_parameterization.weights(nS, 1)
optimizerW = optim.Adam(weight_obj.parameters(),lr=l_rate)

obj = Simulate_episode(T, nS, nA, rep_cost, behaviour_policy, state);
data,target_policy = obj.play_episode()
target_policy = np.array([0,0]);
print("Behaviour policy:",behaviour_policy);
print("Target_policy:",target_policy);
#input("Should we continue?");
for _ in tqdm(range(T)):
    for s in range(nS):
        print(weight_obj(s));
        #print("================");
    batch,reward = get_batch(data,batch_size,T);
    pairs = list(product(batch, repeat=2))
    state1,state2,w_state1,w_state2,w_next_state1,w_next_state2,beta1,beta2,K = get_w(pairs,weight_obj,behaviour_policy,target_policy);
    Z_w_state = get_w(batch, weight_obj, behaviour_policy, target_policy,1);
    #print(len(state1)," is the number of samples used after pairing");
    W_loss = 0;
    for i in range(len(state1)):
        W_loss += (beta1[i]*(w_state1[i]/Z_w_state) - (w_next_state1[i]/Z_w_state))*(beta2[i]*(w_state2[i]/Z_w_state) - (w_next_state2[i]/Z_w_state))*K[i]
    W_loss = W_loss/(2*batch_size);
    optimizerW.zero_grad()
    W_loss.backward()
    optimizerW.step()
    optimizerW.zero_grad()

##Now calculating reward

state_distribution=[];
for i in range(nS):
    w_state = weight_obj(i);
    w_state = w_state.detach().numpy()[0];
    state_distribution.append(w_state);
    
object_rew = est_rew(state_distribution,target_policy,behaviour_policy,data,nS);
print(object_rew.find_reward());