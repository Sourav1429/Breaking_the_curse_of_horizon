#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:25:45 2022

@author: Sourav
"""
import numpy as np
from transition_simulator import Simulate_episode

#policy = np.array([[1,0],[1,0],[0,1],[0,1]]);
policy = np.array([[[1,0],[1,0]],
                   [[0.5,0.5],[1,0.3]]]);
T = 100000
nS = 2
nA = 2;
rep_cost = 0.7
state = 0;
for j in policy:
    obj = Simulate_episode(T,nS,nA,rep_cost,j,state);
    
    data,_ = obj.play_episode();
    #print(data['state']);
    state_distribution = np.zeros(nS);
    reward_distribution = np.zeros(nS);
    rew=0
    for i in range(T):
        state_distribution[data['state'][i]]+=1;
        reward_distribution[data['state'][i]]+=data['reward'][i];
        rew=rew+data['reward'][i]
    state_distribution = state_distribution/T;
    reward_distribution = reward_distribution/T;
    expected_reward = rew/T;
    print(j,"===========================>")
    print("State distribution:",state_distribution);
    print("Reward distribution",reward_distribution);
    print("Expected reward directly",expected_reward);
    print(np.sum(state_distribution*reward_distribution));
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
