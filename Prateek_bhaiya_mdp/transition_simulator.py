#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:40:16 2022

@author: Sourav
"""
import numpy as np
from Machine_replacement_env_create import Machine_Replacement
from find_target_policy import Target_Policy
class Simulate_episode:
    def __init__(self,T,nS,nA,rep_cost,behaviour_policy,state):
        self.nS = nS
        self.nA = nA;
        self.rep_cost = rep_cost;
        self.obj = Machine_Replacement(self.rep_cost,self.nS,self.nA);
        #self.P = self.obj.gen_probability();
        self.P = np.array([[[0,1],[1,0]],
                           [[1,0],[0,1]]])
        self.R = self.obj.gen_reward();
        self.T = T;
        self.behaviour_policy = behaviour_policy;
        self.state = state
        obj = Target_Policy(np.arange(self.nS), np.arange(self.nA), self.P, self.R, state);
        runs = 10;
        base_obj = obj.find_optimum_policy(runs, T)
        self.target_policy = base_obj[-1];
        print("Target_policy",self.target_policy);
    def play_episode(self):
        episode_data = {'state':[],'action':[],'next_state':[],'reward':[]}
        state = self.state;
        t=1;
        while(t<=self.T):
            #print(state);
            action = np.argmax(np.random.multinomial(1,self.behaviour_policy[state,:]));
            next_state = np.argmax(np.random.multinomial(1,self.P[action,state,:]));
            reward = self.R[action,state,next_state];
            episode_data['state'].append(state);
            episode_data['action'].append(action);
            episode_data['next_state'].append(next_state);
            episode_data['reward'].append(reward);
            state = next_state;
            t+=1;
        return (episode_data,self.target_policy);
'''T=10;
nS=4;
nA=2;
rep_cost=0.7;
behaviour_policy=np.array([[1,0],
                  [1,0],
                  [0,1],
                  [0,1]]);
state = 0;
obj = Simulate_episode(T, nS, nA, rep_cost, behaviour_policy, state)
print(obj.play_episode())'''