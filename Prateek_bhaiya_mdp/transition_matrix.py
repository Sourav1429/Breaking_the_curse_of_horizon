#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 13:33:18 2022

@author: Sourav
"""
import numpy as np
from transition_simulator import Simulate_episode
from Machine_replacement_env_create import Machine_Replacement


def onehot(policy,val,nS):
    pol = np.zeros((nS,nA));
    for i in range(nS):
        pol[i][policy[i]]=1;
    return pol;

def find_transition_matrix(nA,nS,T,policy,onehot_encode=1):
    if(onehot_encode==1):
        policy = onehot(policy,nA,nS)
    T_s_s_next = np.zeros((nS,nS));
    for s in range(nS):
        for s_next in range(nS):
            for a in range(nA):
                #print(s,s_next,a);
                #print(T[a,s,s_next]);
                T_s_s_next[s,s_next]+=T[a,s,s_next]*policy[s,a];
    return T_s_s_next;



nS = 2
nA = 2
rep_cost = 0.7
obj = Machine_Replacement(rep_cost,nS,nA);
#P = obj.gen_probability();
P = np.array([[[0,1],[1,0]],
                   [[1,0],[0,1]]])
print(P);
#behaviour_policy=np.array([[0.6,0.4],[0.3,0.7],[0.4,0.6],[0.1,0.9]]);
#P=np.array([
 #   [[0, 1],[1, 0]],
 #   [[1, 0],[0, 1]]]);
#behaviour_policy=np.array([[0.5,0.5],[1,0]]);
'''policies = np.array([[0,0,0,0],
            [0,0,0,1],
            [0,0,1,1],
            [0,1,1,1],
            [1,1,1,1]]);'''
behaviour_policy = np.array([[0.5,0.5],[1,0]]);
policies = np.array([[0,0]]);

for i in policies:
    print(i,"=========>");
    P_policy=find_transition_matrix(nA,nS,P,i)
    print(P_policy);
    P_dash = np.append(P_policy - np.eye(nS),np.ones((nS,1)),axis=1);
    print(P_dash);
    P_last = np.linalg.pinv(np.transpose(P_dash))[:,-1]
    print(P_last)
    print("==============================================================");

print("Behaviour policy");
P_policy = find_transition_matrix(nA, nS, P, behaviour_policy,0);
print(P_policy);
P_dash = np.append(P_policy - np.eye(nS),np.ones((nS,1)),axis=1);
print(P_dash);
P_last = np.linalg.pinv(np.transpose(P_dash))[:,-1]
print(P_last)

