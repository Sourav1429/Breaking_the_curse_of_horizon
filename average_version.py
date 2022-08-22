import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import product
# from math import comb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class w(nn.Module):
    def __init__(self, input_size, output_size):
        super(w, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(self.input_size, self.output_size, bias=False)

    def forward(self, state):
        output = torch.exp(self.linear1(state)) #To ensure that the outputs are always positive. giving Relu will cause problems.
        return output

behaviour_policy = {}
behaviour_policy[0] = [0.5,0.5]
behaviour_policy[1] = [1,0]

transition_probability = {}
transition_probability[0] = [[0, 1],[1, 0]]
transition_probability[1] = [[1, 0],[0, 1]]

initial = {}
initial[0] = 1
initial[1] = 0

W = w(2, 1).to(device)
optimizerW = optim.Adam(W.parameters(),lr=0.1)

for iter in range(5000):

    state = torch.FloatTensor([0,1]).to(device)
    print(W(state))

    state = torch.FloatTensor([1,0]).to(device)
    print(W(state))
    
    print('****************************')

    W_loss = 0
    State1 = []
    State2 = []
    W_state1 = []
    W_state2 = []
    W_next_state1 = []
    W_next_state2 = []
    Beta1 = []
    Beta2 = []
    K = []
    Z_w_state = 0
    count = 0

    training_data = []
    a = random.random()
    if a < initial[0]:
        state = 0
    else:
        state = 1
    for i in range(50):
        b = random.random()
        if b < behaviour_policy[state][0]:
            action = 0
        else:
            action = 1
        c = random.random()
        if c < transition_probability[action][state][0]:
            next_state = 0
        else:
            next_state = 1
        
        training_data.append([state,action,next_state])

        state = next_state

    batch = []
    for i in range(len(training_data)):
        d = random.random()
        if d <= 0.5:
            batch.append(training_data[i])

    pairs = list(product(batch, repeat=2))

    for pair in pairs:
        sample1 = pair[0]
        sample2 = pair[1]
        w_state1 = 0
        w_state2 = 0
        beta1 = 0
        beta2 = 0
        

        if sample1[0] == 0:
            state1 = torch.FloatTensor([0,1]).to(device)
            w_state1 = W(state1)
            if sample1[1] == 0:
                beta1 = 2
            if sample1[1] == 1:
                beta1 = 0

        if sample1[0] == 1:
            state1 = torch.FloatTensor([1,0]).to(device)
            w_state1 = W(state1)
            if sample1[1] == 0:
                beta1 = 1
            if sample1[1] == 1:
                beta1 = 0

        if sample2[0] == 0:
            state2 = torch.FloatTensor([0,1]).to(device)
            w_state2 = W(state2)
            if sample2[1] == 0:
                beta2 = 2
            if sample2[1] == 1:
                beta2 = 0

        if sample2[0] == 1:
            state2 = torch.FloatTensor([1,0]).to(device)
            w_state2 = W(state2)
            if sample2[1] == 0:
                beta2 = 2
            if sample2[1] == 1:
                beta2 = 0

        if sample1[2] == 0:
            next_state1 = torch.FloatTensor([0,1]).to(device)
            w_next_state1 = W(next_state1)

        if sample1[2] == 1:
            next_state1 = torch.FloatTensor([1,0]).to(device)
            w_next_state1 = W(next_state1)

        if sample2[2] == 0:
            next_state2 = torch.FloatTensor([0,1]).to(device)
            w_next_state2 = W(next_state2)

        if sample2[2] == 1:
            next_state2 = torch.FloatTensor([1,0]).to(device)
            w_next_state2 = W(next_state2)

        if sample1[2] != sample2[2]:
            k = 0
        else:
            k = 1

        State1.append(sample1[0])
        State2.append(sample2[0])
        W_state1.append(w_state1)
        W_state2.append(w_state2)
        W_next_state1.append(w_next_state1)
        W_next_state2.append(w_next_state2)
        Beta1.append(beta1)
        Beta2.append(beta2)
        K.append(k)

#Computation of normalization constant below
        
    for i in range(len(batch)):
        if batch[i][0] == 0:
            s = torch.FloatTensor([0,1]).to(device)
            w = W(s)
            Z_w_state += w
            
        elif batch[i][0] == 1:
            s = torch.FloatTensor([1,0]).to(device)
            w = W(s)
            Z_w_state += w          
            
    Z_w_state /= len(batch)
    

    for i in range(len(State1)):
        W_loss += (Beta1[i]*(W_state1[i]/Z_w_state) - (W_next_state1[i]/Z_w_state))*(Beta2[i]*(W_state2[i]/Z_w_state) - (W_next_state2[i]/Z_w_state))*K[i]

    W_loss = W_loss
    optimizerW.zero_grad()
    W_loss.backward()
    optimizerW.step()
    optimizerW.zero_grad()

