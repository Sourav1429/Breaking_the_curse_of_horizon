The folder consists of 8 files as mentioned below in the order of execution.
Files
--------
1) expected_reward_average_case.py :  This is the driver file. This file calls other self made modules such as "weight_parameterization", "transition simulator" and "estimate_reward".
            The module consists of 2 functions
  a) get_batch() :
  -----------------
  From the 'data' collected in by running the 'play_episode()' in 'Simulate_episode' class which is part of 'transition_simulator module'. This function takes the following 
  inputs.
  INPUT
  ------
  i) batch_size: Specification how many samples each batch should hold.
  ii) T : specifying for how many time steps our episode runs.
  
  Functionality:
  --------------
  This function first samples randomly one number out of the T numbers possible. Then following 'batch_size' number of samples it collects and forms the batch.
  
  Return :
  ---------
  This function finally returns the samples to the caller.
 b) get_w():
 --------------
 This function is responsible to pass the state through the network and get the state distribution estimate. This function takes the following inputs.
 INPUT:
 -------
   i) data: This is a dictionary consisting of the simulated data for 'T' instances. The keys are.

    I) state: store a list of the states visited by the simulator at each of the T instances.
    II) action: stores the actions take by the agent in the state 's' present in the corresponding list under key 'state' stored in the form of a list.
    III) next_state: The next state our agent travelled to after taking an action 'a' in state 's' in the corresponding list with keys 'action' and 'state' respectively stored in the form ofa list.
    IV) reward: The reward obtained at each step stored in the form ofa list.

   ii) weight_obj : an istance of the weight_parameterization class(mentioned later)
   iii) behaviour_policy: the behaviour policy followed, to calculate beta.
   iv) target_policy : the target policy that is aimed to achieve to calculate beta.
   v) pair: This a default parameter with default value = 0. When pair = 1 that means we are not passing pairs to the function. Just the simple samples are sent to the function
   in form of data accepted by the variable named 'data'. So the function just finds the estimated distribution of the state by passing through the network and then obtaining the return 
   in a variable called Z_w_state. Finally it sums all the values and returns the result.

   If pair = 0 . Then the data passed is after taking the cross product of the samples. From the pairs, it keeps a list of state visited, action taken, the next_state , reward obtained,
   the estaimated value from the neural network for state1(state in the 1st element of the pair) and state2(state in the 2nd elemnt of the pair). Also we keep track of the kernel value
   in a list named 'K'.

   RETURN:
   --------
   state1: list of current state in 1st element of pair
   state2: List of current state in the 2nd elemnt of pair
   w_state1: list of estimated value of current state in 1st elemet of pair
   w_state2: list of estimated value of current state in the 2nd element of the pair
   w_next_state1: list of estimated value of next_satte in the first element of the pair
   w_next_state2: list of estimated value of next_satte in the 2nd element of the pair
   beta1: importance sampling of the 1st element in pair
   beta2: importance sampling of the 2nd element in pair
   K: the kernel function values.

   Finally in this file, we define all the hyperaparameters, the toatl number of instances for which our episode will run(T), number of state(nS), number of actions(nA), 
   initial state(state), the behaviour policy, find the target policy and data from the play_episode() of transition_simulator module. Finally a loop is executed for 'T' times
   where, we form a batch from the simulated data, print the distribution update at after each update. Then we find the loss value (W_loss) and optimize by passing through Adam
  optimizer.
  After this we obtain the state distribution from the network by passing each state and use it to find the estimated reward value. To do that we create a list of state distribution
  of all the states stored in a listed named as 'state_distribution'. Along with the state distribution we pass other data like behaviour_policy, target_policy, data and the number of states(nS).
_______________________________________________________________________________________________________________________________________________________________________________________________________
2) estimate_reward.py: In this file, we find the estimated reward and compare it with the value obtained by simulating a long episode run and at the end taking the         expectation of the reward obtained.
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
    
    Functions:
    -----------
    a) one_hot(): This function is responsible for converting a state passed into a one_hot encoded vector by first creating a vector of all zero elemnts and then the       state passed is made '1'. Returns the one-hot encoded vector.
    b) find_reward(): This function accepts the data passed to the constructor and then find the expected reward by using the formula SUM(state_distribution(state_s) * beta(s,a)* reward)/SUM(state_distribution(state_s)*beta(s,a)).
    Finally return the expected reward value.
____________________________________________________________________________________________________________________________________________________________________
____________________
3) weights_parameterization.py: In this file we define a class named 'weights' where we specify the neural network structure. 
    Inputs:
    -------
    input_size  : number of input head perceptrons. Basically is equal to number of states nS.
    output_size : number of perceptrons in the output layer.
    
    Functions:
    -----------
    forward(): This function defines the forward propogation of the network. First it converts an ceepted state into its one-hot encoding form and then finally return the value obtained after propogating through the neural network.
______________________________________________________________________________________________________________________________________________________________________________________________
4) Machine_Replacement_create_env.py: This file is responsible to create our MDP. This is a cost based MDP. This file has a class named Machine_Replacement which take the following as input
     INPUTS:
     --------
     i) nS: The number of states for which our MDP is to be designed.
     ii) nA: The number of actions for which our MDP is to be designed.
