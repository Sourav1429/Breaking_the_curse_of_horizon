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
2) estimate_reward.py: In this file, 
