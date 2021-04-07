# AUTHOR: Farzan Memarian
"""
This function is based on cntr
"""
import random
import numpy as np
import math
import copy
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.optimizers import SGD, RMSprop

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys, os
from envs.env2 import MDP
# from src import transformer
from utils import utils
from pdb import set_trace
import time
from numpy import linalg
import pickle


# ------------ REWARD NETWORKS ------------
class net_MLP(nn.Module):

    def __init__(self,
                MDP,
                DFA,
                n_dim,
                FC1_dim = 60,
                FC2_dim = 30,
                out_dim=1):
        super().__init__()
        # an affine operation: y = Wx + b
        depth, _, _ = MDP.grid.size()
        self.fc1 = nn.Linear(depth*n_dim*n_dim+DFA.n_states+4, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self._num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class net_att(nn.Module):
    pass

class net_CNN_non_local_r(nn.Module):

    def __init__(self,
            env,
            FC1_dim = 200,
            FC2_dim = 50,
            out_dim = 1):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.kernel_size = 3
        self.stride = 1
        self.in_dim = env.n_dim
        self.num_kernels_lay1 = 12
        self.num_kernels_lay2 = 24
        self.conv1 = nn.Conv2d(7, self.num_kernels_lay1, kernel_size=self.kernel_size, stride=self.stride)
        # self.bn1 = nn.BatchNorm2d(self.num_kernels_lay1)
        self.conv2 = nn.Conv2d(self.num_kernels_lay1, self.num_kernels_lay2, kernel_size=self.kernel_size, stride=self.stride)
        # self.bn2 = nn.BatchNorm2d(self.num_kernels_lay2)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        # an affine operation: y = Wx + b
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (self.kernel_size - 1) - 1) // stride  + 1
        self.final_conv_dim = conv2d_size_out(conv2d_size_out(self.in_dim, self.kernel_size,
                                self.stride)-1,self.kernel_size, self.stride)-1
        self.final_pool_dim = 5 # this needs to be made automatic
        self.fc1 = nn.Linear(self.num_kernels_lay2 * self.final_conv_dim * self.final_conv_dim + 9, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, out_dim)
        self.softmax = nn.Softmax()

    def forward(self, x, dfa_states, actions):
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.append_dfa(x, dfa_states)
        x = self.append_action(x, actions)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x

    def append_dfa(self, x, dfa_states):
        x = torch.cat([x, dfa_states], 1)
        return x

    def append_action(self, x, actions):
        x = torch.cat([x, actions], 1)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# ------------ POLICY NETWORKS ------------
class net_MLP_pol_v1(nn.Module):

    def __init__(self,
                MDP,
                n_dim = 3,
                FC1_dim = 100,
                FC2_dim = 40,
                out_dim=4):
        super().__init__()
        # an affine operation: y = Wx + b
        depth, n_dim, n_dim = MDP.grid.size()
        self.fc1 = nn.Linear(12*n_dim*n_dim, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self._num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def append_dfa(self, x, dfa_states):
        x = torch.cat([x, dfa_states], 1)
        return x


    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class net_MLP_pol_v2(nn.Module):

    def __init__(self,
                MDP,
                n_dim = 3,
                FC1_dim = 100,
                FC2_dim = 40,
                out_dim=4):
        super().__init__()
        # an affine operation: y = Wx + b
        depth, n_dim, n_dim = MDP.grid.size()
        self.fc1 = nn.Linear(7*n_dim*n_dim+5, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, out_dim)

    def forward(self, x, dfa_states):
        x = x.view(-1, self._num_flat_features(x))
        x = self.append_dfa(x, dfa_states)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def append_dfa(self, x, dfa_states):
        x = torch.cat([x, dfa_states], 1)
        return x


    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class net_MLP_pol_ind(nn.Module):

    def __init__(self,
                MDP,
                n_dim = 3,
                FC1_dim = 100,
                FC2_dim = 40,
                out_dim=4):
        super().__init__()
        # an affine operation: y = Wx + b
        depth, n_dim, n_dim = MDP.grid.size()
        self.fc1 = nn.Linear(7*n_dim*n_dim, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self._num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def append_dfa(self, x, dfa_states):
        x = torch.cat([x, dfa_states], 1)
        return x


    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class net_MLP_pol_local(nn.Module):

    def __init__(self,
                MDP,
                DFA,
                n_dim = 5,
                FC1_dim = 60,
                FC2_dim = 30,
                out_dim=4):
        super().__init__()
        # an affine operation: y = Wx + b
        depth, _, _ = MDP.grid.size()
        self.fc1 = nn.Linear(depth*n_dim*n_dim+DFA.n_states, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self._num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class net_CNN_pol_v1(nn.Module):

    def __init__(self,
            env,
            FC1_dim = 100,
            FC2_dim = 40,
            out_dim=4):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.kernel_size = 2
        self.stride = 1
        self.in_dim = env.n_dim
        self.num_kernels_lay1 = 12
        self.num_kernels_lay2 = 24
        self.conv1 = nn.Conv2d(12, self.num_kernels_lay1, kernel_size=self.kernel_size, stride=self.stride)
        # self.bn1 = nn.BatchNorm2d(self.num_kernels_lay1)
        self.conv2 = nn.Conv2d(self.num_kernels_lay1, self.num_kernels_lay2, kernel_size=self.kernel_size, stride=self.stride)
        # self.bn2 = nn.BatchNorm2d(self.num_kernels_lay2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # an affine operation: y = Wx + b
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (self.kernel_size - 1) - 1) // stride  + 1
        self.final_conv_dim = conv2d_size_out(conv2d_size_out(self.in_dim, self.kernel_size, self.stride),self.kernel_size, self.stride)
        self.final_pool_dim = 5 # this needs to be made automatic
        self.fc1 = nn.Linear(self.num_kernels_lay2 * self.final_conv_dim * self.final_conv_dim, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, out_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x

    def append_dfa(self, x, dfa_states):
        x = torch.cat([x, dfa_states], 1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class net_CNN_pol_v2(nn.Module):

    def __init__(self,
            mdp,
            dfa,
            FC1_dim = 100,
            FC2_dim = 40,
            out_dim=4):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.kernel_size = 2
        self.stride = 1
        self.in_dim = mdp.n_dim
        self.num_kernels_lay1 = 12
        self.num_kernels_lay2 = 24
        self.conv1 = nn.Conv2d(7, self.num_kernels_lay1, kernel_size=self.kernel_size, stride=self.stride)
        # self.bn1 = nn.BatchNorm2d(self.num_kernels_lay1)
        self.conv2 = nn.Conv2d(self.num_kernels_lay1, self.num_kernels_lay2, kernel_size=self.kernel_size, stride=self.stride)
        # self.bn2 = nn.BatchNorm2d(self.num_kernels_lay2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # an affine operation: y = Wx + b
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (self.kernel_size - 1) - 1) // stride  + 1
        self.final_conv_dim = conv2d_size_out(conv2d_size_out(self.in_dim, self.kernel_size, self.stride),self.kernel_size, self.stride)
        self.final_pool_dim = 5 # this needs to be made automatic
        self.fc1 = nn.Linear(self.num_kernels_lay2 * self.final_conv_dim * self.final_conv_dim + dfa.n_states, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, out_dim)
        self.softmax = nn.Softmax()

    def forward(self, x, dfa_states):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.append_dfa(x, dfa_states)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x

    def append_dfa(self, x, dfa_states):
        x = torch.cat([x, dfa_states], 1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class net_CNN_pol_ind(nn.Module):

    def __init__(self,
            mdp,
            dfa,
            FC1_dim = 100,
            FC2_dim = 40,
            out_dim=4):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.kernel_size = 2
        self.stride = 1
        self.in_dim = mdp.n_dim
        self.num_kernels_lay1 = 12
        self.num_kernels_lay2 = 24
        self.conv1 = nn.Conv2d(7, self.num_kernels_lay1, kernel_size=self.kernel_size, stride=self.stride)
        # self.bn1 = nn.BatchNorm2d(self.num_kernels_lay1)
        self.conv2 = nn.Conv2d(self.num_kernels_lay1, self.num_kernels_lay2, kernel_size=self.kernel_size, stride=self.stride)
        # self.bn2 = nn.BatchNorm2d(self.num_kernels_lay2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # an affine operation: y = Wx + b
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (self.kernel_size - 1) - 1) // stride  + 1
        self.final_conv_dim = conv2d_size_out(conv2d_size_out(self.in_dim, self.kernel_size, self.stride),self.kernel_size, self.stride)
        self.final_pool_dim = 5 # this needs to be made automatic
        # self.fc1 = nn.Linear(self.num_kernels_lay2 * self.final_conv_dim * self.final_conv_dim, FC1_dim)
        self.fc1 = nn.Linear(self.num_kernels_lay2 * self.final_pool_dim * self.final_pool_dim, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, out_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x

    def append_dfa(self, x, dfa_states):
        x = torch.cat([x, dfa_states], 1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# ------------- IRL AGENTS ----------------
class agent_RL_parent:
    def __init__(self,
                envs,
                init_vars,
                agent_mode
                ):

        self.thresh_Q_optimal = init_vars["thresh_Q_optimal"]
        self.thresh_y_optimal = init_vars["thresh_y_optimal"]

        self.device = torch.device(init_vars["device"])
        self.envs = envs
        self.MDP, self.DFA, self.PA, self.DFA_GT, self.PA_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
        self.gamma=init_vars["gamma"]
        self.n_actions = 4

        # replay memories and transitions
        self.Transition = init_vars["Transition"]

        # Q and pi and y optimal, only phi_theta is initialized here among theta based vars
        self.Q_optimal = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.V_optimal = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        self.pi_theta, self.pi_optimal_soft, self.pi_optimal_greedy = self._pi_init()

        self.visitation_counts = None # to be filled in by calling demo_visitation_calculator()
        self.visitation_counts_arr = None
        self.visitation_counts_arr_all = None
        self.visitation_counts_all = None

    def demo_visitation_calculator(self, num_trajs_used, trajs):
        # fills in self.visitation_counts, to be called from the  main function
        self.visitation_counts_arr = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.visitation_counts = {}
        self.visitation_counts_arr_all = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.visitation_counts_all = {}

        # trajs = pickle.load(open(trajs_Address  + "trajectories.pkl", "rb" ) )
        for traj in trajs[:num_trajs_used]:
            for s_a_pair in traj[:-1]:
                i,j,k,l = s_a_pair[1][0], s_a_pair[1][1], s_a_pair[2], s_a_pair[3]

                self.visitation_counts_arr[i,j,k,l] += 1
                if (i,j,k,l) in self.visitation_counts.keys():
                    self.visitation_counts[(i,j,k,l)] += 1
                else:
                    self.visitation_counts[(i,j,k,l)] = 1

        # trajs = pickle.load(open(trajs_Address  + "trajectories.pkl", "rb" ) )
        for traj in trajs:
            for s_a_pair in traj[:-1]:
                i,j,k,l = s_a_pair[1][0], s_a_pair[1][1], s_a_pair[2], s_a_pair[3]
                self.visitation_counts_arr_all[i,j,k,l] += 1
                if (i,j,k,l) in self.visitation_counts_all.keys():
                    self.visitation_counts_all[(i,j,k,l)] += 1
                else:
                    self.visitation_counts_all[(i,j,k,l)] = 1

    def select_optimal_action_soft(self):
        # softmax policy
        # agent_env_state = utils.agent_env_state(current_state, env_state)
        mdp_state, dfa_state = self.PA.return_current_state()
        i,j = mdp_state[0], mdp_state[1]
        k = dfa_state
        probabilities = self.pi_optimal_soft[i,j,k,:]
        action_idxs = np.arange(4)
        action_idx = np.random.choice(action_idxs, 1, p=probabilities)[0]
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def select_action_soft_pi_theta(self):
        # softmax policy
        # agent_env_state = utils.agent_env_state(current_state, env_state)
        mdp_state, dfa_state = self.PA.return_current_state()
        i,j = mdp_state[0], mdp_state[1]
        k = dfa_state
        probabilities = self.pi_theta[i,j,k,:]
        action_idxs = np.arange(4)
        action_idx = np.random.choice(action_idxs, 1, p=probabilities)[0]
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def select_optimal_action_greedy(self):
        # greedy policy directry from Q function
        mdp_state, dfa_state = self.PA.return_current_state()
        i,j = mdp_state[0], mdp_state[1]
        k = dfa_state
        action_idx = np.argmax(self.Q_theta[i,j,k,:])
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def random_action_selection(self):
        # print("random action selected")
        # i = self.MDP.current_state[0,0].item()
        # j = self.MDP.current_state[0,1].item()
        # allowable_action_idxs = self.MDP.allowable_action_idxs[i,j]
        all_actions = self.MDP.all_actions
        action_idx = int(np.random.choice(np.arange(4)))
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def _calc_Q_optimal(self,thresh):
        """

        Summary: value iteration using dynamic programming based on bellman equation, asynchronous updates
        find the optimal Q based on complete knowledge of task specification

        the Q function is manually forced to be zero at DFA states 3 and 4 cause they are terminal.
        The way this has been done is by not updating them from their initial value of zero

        output: it doesnt return anything, but it will update selt.Q_optimal
        """
        delta = thresh + 1
        counter=0
        while delta > thresh:
            counter+=1
            old_Q = copy.deepcopy(self.Q_optimal)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.DFA.non_terminal_states: # for accepting states, the q function is zero
                        for l in range(4): # for the 4 actionss
                            ip,jp,kp,reward = self.PA.transition_table[(i,j,k,l)]
                            self.Q_optimal[i,j,k,l] = reward + self.gamma * math.log(np.sum(np.exp(self.Q_optimal[ip,jp,kp,:])))

            delta_array = np.abs(self.Q_optimal - old_Q)
            delta = np.mean(delta_array) # one norm

        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                # for k in self.DFA.non_terminal_states: # for k=3 and k=4, Q function is zero
                for k in range(self.DFA.n_states):
                    self.V_optimal[i,j,k] = math.log(np.sum(np.exp(self.Q_optimal[i,j,k,:])))

    def calc_Q_theta(self, thresh):
        # dynamic programming based on bellman equation, in other words, policy evaluation using DP
        # update Q_theta based on the current value of theta and the current reward function

        if self.reward_input_batch is None:
            self.reward_input_batch = self.reward_net_input_batch_method()

        start = time.time()
        with torch.no_grad():
            reward = self.reward_net_method(self.reward_input_batch)
        end = time.time()
        reward_net_time = end-start
        reward = reward.view((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        delta = thresh + 1 # initialize Theta such that the loop starts running
        counter = 0
        while delta > thresh:
            counter += 1
            old_Q_theta = copy.deepcopy(self.Q_theta)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.DFA.non_terminal_states:
                        for l in range(4): # for the 4 actions
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            self.Q_theta[i,j,k,l] = reward[i,j,k,l] + self.gamma * math.log(np.sum(np.exp(self.Q_theta[ip,jp,kp,:])))

            delta_array = np.abs(self.Q_theta - old_Q_theta)
            delta = np.mean(delta_array) # 1 norm


        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                # for k in self.DFA.non_terminal_states: # for k=3 and k=4, Q function is zero
                for k in range(self.DFA.n_states):
                    self.V_theta[i,j,k] = math.log(np.sum(np.exp(self.Q_theta[i,j,k,:])))


        return counter, reward_net_time

    def calc_Q_theta_for_base_eval(self, thresh):
        # dynamic programming based on bellman equation, in other words, policy evaluation using DP
        # update Q_theta based on the current value of theta and the current reward function
        if self.reward_input_batch is None:
            self.reward_input_batch = reward_net_input_batch_method()

        start = time.time()
        with torch.no_grad():
            reward = reward_net_method(self.reward_input_batch)
        end = time.time()
        reward_net_time = end-start
        reward = reward.view((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        delta = thresh + 1 # initialize Theta such that the loop starts running
        counter = 0
        while delta > thresh:
            counter += 1
            old_Q_theta = copy.deepcopy(self.Q_theta)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in range(self.DFA.n_states):
                        for l in range(4): # for the 4 actions
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            self.Q_theta[i,j,k,l] = reward[i,j,0,l] + self.gamma * math.log(np.sum(np.exp(self.Q_theta[ip,jp,kp,:])))

            delta_array = np.abs(self.Q_theta - old_Q_theta)
            delta = np.mean(delta_array) # 1 norm
        return counter, reward_net_time

    def _pi_init(self):
        pi_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        pi_optimal_soft = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        pi_optimal_greedy = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        return pi_theta, pi_optimal_soft, pi_optimal_greedy

    def _calc_pi_optimal(self):
        Q_sum_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        # Q_sum_reduced = np.sum(np.exp(self.Q_optimal),3)
        for ind in range(4):
            Q_sum_act[:,:,:,ind] = self.V_optimal
        self.pi_optimal_soft = np.exp(self.Q_optimal - Q_sum_act)
        self.pi_optimal_greedy = np.argmax(self.Q_optimal,3)

    def calc_pi_theta(self):
        Q_sum_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        # Q_max_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        # for ind in range(4):
        #     Q_max_act[:,:,:,ind] = np.max(self.Q_theta,3)
        # scaled_down_Q = self.Q_theta - Q_max_act
        # Q_sum_reduced = np.sum(np.exp(scaled_down_Q),3)
        # for ind in range(4):
        #     Q_sum_act[:,:,:,ind] = Q_sum_reduced

        for ind in range(4):
            Q_sum_act[:,:,:,ind] = self.V_theta
        self.pi_theta = copy.deepcopy(np.exp(self.Q_theta - Q_sum_act))

    def _calc_y_optimal(self, thresh):
        """
        summary: calculates y(s,q) using dynamic programming based on Eq. 15 in the writeup.
        It find y for pi_optimal_soft

        output: returns y for all (s,q)
        """

        delta = thresh+1
        counter = 0
        y_optimal = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        for q_accept in self.DFA.accepting_states:
            y_optimal[:,:,q_accept] = np.ones((self.MDP.n_dim, self.MDP.n_dim)) # y is always one for accepting states

        while delta > thresh:
            counter += 1
            old_y_optimal = copy.deepcopy(y_optimal)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.DFA.non_terminal_states: # we excluded non-terminal states
                        y_sum = 0
                        for l in range(4):
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            # if kp == 4:
                            #     set_trace()
                            y_sum += self.pi_optimal_soft[i,j,k,l] * y_optimal[ip,jp,kp]
                        y_optimal[i,j,k] = y_sum

            delta_array = np.abs(y_optimal - old_y_optimal)
            delta = np.mean(delta_array) # 1 norm

        return y_optimal

    def _calc_grad_R_theta(self):

        # SHOULD BE DOUBLE CHECKED 1234567891011
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                for k in range(self.DFA.n_states):
                    for l in range(4): # for the 4 actions
                        # ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                        reward_input = self.reward_net_input_method(i,j,k,l)
                        self.reward_net.zero_grad()
                        reward = self.reward_net_method(reward_input)
                        reward.backward()
                        start_pos = 0
                        for f in self.reward_net.parameters():
                            # SHOULD BE DOUBLE CHECKED 1234567891011
                            f_grad_data = f.grad.data
                            num_flat_features = self._num_flat_features(f_grad_data)
                            f_grad_data_flat = f_grad_data.view(-1, num_flat_features).cpu().numpy()
                            self.grad_R_theta[i,j,k,l,start_pos:start_pos+num_flat_features] = f_grad_data_flat[:]
                            start_pos += num_flat_features

    def calc_grad_Q_theta(self, thresh):
        # SHOULD BE DOUBLE CHECKED 1234567891011
        # based on Eq. 18, using DP
        # Possibly TO BE ENFORCED: the gradient of Q at dfa states 3 and 4 needs to be zero
        self._calc_grad_R_theta()
        delta = thresh + 1 # initialize Theta such that the loop starts running
        counter = 0
        while delta > thresh:
            counter += 1
            old_grad_Q_theta = copy.deepcopy(self.grad_Q_theta)

            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.DFA.non_terminal_states:
                        for l in range(4): # for the 4 actions
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            second_term = np.zeros((self.theta_size))
                            for lp in range(4):
                                second_term += self.gamma * self.pi_theta[ip,jp,kp,lp] * self.grad_Q_theta[ip,jp,kp,lp,:]
                            self.grad_Q_theta[i,j,k,l,:] = self.grad_R_theta[i,j,k,l,:] + second_term

            delta_array = np.abs(self.grad_Q_theta - old_grad_Q_theta)
            delta = np.mean(delta_array) # 1 norm

    def calc_grad_L_D_theta(self, num_trajs_used):
        """ TO BE DOUBLE CHECKED 1234567891011
        - based on eq.4 in the writeup
        - currently the second term is ignored, add it later
        """
        grad_L_D_theta = np.zeros(self.theta_size)
        for key in self.visitation_counts.keys():
            i,j,k,l = key # l corresponds to action, k corresponds to dfa_state, i,j are for mdp state
            # for dfa_state in self.DFA.accepting_states:
            #     assert k != dfa_state
            num = self.visitation_counts[key]
            second_term = np.zeros(self.theta_size)
            for action in range(4):
                second_term += self.pi_theta[i,j,k,action] * self.grad_Q_theta[i,j,k,action,:]
            grad_L_D_theta += num * (self.grad_Q_theta[i,j,k,l,:] - second_term)

        # this part is to get the thetas to be used for l2 regularization
        weights_flat = np.zeros(self.theta_size)
        start_pos = 0
        for weights in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            num_flat_features = self._num_flat_features(weights)
            weights = copy.deepcopy(weights.view(-1, num_flat_features).detach().cpu().numpy())
            weights_flat[start_pos:start_pos+num_flat_features] = weights[:]
            start_pos += num_flat_features

        grad_L_D_theta = grad_L_D_theta / num_trajs_used
        return grad_L_D_theta, 2 *  weights_flat

    def calc_y_theta(self,thresh):
        """
        summary: calculates y(s,q) using dynamic programming based on Eq. 15 in the writeup


        output: returns y for all (s,q)
        """

        delta = thresh + 1
        counter = 0
        adj_matrix = np.zeros((self.MDP.n_dim*self.MDP.n_dim-13,self.MDP.n_dim*self.MDP.n_dim-13))
        print(adj_matrix.shape)
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                if (i,j) == (4,0) or (i,j) == (4,1) or (i,j) == (4,2) or (i,j) == (4,6) or (i,j) == (4,7) or (i,j) == (4,8) or (i,j) == (6,3) or (i,j) == (6,4) or (i,j) == (6,5) or (i,j) == (7,3) or (i,j) == (7,4) or (i,j) == (7,5) or (i,j) == (6,3) or (i,j) == (8,4) or (i,j) == (8,5):
                    continue
                if i<=3:
                    cell_num = i*self.MDP.n_dim+j
                elif i==4:
                    cell_num = i*self.MDP.n_dim+j-3
                elif i==5:
                    cell_num = i*self.MDP.n_dim+j-6
                elif i==6 and j<=3:
                    cell_num = i*self.MDP.n_dim+j-6
                elif i==6 and j>5:
                    cell_num = i*self.MDP.n_dim+j-9
                elif i==7 and j<=3:
                    cell_num = i*self.MDP.n_dim+j-9
                elif i==7 and j>5:
                    cell_num = i*self.MDP.n_dim+j-12
                elif i==8 and j<=3:
                    cell_num = i*self.MDP.n_dim+j-12
                elif i==8 and j>5:
                    cell_num = i*self.MDP.n_dim+j-15
                for k in self.DFA.non_terminal_states: # we excluded state 4 as the y(s,q) is zero for that state
                    for l in range(4):
                        ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                        if kp == 1:
                            adj_matrix[cell_num,adj_matrix.shape[0]-2] = self.pi_theta[i,j,k,l]
                        elif kp == 2:
                            adj_matrix[cell_num,adj_matrix.shape[0]-1] = self.pi_theta[i,j,k,l]
                        elif ip<=3:
                            adj_matrix[cell_num,ip*self.MDP.n_dim+jp] = self.pi_theta[i,j,k,l]
                        elif ip==4:
                            adj_matrix[cell_num,ip*self.MDP.n_dim+jp-3] = self.pi_theta[i,j,k,l]
                        elif ip==5:
                            adj_matrix[cell_num,ip*self.MDP.n_dim+jp-6] = self.pi_theta[i,j,k,l]
                        elif ip==6 and jp<=3:
                            adj_matrix[cell_num,ip*self.MDP.n_dim+jp-6] = self.pi_theta[i,j,k,l]
                        elif ip==6 and jp>5:
                            adj_matrix[cell_num,ip*self.MDP.n_dim+jp-9] = self.pi_theta[i,j,k,l]
                        elif ip==7 and jp<=3:
                            adj_matrix[cell_num,ip*self.MDP.n_dim+jp-9] = self.pi_theta[i,j,k,l]
                        elif ip==7 and jp>5:
                            adj_matrix[cell_num,ip*self.MDP.n_dim+jp-12] = self.pi_theta[i,j,k,l]
                        elif ip==8 and jp<=3:
                            adj_matrix[cell_num,ip*self.MDP.n_dim+jp-12] = self.pi_theta[i,j,k,l]
                        elif ip==8 and jp>5:
                            adj_matrix[cell_num,ip*self.MDP.n_dim+jp-15] = self.pi_theta[i,j,k,l]
        print(max(abs(np.linalg.eigvals(adj_matrix))))
        # exit()
        y_theta_below = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        y_theta_above = np.ones((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        for q_accept in self.DFA.accepting_states:
            y_theta_below[:,:,q_accept] = np.ones((self.MDP.n_dim, self.MDP.n_dim)) # y is always one for accepting states
        for q_fail in self.DFA.failing_states:
            y_theta_above[:,:,q_fail] = np.zeros((self.MDP.n_dim, self.MDP.n_dim)) # y is always zero for failing states

        while delta > thresh:
            counter += 1
            # old_y_theta = copy.deepcopy(self.y_theta)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.DFA.non_terminal_states: # we excluded state 4 as the y(s,q) is zero for that state
                        y_sum_below = 0
                        y_sum_above = 0
                        for l in range(4):
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            y_sum_below += self.pi_theta[i,j,k,l] * y_theta_below[ip,jp,kp]
                            y_sum_above += self.pi_theta[i,j,k,l] * y_theta_above[ip,jp,kp]
                        y_theta_below[i,j,k] = y_sum_below
                        y_theta_above[i,j,k] = y_sum_above

            # if counter%1000 == 0:
            #     print(y_theta_above[:,:,0])

            delta_array = np.abs(y_theta_above - y_theta_below)
            delta = np.mean(delta_array)  # 1 norm
        self.y_theta =  copy.deepcopy(y_theta_below)

        # # self.y_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        # for q_accept in self.DFA.accepting_states:
        #     self.y_theta[:,:,q_accept] = np.ones((self.MDP.n_dim, self.MDP.n_dim)) # y is always one for accepting states
        # while delta > thresh:
        #     counter += 1
        #     old_y_theta = copy.deepcopy(self.y_theta)
        #     for i in range(self.MDP.n_dim):
        #         for j in range(self.MDP.n_dim):
        #             for k in self.DFA.non_terminal_states: # we excluded state 4 as the y(s,q) is zero for that state
        #                 y_sum = 0
        #                 for l in range(4):
        #                     ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
        #                     y_sum += self.pi_theta[i,j,k,l] * self.y_theta[ip,jp,kp]
        #                 self.y_theta[i,j,k] = y_sum
        #
        #     if counter%1000 == 0:
        #         print(self.y_theta[:,:,0])
        #
        #     delta_array = np.abs(self.y_theta - old_y_theta)
        #     delta = np.mean(delta_array)  # 1 norm

    def calc_y_theta_eval(self,thresh):
        """
        summary: calculates y(s,q) using dynamic programming based on Eq. 15 in the writeup


        output: returns y for all (s,q)
        """

        delta = thresh + 1
        counter = 0
        # self.y_theta_eval = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        for q_accept in self.DFA.accepting_states:
            self.y_theta_eval[:,:,q_accept] = np.ones((self.MDP.n_dim, self.MDP.n_dim)) # y is always one for accepting states

        while delta > thresh:
            counter += 1
            old_y_theta = copy.deepcopy(self.y_theta_eval)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.DFA.non_terminal_states:
                        y_sum = 0
                        for l in range(4):
                            ip,jp,kp = self.PA_GT.transition_table[(i,j,k,l)][:3]
                            y_sum += self.pi_theta[i,j,k,l] * self.y_theta_eval[ip,jp,kp]
                        self.y_theta_eval[i,j,k] = y_sum

            delta_array = np.abs(self.y_theta_eval - old_y_theta)
            delta = np.mean(delta_array)  # 1 norm

    def _get_size_theta(self):
        size = 0
        for f in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            dims = f.size()
            layer_size = 1
            for dim in dims:
                layer_size *= dim
            size += layer_size
        return size

    def calc_grad_pi_theta(self):
        # SHOULD BE DOUBLE CHECKED 1234567891011
        # based on eq. 17
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                for k in range(self.DFA.n_states):
                    for l in range(4): # for the 4 actions
                        second_term = np.zeros(self.theta_size)
                        for lp in range(4):
                            second_term += self.pi_theta[i,j,k,lp] * self.grad_Q_theta[i,j,k,lp,:]
                        self.grad_pi_theta[i,j,k,l,:] = self.pi_theta[i,j,k,l] * (self.grad_Q_theta[i,j,k,l,:] - second_term)

    def calc_grad_y_theta(self, thresh):
        # Enforce ---> make sure grad_y is zero at dfa_state 3 and 4
        # using eq.16, DP

        delta = thresh + 1
        counter = 0
        while delta > thresh:
            counter += 1
            old_grad_y_theta = copy.deepcopy(self.grad_y_theta)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.DFA.non_terminal_states: # exclude dfa states 3 and 4 as their gradient must stay zero
                        sum_action_term = np.zeros(self.theta_size)
                        for l in range(4): # for actions
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            sum_action_term += (self.pi_theta[i,j,k,l] * self.grad_y_theta[ip,jp,kp,:] +
                                                self.y_theta[ip,jp,kp] * self.grad_pi_theta[i,j,k,l,:])
                        self.grad_y_theta[i,j,k,:] = sum_action_term

            delta_array = np.abs(self.grad_y_theta - old_grad_y_theta)
            delta = np.mean(delta_array) # 1 norm

    def calc_grad_L_phi_theta(self):
        # 1234567891011 MUST BE FIXED
        # using eqs 14 and 16
        grad_L_phi_theta = np.zeros(self.theta_size)
        for state in self.MDP.non_imp_obj_idxs_all:
            grad_L_phi_theta += self.grad_y_theta[state[0],state[1],0]

        grad_L_phi_theta /= len(self.MDP.non_imp_obj_idxs_all)
        return grad_L_phi_theta

    def calc_L_D_theta(self, num_trajs_used):
        Q_max_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        for ind in range(4):
            Q_max_act[:,:,:,ind] = np.max(self.Q_theta,3)
        scaled_down_Q = self.Q_theta - Q_max_act
        L_D_theta = 0
        for key in self.visitation_counts_all.keys():
            i,j,k,l = key # l corresponds to action, k corresponds to dfa_state, i,j are for mdp state
            num = self.visitation_counts_all[key]
            second_term = 0
            for action in range(4):
                second_term += math.exp(scaled_down_Q[i,j,k,action])
            L_D_theta += num * (scaled_down_Q[i,j,k,l] - math.log(second_term))
        return L_D_theta / num_trajs_used

    def calc_L_phi_theta(self):
        # using eqs 14
        L_phi_theta = 0
        vals = []
        for state in self.MDP.non_imp_obj_idxs_all:
            L_phi_theta += self.y_theta[state[0],state[1],0]
            vals.append(self.y_theta[state[0],state[1],0])
        L_phi_theta /= len(self.MDP.non_imp_obj_idxs_all)
        min_y = min(vals)
        return L_phi_theta, min_y

    def calc_L_phi_theta_eval(self):
        # using eqs 14
        L_phi_theta = 0
        vals = []
        for state in self.MDP.non_imp_obj_idxs_all:
            L_phi_theta += self.y_theta_eval[state[0],state[1],0]
            vals.append(self.y_theta_eval[state[0],state[1],0])
        L_phi_theta /= len(self.MDP.non_imp_obj_idxs_all)
        min_y = min(vals)
        return L_phi_theta, min_y

    def _num_flat_features(self, x):
        size = x.size()  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def update_theta(self, grad_j):
        # SHOULD BE DOUBLE CHECKED 1234567891011
        # Update reward network parameters using full gradient descent
        with torch.no_grad():
            start_idx = 0
            for layer_w in self.reward_net.parameters():
                length = self._num_flat_features(layer_w)
                shape = layer_w.size()
                dims = []
                for dim in shape:
                    dims.append(dim)
                dims_tuple = tuple(dims)
                layer_w += self.lr * torch.tensor(grad_j[start_idx:start_idx+length].reshape(dims_tuple), dtype=torch.float32, device=self.device)
                start_idx += length

    def eval_reward(self):
        reward_store = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                for k in range(self.DFA.n_states):
                    for l in range(4): # for the 4 actions

                        reward_input = self.reward_net_input_method(i,j,k,l)
                        # instead of using the reward from transition_table, use reward_net
                        reward_store[i,j,k,l] = self.reward_net_method(reward_input)
        return reward_store

    def eval_task_performance(self, num_trajs_used, thresh_Q_theta, thresh_Y_theta, trajs):
        self.demo_visitation_calculator(num_trajs_used, trajs)
        self.calc_Q_theta(thresh=thresh_Q_theta)
        self.calc_pi_theta()
        self.calc_y_theta(thresh=thresh_Y_theta)
        L_phi, min_y =  self.calc_L_phi_theta()

        return (L_phi, min_y, copy.deepcopy(self.y_theta), copy.deepcopy(self.Q_theta),
            copy.deepcopy(self.pi_theta), copy.deepcopy(self.visitation_counts_arr),
            copy.deepcopy(self.pi_optimal_soft), copy.deepcopy(self.pi_optimal_greedy))

    def eval_task_performance_base(self, num_trajs_used, thresh_Q_theta, thresh_Y_theta, trajs):
        self.demo_visitation_calculator(num_trajs_used, trajs)
        self.calc_Q_theta_for_base_eval(thresh=thresh_Q_theta)
        self.calc_pi_theta()
        self.calc_y_theta(thresh=thresh_Y_theta)
        L_phi, min_y =  self.calc_L_phi_theta()

        return (L_phi, min_y, copy.deepcopy(self.y_theta), copy.deepcopy(self.Q_theta),
            copy.deepcopy(self.pi_theta), copy.deepcopy(self.visitation_counts_arr),
            copy.deepcopy(self.pi_optimal_soft), copy.deepcopy(self.pi_optimal_greedy))

    def eval_L_phi(self, num_trajs_used, thresh_Q_theta, thresh_Y_theta):
        self.calc_Q_theta(thresh=thresh_Q_theta)
        self.calc_pi_theta()
        self.calc_y_theta(thresh=thresh_Y_theta)
        self.calc_y_theta_eval(thresh=thresh_Y_theta)

        L_phi, min_y =  self.calc_L_phi_theta()
        L_phi_eval, min_y_eval =  self.calc_L_phi_theta_eval()

        return L_phi, min_y, L_phi_eval, min_y_eval

    def eval_L_phi_base(self, num_trajs_used, thresh_Q_theta, thresh_Y_theta):
        self.calc_Q_theta_for_base_eval(thresh=thresh_Q_theta)
        self.calc_pi_theta()
        self.calc_y_theta(thresh=thresh_Y_theta)
        L_phi, min_y =  self.calc_L_phi_theta()
        return L_phi, min_y

class agent_RL_parent_bits:
    def __init__(self,
                envs,
                init_vars,
                agent_mode
                ):

        self.device = torch.device(init_vars["device"])
        self.n_imp_objs = init_vars["n_imp_objs"]
        self.envs = envs
        self.MDP, self.DFA, self.PA, self.DFA_GT, self.PA_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
        self.gamma=init_vars["gamma"]
        self.n_actions = 4

        # replay memories and transitions
        self.Transition = init_vars["Transition"]

        # Q and pi and y optimal, only phi_theta is initialized here among theta based vars
        self.pi_theta, self.pi_optimal_soft, self.pi_optimal_greedy = self._pi_init()

        self.visitation_counts = None # to be filled in by calling demo_visitation_calculator()
        self.visitation_counts_arr = None
        self.visitation_counts_arr_all = None
        self.visitation_counts_all = None

    def demo_visitation_calculator(self, num_trajs_used, trajs):
        # fills in self.visitation_counts, to be called from the  main function
        self.visitation_counts_arr = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.visitation_counts = {}
        self.visitation_counts_arr_all = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.visitation_counts_all = {}

        # trajs = pickle.load(open(trajs_Address  + "trajectories.pkl", "rb" ) )
        for traj in trajs[:num_trajs_used]:
            for s_a_pair in traj[:-1]:
                i,j,k,l = s_a_pair[1][0], s_a_pair[1][1], s_a_pair[2], s_a_pair[3]

                self.visitation_counts_arr[i,j,k,l] += 1
                if (i,j,k,l) in self.visitation_counts.keys():
                    self.visitation_counts[(i,j,k,l)] += 1
                else:
                    self.visitation_counts[(i,j,k,l)] = 1

        # trajs = pickle.load(open(trajs_Address  + "trajectories.pkl", "rb" ) )
        for traj in trajs:
            for s_a_pair in traj[:-1]:
                i,j,k,l = s_a_pair[1][0], s_a_pair[1][1], s_a_pair[2], s_a_pair[3]
                self.visitation_counts_arr_all[i,j,k,l] += 1
                if (i,j,k,l) in self.visitation_counts_all.keys():
                    self.visitation_counts_all[(i,j,k,l)] += 1
                else:
                    self.visitation_counts_all[(i,j,k,l)] = 1



    def select_action_soft_pi_theta(self):
        # softmax policy
        # agent_env_state = utils.agent_env_state(current_state, env_state)
        mdp_state, dfa_state = self.PA.return_current_state()
        dfa_state = self.DFA.convert_dfa_state_to_int(dfa_state)
        i,j = mdp_state[0], mdp_state[1]
        k = dfa_state
        probabilities = self.pi_theta[i,j,k,:]
        action_idxs = np.arange(4)
        action_idx = np.random.choice(action_idxs, 1, p=probabilities)[0]
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def select_optimal_action_greedy(self):
        # greedy policy directry from Q function
        mdp_state, dfa_state = self.PA.return_current_state()
        dfa_state = self.DFA.convert_dfa_state_to_int(dfa_state)
        i,j = mdp_state[0], mdp_state[1]
        k = dfa_state
        action_idx = np.argmax(self.Q_optimal[i,j,k,:])
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def random_action_selection(self):
        # print("random action selected")
        # i = self.MDP.current_state[0,0].item()
        # j = self.MDP.current_state[0,1].item()
        # allowable_action_idxs = self.MDP.allowable_action_idxs[i,j]
        all_actions = self.MDP.all_actions
        action_idx = int(np.random.choice(np.arange(4)))
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def calc_Q_theta(self, thresh):
        # dynamic programming based on bellman equation, in other words, policy evaluation using DP
        # update Q_theta based on the current value of theta and the current reward function
        if self.reward_input_batch is None:
            self.reward_input_batch = self.reward_net_input_batch_method()

        start = time.time()
        with torch.no_grad():
            reward = self.reward_net_method(self.reward_input_batch)
        end = time.time()
        reward_net_time = end-start
        reward = reward.view((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        delta = thresh + 1 # initialize Theta such that the loop starts running
        counter = 0
        while delta > thresh:
            counter += 1
            old_Q_theta = copy.deepcopy(self.Q_theta)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in range(self.DFA.n_states):
                        for l in range(4): # for the 4 actions
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            self.Q_theta[i,j,k,l] = reward[i,j,k,l] + self.gamma * math.log(np.sum(np.exp(self.Q_theta[ip,jp,kp,:])))

            delta_array = np.abs(self.Q_theta - old_Q_theta)
            delta = np.mean(delta_array) # 1 norm

        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                # for k in self.DFA.non_terminal_states: # for k=3 and k=4, Q function is zero
                for k in range(self.DFA.n_states):
                    self.V_theta[i,j,k] = math.log(np.sum(np.exp(self.Q_theta[i,j,k,:])))


        return counter, reward_net_time

    # def calc_Q_theta_for_base_eval(self, thresh):
    #     # dynamic programming based on bellman equation, in other words, policy evaluation using DP
    #     # update Q_theta based on the current value of theta and the current reward function
    #     if self.reward_input_batch is None:
    #         self.reward_input_batch = reward_net_input_batch_method()

    #     start = time.time()
    #     with torch.no_grad():
    #         reward = reward_net_method(self.reward_input_batch)
    #     end = time.time()
    #     reward_net_time = end-start
    #     reward = reward.view((self.MDP.n_dim, self.MDP.n_dim, self.DFA.states, 4))
    #     delta = thresh + 1 # initialize Theta such that the loop starts running
    #     counter = 0
    #     while delta > thresh:
    #         counter += 1
    #         old_Q_theta = copy.deepcopy(self.Q_theta)
    #         for i in range(self.MDP.n_dim):
    #             for j in range(self.MDP.n_dim):
    #                 for k in range(self.DFA.n_states):
    #                     for l in range(4): # for the 4 actions
    #                         ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
    #                         self.Q_theta[i,j,k,l] = reward[i,j,0,l] + self.gamma * math.log(np.sum(np.exp(self.Q_theta[ip,jp,kp,:])))

    #         delta_array = np.abs(self.Q_theta - old_Q_theta)
    #         delta = np.mean(delta_array) # 1 norm
    #     return counter, reward_net_time

    def _pi_init(self):
        pi_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        pi_optimal_soft = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        pi_optimal_greedy = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        return pi_theta, pi_optimal_soft, pi_optimal_greedy


    def calc_pi_theta(self):
        Q_sum_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        # Q_max_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        # for ind in range(4):
        #     Q_max_act[:,:,:,ind] = np.max(self.Q_theta,3)
        # scaled_down_Q = self.Q_theta - Q_max_act
        # Q_sum_reduced = np.sum(np.exp(scaled_down_Q),3)
        # for ind in range(4):
        #     Q_sum_act[:,:,:,ind] = Q_sum_reduced

        for ind in range(4):
            Q_sum_act[:,:,:,ind] = self.V_theta
        self.pi_theta = copy.deepcopy(np.exp(self.Q_theta - Q_sum_act))

    def _calc_grad_R_theta(self):
        # SHOULD BE DOUBLE CHECKED 1234567891011
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                for k in range(self.DFA.n_states):
                    for l in range(4): # for the 4 actions
                        # ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                        reward_input = self.reward_net_input_method(i,j,k,l)
                        self.reward_net.zero_grad()
                        reward = self.reward_net_method(reward_input)
                        reward.backward()
                        start_pos = 0
                        for f in self.reward_net.parameters():
                            # SHOULD BE DOUBLE CHECKED 1234567891011
                            f_grad_data = f.grad.data
                            num_flat_features = self._num_flat_features(f_grad_data)
                            f_grad_data_flat = f_grad_data.view(-1, num_flat_features).cpu().numpy()
                            self.grad_R_theta[i,j,k,l,start_pos:start_pos+num_flat_features] = f_grad_data_flat[:]
                            start_pos += num_flat_features

    def calc_grad_Q_theta(self, thresh):
        # SHOULD BE DOUBLE CHECKED 1234567891011
        # based on Eq. 18, using DP
        # Possibly TO BE ENFORCED: the gradient of Q at dfa states 3 and 4 needs to be zero
        self._calc_grad_R_theta()
        delta = thresh + 1 # initialize Theta such that the loop starts running
        counter = 0
        while delta > thresh:
            counter += 1
            old_grad_Q_theta = copy.deepcopy(self.grad_Q_theta)

            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in range(self.DFA.n_states):
                        for l in range(4): # for the 4 actions
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            second_term = np.zeros((self.theta_size))
                            for lp in range(4):
                                second_term += self.gamma * self.pi_theta[ip,jp,kp,lp] * self.grad_Q_theta[ip,jp,kp,lp,:]
                            self.grad_Q_theta[i,j,k,l,:] = self.grad_R_theta[i,j,k,l,:] + second_term

            delta_array = np.abs(self.grad_Q_theta - old_grad_Q_theta)
            delta = np.mean(delta_array) # 1 norm

    def calc_grad_L_D_theta(self, num_trajs_used):
        """ TO BE DOUBLE CHECKED 1234567891011
        - based on eq.4 in the writeup
        - currently the second term is ignored, add it later
        """
        grad_L_D_theta = np.zeros(self.theta_size)
        for key in self.visitation_counts.keys():
            i,j,k,l = key # l corresponds to action, k corresponds to dfa_state, i,j are for mdp state
            # for dfa_state in self.DFA.accepting_states:
            #     assert k != dfa_state
            num = self.visitation_counts[key]
            second_term = np.zeros(self.theta_size)
            for action in range(4):
                second_term += self.pi_theta[i,j,k,action] * self.grad_Q_theta[i,j,k,action,:]
            grad_L_D_theta += num * (self.grad_Q_theta[i,j,k,l,:] - second_term)

        # this part is to get the thetas to be used for l2 regularization
        weights_flat = np.zeros(self.theta_size)
        start_pos = 0
        for weights in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            num_flat_features = self._num_flat_features(weights)
            weights = copy.deepcopy(weights.view(-1, num_flat_features).detach().cpu().numpy())
            weights_flat[start_pos:start_pos+num_flat_features] = weights[:]
            start_pos += num_flat_features

        grad_L_D_theta = grad_L_D_theta / num_trajs_used
        return grad_L_D_theta, 2 *  weights_flat

    def _get_size_theta(self):
        size = 0
        for f in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            dims = f.size()
            layer_size = 1
            for dim in dims:
                layer_size *= dim
            size += layer_size
        return size

    def calc_grad_pi_theta(self):
        # SHOULD BE DOUBLE CHECKED 1234567891011
        # based on eq. 17
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                for k in range(self.DFA.n_states):
                    for l in range(4): # for the 4 actions
                        second_term = np.zeros(self.theta_size)
                        for lp in range(4):
                            second_term += self.pi_theta[i,j,k,lp] * self.grad_Q_theta[i,j,k,lp,:]
                        self.grad_pi_theta[i,j,k,l,:] = self.pi_theta[i,j,k,l] * (self.grad_Q_theta[i,j,k,l,:] - second_term)

    def calc_L_D_theta(self, num_trajs_used):
        Q_max_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        for ind in range(4):
            Q_max_act[:,:,:,ind] = np.max(self.Q_theta,3)
        scaled_down_Q = self.Q_theta - Q_max_act
        L_D_theta = 0
        for key in self.visitation_counts_all.keys():
            i,j,k,l = key # l corresponds to action, k corresponds to dfa_state, i,j are for mdp state
            num = self.visitation_counts_all[key]
            second_term = 0
            for action in range(4):
                second_term += math.exp(scaled_down_Q[i,j,k,action])
            L_D_theta += num * (scaled_down_Q[i,j,k,l] - math.log(second_term))
        return L_D_theta / num_trajs_used

    def _num_flat_features(self, x):
        size = x.size()  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def update_theta(self, grad_j):
        # SHOULD BE DOUBLE CHECKED 1234567891011
        # Update reward network parameters using full gradient descent
        with torch.no_grad():
            start_idx = 0
            for layer_w in self.reward_net.parameters():
                length = self._num_flat_features(layer_w)
                shape = layer_w.size()
                dims = []
                for dim in shape:
                    dims.append(dim)
                dims_tuple = tuple(dims)
                layer_w += self.lr * torch.tensor(grad_j[start_idx:start_idx+length].reshape(dims_tuple), dtype=torch.float32, device=self.device)
                start_idx += length

    def eval_reward(self):
        reward_store = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                for k in range(self.DFA.n_states):
                    for l in range(4): # for the 4 actions

                        reward_input = self.reward_net_input_method(i,j,k,l)
                        # instead of using the reward from transition_table, use reward_net
                        reward_store[i,j,k,l] = self.reward_net_method(reward_input)
        return reward_store

    def eval_task_performance(self, num_trajs_used, thresh_Q_theta, thresh_Y_theta, trajs):
        self.demo_visitation_calculator(num_trajs_used, trajs)
        self.calc_Q_theta(thresh=thresh_Q_theta)
        self.calc_pi_theta()
        self.calc_y_theta(thresh=thresh_Y_theta)
        L_phi, min_y =  self.calc_L_phi_theta()

        return (L_phi, min_y, copy.deepcopy(self.y_theta), copy.deepcopy(self.Q_theta),
            copy.deepcopy(self.pi_theta), copy.deepcopy(self.visitation_counts_arr),
            copy.deepcopy(self.pi_optimal_soft), copy.deepcopy(self.pi_optimal_greedy))

class agent_RL(agent_RL_parent):

    def __init__(self,
                envs,
                init_vars,
                agent_mode
                ):
        super().__init__(envs, init_vars, agent_mode)

        if agent_mode in ["val", "test"]:
            pass
        else:
            self._calc_Q_optimal(thresh=self.thresh_Q_optimal) # fills in self.Q_optimal and self.V_optimal
            self._calc_pi_optimal() # fills in self.pi_optimal_soft, self.pi_optimal_greedy
            self.y_optimal = self._calc_y_optimal(thresh=self.thresh_y_optimal)

        # new networks
        self.reward_net_name = init_vars["reward_net"]
        self.reward_input_size = init_vars["reward_input_size"]
        if self.reward_net_name == "MLP":
            self.reward_net = net_MLP(self.MDP, self.DFA, n_dim=init_vars["reward_input_size"], out_dim=1).to(self.device)
        self.lr = init_vars["lr"]

        # theta based variables
        self.reward_input_batch = None
        self.Q_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.V_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        self.R_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.y_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        self.y_theta_eval = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA_GT.n_states))
        self.theta_size = self._get_size_theta()
        self.grad_Q_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_R_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_pi_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_y_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, self.theta_size))


    def reward_net_input_batch_method(self):
        return utils.reward_net_input_batch(self.MDP, self.DFA, self.PA ,
                                            self.device, self.reward_input_size)

    def reward_net_method(self, reward_input_batch):
        return self.reward_net(reward_input_batch)

    def reward_net_input_method(self,i,j,k,l):
        neigh = self.MDP.neigh_select_reward(i,j,self.reward_input_size)
        return utils.reward_net_input(neigh,k,l,self.device, self.DFA.n_states)

class agent_RL_bits(agent_RL_parent_bits):

    def __init__(self,
                envs,
                init_vars,
                agent_mode
                ):
        super().__init__(envs, init_vars, agent_mode)

        # new networks
        self.reward_net_name = init_vars["reward_net"]
        self.reward_input_size = init_vars["reward_input_size"]
        if self.reward_net_name == "MLP":
            self.reward_net = net_MLP(self.MDP, self.DFA, n_dim=init_vars["reward_input_size"], out_dim=1).to(self.device)
        self.lr = init_vars["lr"]

        # theta based variables
        self.reward_input_batch = None
        self.Q_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.V_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        self.R_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.theta_size = self._get_size_theta()
        self.grad_Q_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_R_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_pi_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_y_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, self.theta_size))

    def reward_net_input_batch_method(self):
        return utils.reward_net_input_batch(self.MDP, self.DFA, self.PA ,
                                            self.device, self.reward_input_size)

    def reward_net_method(self, reward_input_batch):
        return self.reward_net(reward_input_batch)

    def reward_net_input_method(self,i,j,k,l):
        neigh = self.MDP.neigh_select_reward(i,j,self.reward_input_size)
        return utils.reward_net_input(neigh,k,l,self.device, self.DFA.n_states)

class agent_RL_non_local_r(agent_RL_parent):

    def __init__(self,
                envs,
                init_vars,
                agent_mode
                ):
        super().__init__(envs, init_vars, agent_mode)

        if agent_mode in ["val", "test"]:
            pass
        else:
            self._calc_Q_optimal(thresh=self.thresh_Q_optimal) # fills in self.Q_optimal and self.V_optimal
            self._calc_pi_optimal() # fills in self.pi_optimal_soft, self.pi_optimal_greedy
            self.y_optimal = self._calc_y_optimal(thresh=self.thresh_y_optimal)

        # new networks
        self.reward_net_name = init_vars["reward_net"]

        if self.reward_net_name == "CNN":
            self.reward_net = net_CNN_non_local_r(self.MDP, out_dim=1).to(self.device)
        self.lr = init_vars["lr"]

        # theta based variables
        self.reward_input_batch = None
        self.Q_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.R_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.y_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        self.theta_size = self._get_size_theta()
        self.grad_Q_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_R_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_pi_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_y_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, self.theta_size))

    def reward_net_input_batch_method(self):
        return utils.reward_net_input_batch_non_local(
                self.MDP, self.DFA, self.PA, self.device)

    def reward_net_method(self, reward_input_batch):
        return self.reward_net(self.reward_input_batch[0], self.reward_input_batch[1], self.reward_input_batch[2])

    def reward_net_input_method(self,i,j,k,l):
        return utils.reward_net_input_non_local(self.MDP,i,j,k,l,self.device)

class agent_RL_baseline(agent_RL_parent):

    def __init__(self,
                envs,
                init_vars,
                agent_mode
                ):
        super().__init__(envs, init_vars, agent_mode)

        # reg networks
        self.reward_net_name = init_vars["reward_net"]
        self.reward_input_size = init_vars["reward_input_size"]
        if self.reward_net_name == "MLP":
            self.reward_net = net_MLP(self.MDP, self.DFA, n_dim=init_vars["reward_input_size"], out_dim=1).to(self.device)
        self.lr = init_vars["lr"]

        # theta based variables
        self.reward_input_batch = None
        self.Q_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.R_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.y_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        self.theta_size = self._get_size_theta()
        self.grad_Q_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_R_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_pi_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4, self.theta_size))
        self.grad_y_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, self.theta_size))

    def demo_visitation_calculator(self, num_trajs_used, trajs):
        # fills in self.visitation_counts, to be called from the  main function
        self.visitation_counts_arr = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        self.visitation_counts = {}
        # trajs = pickle.load(open(trajs_Address  + "trajectories.pkl", "rb" ) )
        for traj in trajs[:num_trajs_used]:
            for s_a_pair in traj[:-1]:
                i,j,k,l = s_a_pair[1][0], s_a_pair[1][1], s_a_pair[2], s_a_pair[3]
                k = 0
                self.visitation_counts_arr[i,j,k,l] += 1
                if (i,j,k,l) in self.visitation_counts.keys():
                    self.visitation_counts[(i,j,k,l)] += 1
                else:
                    self.visitation_counts[(i,j,k,l)] = 1

    def calc_L_D_theta(self, num_trajs_used):

        Q_max_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        for ind in range(4):
            Q_max_act[:,:,:,ind] = np.max(self.Q_theta,3)
        scaled_down_Q = self.Q_theta - Q_max_act
        L_D_theta = 0
        for key in self.visitation_counts.keys():
            i,j,k,l = key # l corresponds to action, k corresponds to dfa_state, i,j are for mdp state
            num = self.visitation_counts[key]
            second_term = 0
            for action in range(4):
                second_term += math.exp(scaled_down_Q[i,j,k,action])
            L_D_theta += num * (scaled_down_Q[i,j,k,l] - math.log(second_term))

        return L_D_theta / num_trajs_used

    def reward_net_input_batch_method(self):
        return utils.reward_net_input_batch(self.MDP, self.DFA, self.PA ,
                                            self.device, self.reward_input_size)

    def reward_net_method(self, reward_input_batch):
        return self.reward_net(reward_input_batch)

    def reward_net_input_method(self,i,j,k,l):
        neigh = self.MDP.neigh_select_reward(i,j,self.reward_input_size)
        return utils.reward_net_input(neigh,k,l,self.device, self.DFA.n_states)


# ---------- BEHAVIORAL CLONGING AGENTS --------------
class agent_BC_parent:

    def __init__(self,
                envs,
                init_vars,
                agent_mode
                ):

        self.gamma=init_vars["gamma"]
        self.Transition = init_vars["Transition"]
        self.lr = init_vars["lr"]
        self.policy_net_name = init_vars["policy_net"]
        self.policy_input_size = None
        self.policy_optimizer = init_vars["policy_optimizer"]
        self.policy_loss = init_vars["policy_loss"]
        self.batch_size = init_vars["batch_size"]
        self.thresh_Q_optimal = init_vars["thresh_Q_optimal"]
        self.thresh_y_optimal = init_vars["thresh_y_optimal"]
        self.device = torch.device(init_vars["device"])

        self.MDP, self.DFA, self.PA, self.DFA_GT, self.PA_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
        self.n_actions = 4

        # Q and pi and y optimal, only phi_theta is initialized here among theta based vars
        self.pi_theta, self.pi_optimal_soft, self.pi_optimal_greedy = self._pi_init()

        # following 3 to be filled in by calling demo_training_set()
        self.mdp_states_demo = None
        self.dfa_states_demo = None
        self.actions_demo = None
        self.mdp_states_all = None
        self.dfa_states_all = None

        self.y_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))

    def demo_training_set(self, num_trajs_used, trajs):
        # fills in self.visitation_counts, to be called from the  main function
        # trajs = pickle.load(open(trajs_Address  + "trajectories.pkl", "rb" ) )
        all_state_action_pairs = []
        for traj in trajs[:num_trajs_used]:
            for s_a_pair in traj[:-1]:
                i,j,k,l = s_a_pair[1][0], s_a_pair[1][1], s_a_pair[2], s_a_pair[3]
                all_state_action_pairs.append([i,j,k,l])
        random.shuffle(all_state_action_pairs)

        mdp_states = []
        dfa_states = []
        actions = []
        for item in all_state_action_pairs:
            mdp_states.append(item[:2])
            dfa_states.append(self.one_hot_encode_DFA(item[2]))
            actions.append(torch.tensor(np.array(item[3])))
        self.mdp_states_demo = torch.tensor(mdp_states, dtype=torch.long)
        self.dfa_states_demo = torch.tensor(dfa_states, dtype=torch.float32, device=self.device)
        self.actions_demo = torch.tensor(actions, dtype=torch.long, device=self.device)

    def calc_all_mdp_dfa_inputs(self):
        # fills in self.visitation_counts, to be called from the  main function
        # trajs = pickle.load(open(trajs_Address  + "trajectories.pkl", "rb" ) )
        all_states = []
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                for k in range(self.DFA.n_states):
                    all_states.append([i,j,k])
        mdp_states = []
        dfa_states = []
        for item in all_states:
            mdp_states.append(item[:2])
            dfa_states.append(self.one_hot_encode_DFA(item[2]))
        self.mdp_states_all = torch.tensor(mdp_states, dtype=torch.int32)
        self.dfa_states_all = torch.tensor(dfa_states, dtype=torch.float32, device=self.device)

    def one_hot_encode_action(self,action):
        encoded_action = [0,0,0,0]
        encoded_action[action] = 1
        return encoded_action

    def one_hot_encode_DFA(self,dfa_state):
        encoded_dfa = [0,0,0,0,0,0,0]
        encoded_dfa[dfa_state] = 1
        return encoded_dfa

    def set_optim(self, params, optimizer_str, loss_str, lr):
        if optimizer_str == 'SGD':
            optimizer = optim.SGD(params, lr=lr)
        elif optimizer_str == 'Adam':
            optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        elif optimizer_str == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-08, weight_decay=0,
                momentum=0, centered=False)

        if loss_str == 'MSEloss':
            criterion = nn.MSELoss()
        elif loss_str == 'SmoothL1Loss':
            criterion = nn.SmoothL1Loss()
        elif loss_str == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()

        return optimizer, criterion

    def policy_net_batch_method(self):
        pass

    def policy_net_all_eval_method(self):
        pass

    def update_theta(self):
        sample_size = self.batch_size
        prob_dists, action_batch = self.policy_net_batch_method()
        loss = self.policy_criterion(prob_dists, action_batch)
        self.policy_optimizer.zero_grad()
        params_t = list(self.policy_net.parameters())
        loss.backward()
        self.policy_optimizer.step()
        return loss.item()

    def calc_pi_theta(self):

        policy, pi_theta_tensor = self.policy_net_all_eval_method()
        pi_theta = pi_theta_tensor.to(device="cpu").numpy()

        pi_sum_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        pi_max_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, 4))
        for ind in range(4):
            pi_max_act[:,:,:,ind] = np.max(pi_theta,3)
        scaled_down_pi = pi_theta - pi_max_act
        pi_sum_reduced = np.sum(np.exp(scaled_down_pi),3)
        for ind in range(4):
            pi_sum_act[:,:,:,ind] = pi_sum_reduced

        self.pi_theta = copy.deepcopy(np.exp(scaled_down_pi) / pi_sum_act)
        return self.pi_theta

    def calc_y_theta(self,thresh):
        """
        summary: calculates y(s,q) using dynamic programming based on Eq. 15 in the writeup

        output: returns y for all (s,q)
        """

        delta = thresh + 1
        counter = 0
        for q_accept in self.DFA.accepting_states:
            y_optimal[:,:,q_accept] = np.ones((self.MDP.n_dim, self.MDP.n_dim)) # y is always one for accepting states

        while delta > thresh:
            counter += 1
            old_y_theta = copy.deepcopy(self.y_theta)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.DFA.non_terminal_states: # we excluded state 4 as the y(s,q) is zero for that state
                        y_sum = 0
                        for l in range(4):
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            y_sum += self.pi_theta[i,j,k,l] * self.y_theta[ip,jp,kp]
                        self.y_theta[i,j,k] = y_sum

            delta_array = np.abs(self.y_theta - old_y_theta)
            delta = np.mean(delta_array)  # 1 norm

    def select_optimal_action_soft(self):
        # softmax policy
        # agent_env_state = utils.agent_env_state(current_state, env_state)
        mdp_state, dfa_state = self.PA.return_current_state()
        i,j = mdp_state[0], mdp_state[1]
        k = dfa_state
        probabilities = self.pi_optimal_soft[i,j,k,:]
        action_idxs = np.arange(4)
        action_idx = np.random.choice(action_idxs, 1, p=probabilities)[0]
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def select_action_soft_pi_theta(self):
        # softmax policy
        # agent_env_state = utils.agent_env_state(current_state, env_state)
        mdp_state, dfa_state = self.PA.return_current_state()
        i,j = mdp_state[0], mdp_state[1]
        k = dfa_state
        probabilities = self.pi_theta[i,j,k,:]
        action_idxs = np.arange(4)
        action_idx = np.random.choice(action_idxs, 1, p=probabilities)[0]
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def select_optimal_action_greedy(self):
        # greedy policy directry from Q function
        mdp_state, dfa_state = self.PA.return_current_state()
        i,j = mdp_state[0], mdp_state[1]
        k = dfa_state
        action_idx = np.argmax(self.Q_optimal[i,j,k,:])
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def random_action_selection(self):
        # print("random action selected")
        # i = self.MDP.current_state[0,0].item()
        # j = self.MDP.current_state[0,1].item()
        # allowable_action_idxs = self.MDP.allowable_action_idxs[i,j]
        all_actions = self.MDP.all_actions
        action_idx = int(np.random.choice(np.arange(4)))
        action = self.MDP.all_actions[action_idx]
        return action_idx, action

    def _pi_init(self):
        pi_theta = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        pi_optimal_soft = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        pi_optimal_greedy = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))
        return pi_theta, pi_optimal_soft, pi_optimal_greedy

    def _calc_pi_optimal(self):
        Q_sum_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        Q_sum_reduced = np.sum(np.exp(self.Q_optimal),3)
        for ind in range(4):
            Q_sum_act[:,:,:,ind] = Q_sum_reduced
        self.pi_optimal_soft = np.exp(self.Q_optimal) / Q_sum_act
        self.pi_optimal_greedy = np.argmax(self.Q_optimal,3)

    def _calc_y_optimal(self, thresh):
        """
        summary: calculates y(s,q) using dynamic programming based on Eq. 15 in the writeup.
        It find y for pi_optimal_soft

        output: returns y for all (s,q)
        """

        delta = thresh+1
        counter = 0
        y_optimal = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states))

        for q_accept in self.DFA.accepting_states:
            y_optimal[:,:,q_accept] = np.ones((self.MDP.n_dim, self.MDP.n_dim)) # y is always one for q3

        while delta > thresh:
            counter += 1
            old_y_optimal = copy.deepcopy(y_optimal)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.DFA.non_terminal_states: # we excluded state 4 as the y(s,q) is zero for that state
                        y_sum = 0
                        for l in range(4):
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            y_sum += self.pi_optimal_soft[i,j,k,l] * y_optimal[ip,jp,kp]
                        y_optimal[i,j,k] = y_sum

            delta_array = np.abs(y_optimal - old_y_optimal)
            delta = np.mean(delta_array) # 1 norm

        return y_optimal

    def calc_grad_L_D_theta(self, num_trajs_used):
        pass

    def calc_grad_pi_theta(self):
        # SHOULD BE DOUBLE CHECKED 1234567891011
        pass

    def calc_grad_y_theta(self, thresh):
        # Enforce ---> make sure grad_y is zero at dfa_state 3 and 4
        # using eq.16, DP

        delta = thresh + 1
        counter = 0
        while delta > thresh:
            counter += 1
            old_grad_y_theta = copy.deepcopy(self.grad_y_theta)
            for i in range(self.MDP.n_dim):
                for j in range(self.MDP.n_dim):
                    for k in self.non_terminal_states: # exclude dfa states 3 and 4 as their gradient must stay zero
                        sum_action_term = np.zeros(self.theta_size)
                        for l in range(4): # for actions
                            ip,jp,kp = self.PA.transition_table[(i,j,k,l)][:3]
                            sum_action_term += (self.pi_theta[i,j,k,l] * self.grad_y_theta[ip,jp,kp,:] +
                                                self.y_theta[ip,jp,kp] * self.grad_pi_theta[i,j,k,l,:])
                        self.grad_y_theta[i,j,k,:] = sum_action_term

            delta_array = np.abs(self.grad_y_theta - old_grad_y_theta)
            delta = np.mean(delta_array) # 1 norm

    def calc_grad_L_phi_theta(self):
        # 1234567891011 MUST BE FIXED
        # using eqs 14 and 16
        grad_L_phi_theta = np.zeros(self.theta_size)
        for state in self.MDP.non_imp_obj_idxs_all:
            grad_L_phi_theta += self.grad_y_theta[state[0],state[1],0]

        grad_L_phi_theta /= len(self.MDP.non_imp_obj_idxs_all)
        return grad_L_phi_theta

    def calc_L_D_theta(self, num_trajs_used):

        Q_max_act = np.zeros((self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states,4))
        for ind in range(4):
            Q_max_act[:,:,:,ind] = np.max(self.Q_theta,3)
        scaled_down_Q = self.Q_theta - Q_max_act
        L_D_theta = 0
        for key in self.visitation_counts.keys():
            i,j,k,l = key # l corresponds to action, k corresponds to dfa_state, i,j are for mdp state
            num = self.visitation_counts[key]
            second_term = 0
            for action in range(4):
                second_term += math.exp(scaled_down_Q[i,j,k,action])
            L_D_theta += num * (scaled_down_Q[i,j,k,l] - math.log(second_term))
        return L_D_theta / num_trajs_used

    def calc_L_phi_theta(self):
        # using eqs 14
        L_phi_theta = 0
        vals = []
        for state in self.MDP.non_imp_obj_idxs_all:
            L_phi_theta += self.y_theta[state[0],state[1],0]
            vals.append(self.y_theta[state[0],state[1],0])
        L_phi_theta /= len(self.MDP.non_imp_obj_idxs_all)
        min_y = min(vals)
        return L_phi_theta, min_y

    def _num_flat_features(self, x):
        size = x.size()  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def eval_task_performance_base(self, num_trajs_used, thresh_Q_theta, thresh_Y_theta, trajs):
        self.demo_visitation_calculator(num_trajs_used, trajs)
        self.calc_Q_theta_for_base_eval(thresh=thresh_Q_theta)
        self.calc_pi_theta()
        self.calc_y_theta(thresh=thresh_Y_theta)
        L_phi, min_y =  self.calc_L_phi_theta()

        return (L_phi, min_y, copy.deepcopy(self.y_theta), copy.deepcopy(self.Q_theta),
            copy.deepcopy(self.pi_theta), copy.deepcopy(self.visitation_counts_arr),
            copy.deepcopy(self.pi_optimal_soft), copy.deepcopy(self.pi_optimal_greedy))

    def eval_task_performance(self, num_trajs_used, thresh_Q_theta, thresh_Y_theta, trajs):
        self.calc_all_mdp_dfa_inputs()
        self.calc_pi_theta()
        self.calc_y_theta(thresh=thresh_Y_theta)
        L_phi, min_y =  self.calc_L_phi_theta()
        return (L_phi, min_y, copy.deepcopy(self.y_theta), copy.deepcopy(self.pi_theta))

    def eval_L_phi(self, num_trajs_used, thresh_Y_theta):
        self.calc_all_mdp_dfa_inputs()
        self.calc_pi_theta()
        self.calc_y_theta(thresh=thresh_Y_theta)
        L_phi, min_y =  self.calc_L_phi_theta()
        return L_phi, min_y

    def eval_L_phi_base(self, num_trajs_used, thresh_Q_theta, thresh_Y_theta):
        self.calc_Q_theta_for_base_eval(thresh=thresh_Q_theta)
        self.calc_pi_theta()
        self.calc_y_theta(thresh=thresh_Y_theta)
        L_phi, min_y =  self.calc_L_phi_theta()
        return L_phi, min_y

class agent_RL_BC_v2(agent_BC_parent):

    def __init__(self,
                envs,
                init_vars,
                agent_mode
                ):
        super().__init__(envs, init_vars, agent_mode)

        # reg networks
        if self.policy_net_name == "att":
            self.policy_net = net_att(self.MDP, out_dim=4).to(self.device)
        elif self.policy_net_name == "CNN":
            self.policy_net = net_CNN_pol_v2(self.MDP, self.DFA, out_dim=4).to(self.device)
        elif self.policy_net_name == "MLP":
            self.policy_net = net_MLP_pol_v2(self.MDP, self.DFA, out_dim=4).to(self.device)

        self.policy_optimizer, self.policy_criterion = self.set_optim(self.policy_net.parameters(),
            self.policy_optimizer, self.policy_loss, self.lr)

    def policy_net_batch_method(self):
        mdp_batch, dfa_batch, action_batch = utils.policy_net_X_Y_batch_v2(copy.deepcopy(self.MDP.grid),
            self.mdp_states_demo, self.dfa_states_demo, self.actions_demo, self.batch_size, self.device)
        return self.policy_net(mdp_batch, dfa_batch), action_batch

    def policy_net_all_eval_method(self):
        mdp_batch, dfa_batch = utils.policy_net_all_eval_input_v2(self.MDP.grid, self.mdp_states_all,
                            self.dfa_states_all, self.device)
        with torch.no_grad():
            policy = self.policy_net(mdp_batch, dfa_batch)
            pi_theta_tensor = policy.view(self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, -1)
        return policy, pi_theta_tensor

class agent_RL_BC_local(agent_BC_parent):

    def __init__(self,
                envs,
                init_vars,
                agent_mode
                ):
        super().__init__(envs, init_vars, agent_mode)

        self.policy_input_size = init_vars["policy_input_size"]
        # reg networks
        if self.policy_net_name == "MLP":
            self.policy_net = net_MLP_pol_local(self.MDP, self.DFA, n_dim=self.policy_input_size, out_dim=4).to(self.device)

        self.policy_optimizer, self.policy_criterion = self.set_optim(self.policy_net.parameters(),
            self.policy_optimizer, self.policy_loss, self.lr)

    def policy_net_batch_method(self):
        policy_input, action_batch = utils.policy_net_X_Y_batch_local(copy.deepcopy(self.MDP), copy.deepcopy(self.DFA),
            self.mdp_states_demo, self.dfa_states_demo, self.actions_demo, self.policy_input_size, self.batch_size, self.device)
        return self.policy_net(policy_input), action_batch

    def policy_net_all_eval_method(self):
        policy_input = utils.policy_net_all_eval_input_local(self.MDP, self.DFA, self.policy_input_size, self.device)

        with torch.no_grad():
            policy = self.policy_net(policy_input)
            pi_theta_tensor = policy.view(self.MDP.n_dim, self.MDP.n_dim, self.DFA.n_states, -1)
        return policy, pi_theta_tensor
