# AUTHOR: Farzan Memarian
from pdb import set_trace 
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import pickle
import random



# policy networks
def policy_net_X_Y_batch_v1(grid_orig, mdp_states_demo, dfa_states_demo, actions_demo, batch_size, device):
	grid = copy.deepcopy(grid_orig.to(device))
	m,n = mdp_states_demo.size()
	randIndex = random.sample(range(m), batch_size)
	mdp_samples = copy.deepcopy(mdp_states_demo[randIndex])
	dfa_batch = copy.deepcopy(dfa_states_demo[randIndex])
	action_batch = copy.deepcopy(actions_demo[randIndex]) 
	_,_,n_dim = grid.size()
	input_batch_list = []
	for item in range(batch_size):
		mdp_loc_map =  torch.zeros((1,n_dim,n_dim),dtype=torch.float32, device=device)
		dfa_encoding = torch.zeros((5,n_dim,n_dim),dtype=torch.float32, device=device)
		dfa_encoding[dfa_batch[item].item(),:,:] += 1
		i,j = mdp_samples[item]
		
		mdp_loc_map[0,i,j] = 1
		batch_item = torch.unsqueeze(torch.cat((dfa_encoding, mdp_loc_map, grid),0), 0)
		input_batch_list.append(batch_item)

	input_batch = input_batch_list[0]
	for item in input_batch_list[1:]:
		input_batch = torch.cat((input_batch, item), 0)
	return input_batch, action_batch

def policy_net_all_eval_input_v1(grid_orig, mdp_states_all, dfa_states_all, device):
	grid = copy.deepcopy(grid_orig.to(device))
	_, _, n_dim = grid.size()
	mdp_list = []
	for idx, item in enumerate(mdp_states_all):
		mdp_loc_map = torch.zeros((1,n_dim,n_dim), device=device)
		dfa_encoding = torch.zeros((5,n_dim,n_dim),dtype=torch.float32, device=device)
		dfa_encoding[dfa_states_all[idx].item(),:,:] += 1
		i,j = item
		mdp_loc_map[0,i,j] = 1
		batch_item = torch.unsqueeze(torch.cat((dfa_encoding, mdp_loc_map, grid),0), 0)
		mdp_list.append(batch_item)

	mdp_batch = mdp_list[0]
	for item in mdp_list[1:]:
		mdp_batch = torch.cat((mdp_batch, item), 0)
	return mdp_batch

def policy_net_X_Y_batch_v2(grid_orig, mdp_states_demo, dfa_states_demo, actions_demo, batch_size, device):
	grid = copy.deepcopy(grid_orig.to(device))
	m,n = mdp_states_demo.size()
	randIndex = random.sample(range(m), batch_size)
	mdp_samples = copy.deepcopy(mdp_states_demo[randIndex])
	dfa_batch = copy.deepcopy(dfa_states_demo[randIndex])
	action_batch = copy.deepcopy(actions_demo[randIndex]) 
	_,_,n_dim = grid.size()
	mdp_batch_list = []
	for item in range(batch_size):
		mdp_loc_map = torch.zeros((1,n_dim,n_dim),dtype=torch.float32,device=device)
		i,j = mdp_samples[item]
		mdp_loc_map[0,i,j] = 1
		batch_item = torch.unsqueeze(torch.cat((mdp_loc_map, grid),0), 0)
		mdp_batch_list.append(batch_item)

	mdp_batch = mdp_batch_list[0]
	for item in mdp_batch_list[1:]:
		mdp_batch = torch.cat((mdp_batch, item), 0)
	return mdp_batch, dfa_batch, action_batch

def policy_net_X_Y_batch_local(MDP_obj, DFA_obj, mdp_states_demo, dfa_states_demo, actions_demo, policy_input_size, batch_size, device):
	m,n = mdp_states_demo.size()
	randIndex = random.sample(range(m), batch_size)
	mdp_samples = copy.deepcopy(mdp_states_demo[randIndex])
	dfa_batch = copy.deepcopy(dfa_states_demo[randIndex])
	action_batch = copy.deepcopy(actions_demo[randIndex]) 
	n_dfa_states = DFA_obj.n_states
	depth,_,_ = MDP_obj.grid.size()
	policy_input_batch = torch.zeros((batch_size,policy_input_size*policy_input_size*depth+n_dfa_states), dtype=torch.float32, device=device)
	for item in range(batch_size):
		i, j = mdp_samples[item]
		l = action_batch[item]
		neigh = MDP_obj.neigh_select_reward(i,j,policy_input_size) 
		num_flat = num_flat_features(neigh)
		neigh = neigh.view(-1, num_flat)
		encoded_dfa = dfa_batch[item]
		policy_input_batch[item,:num_flat] = neigh
		# reward_input_batch[counter,num_flat+0] = i
		# reward_input_batch[counter,num_flat+1] = j
		policy_input_batch[item,num_flat:num_flat+n_dfa_states] = encoded_dfa

	return policy_input_batch, action_batch


def policy_net_all_eval_input_v2(grid_orig, mdp_states_all, dfa_states_all, device):
	grid = copy.deepcopy(grid_orig.to(device))
	_, _, n_dim = grid.size()
	mdp_list = []
	for idx, item in enumerate(mdp_states_all):
		mdp_loc_map = torch.zeros((1,n_dim,n_dim), device=device)

		i,j = item
		mdp_loc_map[0,i,j] = 1
		batch_item = torch.unsqueeze(torch.cat((mdp_loc_map, grid),0), 0)
		mdp_list.append(batch_item)

	mdp_batch = mdp_list[0]
	for item in mdp_list[1:]:
		mdp_batch = torch.cat((mdp_batch, item), 0)
	dfa_batch = dfa_states_all
	return mdp_batch, dfa_batch

def policy_net_all_eval_input_local(MDP_obj, DFA_obj, policy_input_size, device):
	grid = copy.deepcopy(MDP_obj.grid.to(device))
	depth, _, n_dim = grid.size()
	n_dfa_states = DFA_obj.n_states
	total_num_states = n_dim * n_dim * n_dfa_states
	policy_input_all = torch.zeros((total_num_states,policy_input_size*policy_input_size*depth+n_dfa_states), dtype=torch.float32, device=device)
	counter = 0
	for i in range(n_dim):
		for j in range(n_dim):
			for k in range(n_dfa_states):
				neigh = MDP_obj.neigh_select_reward(i,j,policy_input_size) 
				num_flat = num_flat_features(neigh)
				neigh = neigh.view(-1, num_flat)
				encoded_dfa = one_hot_encode_DFA(k, device, n_dfa_states)
				policy_input_all[counter, :num_flat] = neigh
				# reward_input_batch[counter,num_flat+0] = i
				# reward_input_batch[counter,num_flat+1] = j
				policy_input_all[counter, num_flat:num_flat+n_dfa_states] = encoded_dfa
				counter += 1
	return policy_input_all


def policy_net_X_Y_batch_ind_dfa(grid_orig, mdp_states_demo, actions_demo, batch_size, device):
	grid = copy.deepcopy(grid_orig.to(device))
	m,n = mdp_states_demo.size()
	randIndex = random.sample(range(m), batch_size)
	mdp_samples = copy.deepcopy(mdp_states_demo[randIndex])
	action_batch = copy.deepcopy(actions_demo[randIndex]) 
	_,_,n_dim = grid.size()
	mdp_batch_list = []
	for item in range(batch_size):
		mdp_loc_map = torch.zeros((1,n_dim,n_dim),device=device)
		i,j = mdp_samples[item]
		mdp_loc_map[0,i,j] = 1
		batch_item = torch.unsqueeze(torch.cat((mdp_loc_map, grid),0), 0)
		mdp_batch_list.append(batch_item)

	mdp_batch = mdp_batch_list[0]
	for item in mdp_batch_list[1:]:
		mdp_batch = torch.cat((mdp_batch, item), 0)
	return mdp_batch, action_batch

def policy_net_all_eval_input_ind_dfa(grid_orig, mdp_states_all, device):
	grid = copy.deepcopy(grid_orig.to(device))
	_, _, n_dim = grid.size()
	mdp_list = []
	for item in mdp_states_all:
		mdp_loc_map = torch.zeros((1,n_dim,n_dim), device=device)
		i,j = item
		mdp_loc_map[0,i,j] = 1
		batch_item = torch.unsqueeze(torch.cat((mdp_loc_map, grid),0), 0)
		mdp_list.append(batch_item)

	mdp_batch = mdp_list[0]
	for item in mdp_list[1:]:
		mdp_batch = torch.cat((mdp_batch, item), 0)
	return mdp_batch

# reward networks
def reward_net_input(neigh,k,l,device,n_dfa_states):
	# 1234567891011 TO BE COMPLETED, right now only works for non-rgb case
	num_flat = num_flat_features(neigh)
	neigh = neigh.view(-1, num_flat)
	encoded_dfa = one_hot_encode_DFA(k, device, n_dfa_states)
	encoded_action = one_hot_encode_action(l, device)
	reward_input = torch.cat((neigh,encoded_dfa, encoded_action),1)
	return reward_input

def reward_net_input_batch(MDP_obj, DFA_obj, PA_obj ,device, reward_input_size):
	batch_size = MDP_obj.n_dim**2 * DFA_obj.n_states * 4
	n_dfa_states = DFA_obj.n_states
	depth,_,_ = MDP_obj.grid.size()
	reward_input_batch = torch.zeros((batch_size,reward_input_size*reward_input_size*depth+n_dfa_states+4), dtype=torch.float32, device=device)
	counter = 0 
	for i in range(MDP_obj.n_dim):
		for j in range(MDP_obj.n_dim):
			for k in range(DFA_obj.n_states):
				for l in range(4): # for the 4 actions
					# ip,jp,kp = PA_obj.transition_table[(i,j,k,l)][:3]
					neigh = MDP_obj.neigh_select_reward(i,j,reward_input_size) 
					num_flat = num_flat_features(neigh)
					neigh = neigh.view(-1, num_flat)
					encoded_dfa = one_hot_encode_DFA(k, device, n_dfa_states)
					encoded_action = one_hot_encode_action(l, device)
					reward_input_batch[counter,:num_flat] = neigh
					# reward_input_batch[counter,num_flat+0] = i
					# reward_input_batch[counter,num_flat+1] = j
					reward_input_batch[counter,num_flat:num_flat+n_dfa_states] = encoded_dfa
					reward_input_batch[counter,num_flat+n_dfa_states:num_flat+n_dfa_states+4] = encoded_action
					counter += 1
	return copy.deepcopy(reward_input_batch)

def reward_net_input_non_local(MDP,i,j,k,l,device):
	grid = copy.deepcopy(MDP.grid.to(device))
	depth,_,n_dim = grid.size()
	mdp_loc_map = torch.zeros((1,n_dim,n_dim),dtype=torch.float32,device=device)
	mdp_loc_map[0,i,j] = 1
	reward_input = torch.unsqueeze(torch.cat((mdp_loc_map, grid),0), 0)
	dfa = torch.zeros((1, 5),dtype=torch.float32,device=device)
	action = torch.zeros((1, 4),dtype=torch.float32,device=device)
	dfa[0,:] = one_hot_encode_DFA(k, device)
	action[0,:] = one_hot_encode_action(l, device)
	return reward_input, dfa, action

def reward_net_input_batch_non_local(MDP, DFA, PA, device):
	grid = copy.deepcopy(MDP.grid.to(device))
	depth,_,n_dim = grid.size()
	mdp_batch_list = [] 
	dfa_batch = torch.zeros((n_dim*n_dim*DFA.n_states*4, DFA.n_states),dtype=torch.float32,device=device)
	actions_batch = torch.zeros((n_dim*n_dim*DFA.n_states*4, 4),dtype=torch.float32,device=device)
	counter = 0
	for i in range(n_dim):
		for j in range(n_dim):
			for k in range(DFA.n_states):
				for l in range(4):
					mdp_loc_map = torch.zeros((1,n_dim,n_dim),dtype=torch.float32,device=device)
					mdp_loc_map[0,i,j] = 1
					batch_item = torch.unsqueeze(torch.cat((mdp_loc_map, grid),0), 0)
					mdp_batch_list.append(batch_item)
					dfa_batch[counter,:] = one_hot_encode_DFA(k, device)
					actions_batch[counter,:] = one_hot_encode_action(l, device)
					counter += 1

	mdp_batch = mdp_batch_list[0]
	for item in mdp_batch_list[1:]:
		mdp_batch = torch.cat((mdp_batch, item), 0)
	return mdp_batch, dfa_batch, actions_batch

# auxiliary functions
def agent_env_state(agent_loc, env_state):
	env_state = copy.deepcopy(env_state)
	depth, n,m = env_state.size()
	agent_state = torch.zeros((1,n,m), device=env_state.device)
	agent_state[0, agent_loc[0], agent_loc[1]] = 1
	agent_env_state = torch.cat([agent_state, env_state],0)
	return agent_env_state

def agent_env_state_deep(agent_state, env_state):
	agent_env_state = copy.deepcopy(env_state)
	agent_env_state[agent_state[0,0], agent_state[0,1]] = -1
	return torch.unsqueeze(agent_env_state,0)

def flatten(neigh):
	n,m = neigh.size()
	pass

def num_flat_features(x):
    size = x.size()
    num_features = 1
    for s in size:
        num_features *= s
    return num_features 

def one_hot_encode_action(action, device):
	encoded_action = torch.zeros((1,4), dtype=torch.float32, device=device)
	encoded_action[0,action] = 1
	return encoded_action

def one_hot_encode_DFA(dfa_state, device, n_dfa_states):
	encoded_dfa = torch.zeros((1,n_dfa_states), dtype=torch.float32, device=device)
	encoded_dfa[0,dfa_state] = 1
	return encoded_dfa
