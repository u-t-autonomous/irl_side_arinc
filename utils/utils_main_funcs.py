# AUTHOR: Farzan Memarian
from pdb import set_trace
import numpy as np
import copy
import pickle

import sys, os
from os import listdir
from os.path import isfile, join
from collections import namedtuple, OrderedDict
import agent.agent_rl2 as agent_env
# from envs.env2 import *
import envs.env2 as env_mod
# from env2 import MDP, DFA_v0, DFA_v1, DFA_v2, DFA_base, DFA_incomp, PA, PA_base
from agent import agent_rl2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import pickle

import cv2

def produce_mdp(init_vars):
    # create an mdp
    MDP_obj = env_mod.MDP(n_dim = init_vars["n_dim"],
                main_function = init_vars["main_function"],
                n_imp_objs = init_vars["n_imp_objs"],
                n_obstacles = init_vars["n_obstacles"],
                imp_obj_idxs_init_det = init_vars["imp_obj_idxs_init_det"],
                obstacle_idxs_det = init_vars["obstacle_idxs_det"],
                road_idxs_det = init_vars["road_idxs_det"],
                dirt_idxs_det = init_vars["dirt_idxs_det"],
                obj_type = init_vars["obj_type"],
                random_start = init_vars["random_start"],
                init_type=init_vars["init_type"],
                RGB = init_vars["RGB"],
                pad_element = init_vars["pad_element"],
                device = torch.device(init_vars["device"])
                )
    return MDP_obj

def produce_envs(init_vars):
    MDP_obj = produce_mdp(init_vars)
    return produce_envs_from_MDP(init_vars, MDP_obj)

def produce_envs_from_MDP(init_vars, MDP_obj):
    MDP = copy.deepcopy(MDP_obj)
    DFA_obj = env_mod.DFA_from_raw(MDP = MDP,
                RPNI_output_file_name = init_vars["tr_dfa_address"],
                positive_reward = init_vars["positive_reward"],
                negative_reward = init_vars["negative_reward"],
                device = torch.device(init_vars["device"])
                )
    PA_obj = env_mod.PA(MDP, DFA_obj)

    MDP_GT = copy.deepcopy(MDP_obj)
    DFA_obj_GT = env_mod.DFA_from_raw(MDP = MDP_GT,
                RPNI_output_file_name = init_vars["GT_dfa_address"],
                positive_reward = init_vars["positive_reward"],
                negative_reward = init_vars["negative_reward"],
                device = torch.device(init_vars["device"])
                )
    PA_obj_GT = env_mod.PA(MDP_GT, DFA_obj_GT)

    envs = [MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT]
    return envs

def produce_envs_only_tr_from_MDP(init_vars, MDP_obj):
    MDP = copy.deepcopy(MDP_obj)
    DFA_obj = env_mod.DFA_from_raw(MDP = MDP,
                RPNI_output_file_name = init_vars["tr_dfa_address"],
                positive_reward = init_vars["positive_reward"],
                negative_reward = init_vars["negative_reward"],
                device = torch.device(init_vars["device"])
                )
    PA_obj = env_mod.PA(MDP, DFA_obj)

    envs = [MDP_obj, DFA_obj, PA_obj]
    return envs

def produce_envs_only_tr_from_MDP_memoryless(init_vars, MDP_obj):
    MDP = copy.deepcopy(MDP_obj)
    DFA_obj = env_mod.DFA_from_raw_memoryless(MDP = MDP,
                RPNI_output_file_name = init_vars["tr_dfa_address"],
                positive_reward = init_vars["positive_reward"],
                negative_reward = init_vars["negative_reward"],
                device = torch.device(init_vars["device"])
                )
    PA_obj = env_mod.PA(MDP, DFA_obj)

    envs = [MDP_obj, DFA_obj, PA_obj]
    return envs


def produce_envs_from_MDP_bits(init_vars, MDP_obj):
    MDP = copy.deepcopy(MDP_obj)
    DFA_obj = env_mod.DFA_bits(MDP = MDP,
                positive_reward = init_vars["positive_reward"],
                negative_reward = init_vars["negative_reward"],
                device = torch.device(init_vars["device"])
                )
    PA_obj = env_mod.PA(MDP_obj, DFA_obj)


    MDP_GT = copy.deepcopy(MDP_obj)
    DFA_obj_GT = env_mod.DFA_from_raw(MDP = MDP_GT,
                RPNI_output_file_name = init_vars["GT_dfa_address"],
                positive_reward = init_vars["positive_reward"],
                negative_reward = init_vars["negative_reward"],
                device = torch.device(init_vars["device"])
                )
    PA_obj_GT = env_mod.PA(MDP_GT, DFA_obj_GT)
    envs = [MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT]
    return envs

def produce_envs_only_eval_from_MDP(init_vars, MDP_obj):
    MDP = copy.deepcopy(MDP_obj)
    DFA_obj_GT = env_mod.DFA_from_raw(MDP = MDP,
                RPNI_output_file_name = init_vars["GT_dfa_address"],
                positive_reward = init_vars["positive_reward"],
                negative_reward = init_vars["negative_reward"],
                device = torch.device(init_vars["device"])
                )
    PA_obj_GT = env_mod.PA(MDP, DFA_obj_GT)

    envs = [MDP_obj, DFA_obj_GT, PA_obj_GT]
    return envs

def produce_agent(envs, init_vars, agent_mode):

    # create and initialize the agent
    return agent_env.agent_RL(envs = envs,
                init_vars = init_vars,
                agent_mode = agent_mode
                )

def produce_agent_bits(envs, init_vars, agent_mode):
    # create and initialize the agent
    return agent_env.agent_RL_bits(envs = envs,
                init_vars = init_vars,
                agent_mode = agent_mode
                )

def produce_agent_RL_non_local_r(envs, init_vars, agent_mode):
    # create and initialize the agent
    return agent_env.agent_RL_non_local_r(envs = envs,
                init_vars = init_vars,
                agent_mode = agent_mode
                )

def produce_agent_BC(envs, init_vars, agent_mode):
    # create and initialize the agent
    if init_vars["main_function"] in ["train_BC_v1", "test_BC_v1"]:
        return agent_env.agent_RL_BC_v1(envs = envs,
                    init_vars = init_vars,
                    agent_mode = agent_mode
                    )

    elif init_vars["main_function"] in ["train_BC_v2", "test_BC_v2"]:
        return agent_env.agent_RL_BC_v2(envs = envs,
                    init_vars = init_vars,
                    agent_mode = agent_mode
                    )

    elif init_vars["main_function"] in ["train_BC_local", "test_BC_local"]:
        return agent_env.agent_RL_BC_local(envs = envs,
                    init_vars = init_vars,
                    agent_mode = agent_mode
                    )

def produce_agent_BC_3(envs, init_vars, agent_mode):
    # create and initialize the agent

    return agent_env.agent_RL_BC_3(envs = envs,
                init_vars = init_vars,
                agent_mode = agent_mode
                )

def produce_agent_baseline(envs, init_vars, agent_mode):
    # create and initialize the agent
    return agent_env.agent_RL_baseline(envs = envs,
                init_vars = init_vars,
                agent_mode = agent_mode
                )

def create_dir_logs_and_models(init_vars):
    # this function deletes the directory if it already exists
    # and recreates it.
    cmd = 'rm -rf logs/' + init_vars["fileName_tr_test"]
    os.system(cmd)
    cmd = 'mkdir logs/' + init_vars["fileName_tr_test"]
    os.system(cmd)

    cmd = 'rm -rf saved_models/' + init_vars["fileName_tr_test"]
    os.system(cmd)
    cmd = 'mkdir saved_models/' + init_vars["fileName_tr_test"]
    os.system(cmd)

def create_dir_test(init_vars):

    from pathlib import Path

    # this function deletes the directory if it already exists
    # and recreates it.
    cmd = 'rm -rf logs/' + init_vars["fileName_tr_test"] + "/" + init_vars["fileName_grids"] + "/" + init_vars["test_type"] + "/"
    os.system(cmd)

    pathName = Path('logs/' + init_vars["fileName_tr_test"] + "/" + init_vars["fileName_grids"] + "/")

    if not os.path.exists(pathName):
        os.makedirs(pathName)

    cmd = 'mkdir logs/' + init_vars["fileName_tr_test"] + "/" + init_vars["fileName_grids"] + "/" + init_vars["test_type"] + "/"
    os.system(cmd)

def create_dir_opt_trajs(trajs_address):
    # this function deletes the directory if it already exists
    # and recreates it.
    cmd = 'rm -rf ' + trajs_address
    os.system(cmd)
    cmd = 'mkdir ' + trajs_address
    os.system(cmd)

def create_dir_grids(fileName_grids):
    cmd = 'rm -rf grids/' + fileName_grids
    os.system(cmd)
    cmd = 'mkdir grids/' + fileName_grids
    os.system(cmd)

def return_address_grids(fileName_grids):
    return 'grids/' + fileName_grids + "/"

def return_addresses_logs_models(fileName_tr_test):
    logs_address = "logs/" + fileName_tr_test + "/"
    models_address = "saved_models/" + fileName_tr_test + "/"
    return logs_address, models_address

def return_address_test(init_vars):
    fileName_tr_test, fileName_grids, test_type = init_vars["fileName_tr_test"], init_vars["fileName_grids"], init_vars["test_type"]
    test_address = "logs/" + fileName_tr_test + "/" + fileName_grids + "/" + test_type + "/"
    return test_address

def return_address_opt_trajs(grids_address, fileName_opt_trajs, fileName_tr_test):
    trajs_address = grids_address + fileName_opt_trajs + "/" + fileName_tr_test + "/"
    return trajs_address

def run_active_DFA_inference(init_vars, DFA_obj_GT, proc):
    """
    This function takes in a proc which is a DFA inference executive that has been initiated
    This function is only able to answer to queries that are sequence of numbers, not to "equivalent" query
    This function keeps answering queries until the next "equivalent" is reached.
    Then it outputs a new DFA and returns control to the caller function
    """
    query = proc.stdout.readline()
    if query[-1:] == "\n":
        query = query[:-1]
    done = False

    assert query != "equivalent"

    while not done:
        ans = answer_query(query, init_vars, DFA_obj_GT, proc)
        print(query, ans)
        proc.stdin.write(ans)
        proc.stdin.flush()
        query = proc.stdout.readline()
        if query[-1:] == "\n":
            query = query[:-1]
        if query == "equivalent":
            done = True
    return query

    # THE FOLLOWING NEED TO BE EXECUTED AT SOME POINT
    # proc.stdin.close()
    # proc.terminate()
    # proc.wait(timeout=0.2)

def answer_query(query, init_vars, DFA_obj_GT, proc):
    # if query == "equivalent":
    #     is_equivalent = compare_DFAs(init_vars)
    #     if not is_equivalent:
    #         proc.stdin.write("n\n")
    #         proc.stdin.flush()
    #         return provide_counter_ex(init_vars)
    #     else:
    #         proc.stdin.write("y\n")
    #         proc.stdin.flush()
    #         return "y\n"
    if query.isdigit():
        query_list = list(query)
        trace = [int(elem) for elem in query_list]
        accepting = run_trace(trace, DFA_obj_GT)
        if accepting:
            ans = "1\n"
        else:
            ans = "0\n"
    else:
        print("don't know how to answer the query")
        set_trace()
    return ans

def run_trace(trace, DFA):
    old_state = 0
    for predicate in trace:
        next_state, _ = DFA.calc_next_S_and_R_from_predicate(old_state, predicate)
        old_state = next_state
    if next_state in DFA.accepting_states:
        return True
    else:
        return False

def compare_DFAs(init_vars):
    return False

def _find_shortest_trace(traces):
    lengths = np.zeros(len(traces))
    for idx, trace in enumerate(traces):
        lengths[idx] = len(trace)
    return traces[np.argmin(lengths)]

def provide_counter_ex(init_vars, DFA_counter, mdp):
    _, DFA_obj_GT, _ = produce_envs_only_eval_from_MDP(init_vars, mdp)

    if DFA_counter == 1:
        # provide an example that is winning on the GT DFA, the way this function is written,
        # it returns only short traces, i.e. no repetition of predicates that do not lead to
        # transitioning in the DFA
        num_win_trace = 50
        winning_traces = _produce_win_trace(DFA_obj_GT, num_win_trace)
        shortest_trace = _find_shortest_trace(winning_traces)
        return _convert_trace_string(shortest_trace)
    else:
        # provide an example that is winning on the inferred DFA but is losing on the GT DFA or vice versa
        _, DFA_obj_tr, _ = produce_envs_only_tr_from_MDP(init_vars, mdp)
        contradictions = []
        num_win_trace = 50
        winning_traces_tr = _produce_win_trace(DFA_obj_tr, num_win_trace)
        winning_traces_GT = _produce_win_trace(DFA_obj_GT, num_win_trace)

        for trace in winning_traces_GT:
            if not run_trace(trace , DFA_obj_tr):
                contradictions.append(trace)
        for trace in winning_traces_tr:
            if not run_trace(trace , DFA_obj_GT):
                contradictions.append(trace)
        assert len(contradictions) != 0
        shortest_trace = _find_shortest_trace(contradictions)
        return _convert_trace_string(shortest_trace)

def produce_labels_from_map(init_vars):
    resolution = 0.1
    grid_size = 4
    pixels_per_grid = int(grid_size/resolution)
    x_bot = 62
    x_up = 112
    y_bot = 37
    y_up = 87
    map_image = cv2.imread(init_vars["map_address"], cv2.IMREAD_UNCHANGED)
    terrains = cv2.split(map_image)

    x_pixs, y_pixs = terrains[0].shape

    # Clean up each channel from obstacles
    aspalth_masked = np.where(terrains[3] > 0, 0, terrains[0])
    dirt_masked = np.where(terrains[3] > 0, 0, terrains[1])
    grass_masked = np.where(terrains[3] > 0, 0, terrains[2])
    obstacle = terrains[3]

    obstacle_list = []
    road_list = []
    dirt_list = []

    for i in range(x_bot,x_up):
        for j in range(y_bot,y_up):
            if np.any(np.where(obstacle[pixels_per_grid*i:pixels_per_grid*i+pixels_per_grid,
                              pixels_per_grid*j:pixels_per_grid*j+pixels_per_grid]>0,True,False)):
                obstacle_list.append((i-x_bot,j-y_bot))
            elif np.any(np.where(aspalth_masked[pixels_per_grid*i:pixels_per_grid*i+pixels_per_grid,
                              pixels_per_grid*j:pixels_per_grid*j+pixels_per_grid]>0,True,False)):
                road_list.append((i-x_bot,j-y_bot))
            elif np.any(np.where(dirt_masked[pixels_per_grid*i:pixels_per_grid*i+pixels_per_grid,
                              pixels_per_grid*j:pixels_per_grid*j+pixels_per_grid]>0,True,False)):
                dirt_list.append((i-x_bot,j-y_bot))

    init_vars["obstacle_idxs_det"] = [[],obstacle_list]
    init_vars["road_idxs_det"] = road_list
    init_vars["dirt_idxs_det"] = dirt_list

    return init_vars

def get_trajectories(init_vars):
    print("am")
    all_trajs = []
    all_files = [f for f in listdir(init_vars["traj_address"]) if isfile(join(init_vars["traj_address"], f))]
    for f in all_files:
        with open(join(init_vars["traj_address"], f), "rb") as fp:
            traj = pickle.load(fp)

        traj_list = []
        first = True
        for point in traj:
            # x_grid = int(((point[0]+1) // 2) + 13)
            # y_grid = int((point[1] // 2) + 92)
            x_grid = int(((point[0]+2) // 4) + 6)
            y_grid = int((point[1] // 4) + 46)
            if not first:
                if old_x_grid > x_grid:
                    action = 0
                elif old_x_grid < x_grid:
                    action = 1
                elif old_y_grid > y_grid:
                    action = 3
                elif old_y_grid < y_grid:
                    action = 2
                else:
                    continue
                traj_list.append((0,[x_grid,y_grid],0,action))
            old_x_grid = copy.deepcopy(x_grid)
            old_y_grid = copy.deepcopy(y_grid)
            first = False
        traj_list.append("game won")
        all_trajs.append(traj_list)
    return all_trajs

def _convert_trace_string(trace):
    str_trace = ""
    for char in trace:
        str_trace += str(char)
    return str_trace + "\n"

def _produce_win_trace(DFA, num_win_trace):
    traces = []
    thresh_length = 100
    while len(traces) <= num_win_trace:
        trace = []
        old_state = DFA.reset()
        counter_length = 0
        while old_state not in DFA.accepting_states and counter_length < thresh_length:
            counter_length += 1
            predicate = np.random.choice(np.arange(len(DFA.alphabet)), 1)[0]
            next_state, _  = DFA.calc_next_S_and_R_from_predicate(old_state, predicate)
            if next_state != old_state:
                old_state = next_state
                trace.append(predicate)
        if counter_length < thresh_length:
            traces.append(trace)

    return traces
