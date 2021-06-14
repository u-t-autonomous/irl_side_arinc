# AUTHOR: Farzan Memarian
from pdb import set_trace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import argparse
import sys, os
from collections import namedtuple, OrderedDict
sys.path.append(os.getcwd())

# from envs.env2 import MDP, DFA_v0, DFA_v1, DFA_v2, DFA_base, PA, PA_base, DFA_incomp
import agent.agent_rl2 as agent_env
import envs.env2 as env_mod
from utils.utils_main_funcs import (
    produce_mdp, produce_envs, produce_envs_from_MDP, produce_envs_only_eval_from_MDP,
    produce_envs_only_tr_from_MDP, produce_envs_only_tr_from_MDP_memoryless,
    produce_agent, produce_agent_bits, produce_agent_RL_non_local_r, produce_agent_BC, produce_agent_BC_3,
    produce_agent_baseline, create_dir_logs_and_models,
    create_dir_test, create_dir_opt_trajs, create_dir_grids, return_address_grids,
    return_addresses_logs_models, return_address_test, return_address_opt_trajs,
    run_active_DFA_inference, answer_query, run_trace, compare_DFAs, _find_shortest_trace,
    provide_counter_ex, _convert_trace_string, _produce_win_trace, produce_envs_from_MDP_bits,
    produce_labels_from_map, get_trajectories )

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import pickle
import time
import subprocess


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# INITIALIZE VARIABLES DIRECTLY AND FROM COMMAND LINE
def init(args):
    args_dict = vars(args)
    main_function = args_dict["main_function"]
    fileName_tr_test = args_dict["fileName_tr_test"]
    fileName_opt_trajs = args_dict["fileName_opt_trajs"]
    fileName_grids = args_dict["fileName_grids"]


    # *****************
    if main_function == "produce_grids":
        init_vars = OrderedDict()
        init_vars["main_function"]  = args_dict["main_function"]
        init_vars["fileName_grids"] = fileName_grids
        init_vars["device"] = args_dict["device"] # ["cpu","cuda"][0]

        # PARAMETERS OF DFA
        init_vars["positive_reward"] = 10 # this value doesn't matter cause will be overwritten in produce optimal trajs
        init_vars["negative_reward"] = -10

        # MDP GEOMETRICAL PARAMETERS
        init_vars["n_dim"] = args_dict["n_dim"] # default is 9
        init_vars["n_imp_objs"] = args_dict["n_imp_objs"]
        init_vars["n_obstacles"] = args_dict["n_obstacles"]
        init_vars["imp_obj_idxs_init_det"] = [[(4,7)],[(33,45)]] #FV174 is the goal position on excel(bottom right)
                                    # The first list is for n_dim=9, and second is for 12
                                    # only used when "init_det"
        # init_vars["obstacle_idxs_det"] = [[(8,0)],
        # [(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,0),(5,1),(5,2),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,5),(9,6),(10,5),(10,6),(10,8),
        # (11,2),(11,5),(11,6)]]
        # init_vars["obstacle_idxs_det"] = [[(4,0),(4,1),(4,2),(4,6),(4,7),(4,8)],
        # [(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,0),(5,1),(5,2),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,5),(9,6),(10,5),(10,6),(10,8),
        # (11,2),(11,5),(11,6)]] # only used when "init_det"
        init_vars["obstacle_idxs_det"] = [[(1,1),(7,1),
        (1,7),(7,7),(5,4)],
        [(2,2),(2,3),(2,4),(2,5),(2,7),(2,8),(3,0),(3,1),
        (3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(5,9),(5,10),(5,11),(6,9),
        (6,10),(7,8),(7,9),(7,10),(7,11),(8,0),(8,2),(8,3),(8,6),(8,7),(8,9),(9,0),(10,0),
        (10,2),(11,0)]] #GB174 is the goal position on excel(bottom right)
        # init_vars["imp_obj_idxs_init_det_list"] = [[(10,10)],[(10,10)],[(10,10)],[(10,10)],
        # [(10,10)],[(10,10)],[(9,10)],[(6,10)],[(5,10)]]
        init_vars["imp_obj_idxs_init_det_list"] = [[(4,7)]]
        # init_vars["imp_obj_idxs_init_det_list"] = [[(8,8)],[(7,4)]]
        init_vars["obstacle_idxs_det_list"] = [[(1,1),(7,1),
        (1,7),(7,7),(4,5)]]
        # init_vars["obstacle_idxs_det_list"] = [[(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,1),(5,2),(5,3),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,5),(9,6),(10,5),(10,6),(10,8),
        # (11,2),(11,5),(11,6)],[(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,2),(5,3),(5,4),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,5),(9,6),(10,5),(10,6),(10,8),
        # (11,2),(11,5),(11,6)],[(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,3),(5,4),(5,5),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,5),(9,6),(10,5),(10,6),(10,8),
        # (11,2),(11,5),(11,6)],[(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,1),(5,2),(5,3),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,4),(9,5),(10,4),(10,5),(10,8),
        # (11,2),(11,5),(11,6)],[(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,2),(5,3),(5,4),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,3),(9,4),(10,3),(10,4),(10,8),
        # (11,2),(11,5),(11,6)],[(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,3),(5,4),(5,5),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,2),(9,3),(10,2),(10,3),(10,8),
        # (11,2),(11,5),(11,6)],[(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,1),(5,2),(5,3),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,4),(9,5),(10,4),(10,5),(10,8),
        # (11,2),(11,5),(11,6)],[(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,2),(5,3),(5,4),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,3),(9,4),(10,3),(10,4),(10,8),
        # (11,2),(11,5),(11,6)],[(1,1),(2,8),(2,9),(2,10),(2,11),(3,3),(3,4),
        # (3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(5,3),(5,4),(5,5),(6,2),
        # (7,0),(7,1),(7,2),(8,5),(8,6),(8,8),(8,9),(9,0),(9,2),(9,3),(10,2),(10,3),(10,8),
        # (11,2),(11,5),(11,6)]]

        if args_dict["map_address"]:
            init_vars["map_address"] = args_dict["map_address"]
            init_vars = produce_labels_from_map(init_vars)

        init_vars["obj_type"] = args_dict["obj_type"]
        init_vars["RGB"] = [False,True][0]
        init_vars["pad_element"] = -1

        init_vars["random_start"] = [False, True][1] # when deterministic start is picked, it's always (0,0)
                                                     # otherwise some cell on the boundaries is picked
        init_vars["train_grid_init_type"] = args_dict["train_grid_init_type"] #["init_det","init_random"] # this is for the gridworld, not agent
        # if train_grid_init_type = init_det --> produce only a single training grid

        init_vars["num_grids_tr"] = args_dict["num_grids_tr"]
        init_vars["num_grids_test"] = args_dict["num_grids_test"]
        init_vars["num_grids_test"] = len(init_vars["imp_obj_idxs_init_det_list"])


        create_dir_grids(fileName_grids)
        grid_address = return_address_grids(fileName_grids)
        param_file = "params_" + init_vars["main_function"]
        with open(grid_address+param_file,'w') as f:
            for key, value in init_vars.items():
                f.write('{0}: {1}\n'.format(key, value))
        with open(grid_address + "init_vars.pkl","wb") as f:
            pickle.dump(init_vars,f)

    elif main_function in ["produce_opt_trajs", "produce_opt_trajs_time_limited", "produce_opt_trajs_time_limited_bits", "produce_non_opt_trajs",
                "produce_non_opt_trajs_time_limited", "produce_non_opt_trajs_time_limited_no_fail"]:
        # fill this part only if this condition is satisfied

        grids_address = return_address_grids(fileName_grids)
        logs_address, models_address = return_addresses_logs_models(fileName_tr_test)
        with open(logs_address + "init_vars.pkl","rb") as f:
            init_vars = pickle.load(f)


        init_vars["main_function"]  = args_dict["main_function"]



        trajs_address = init_vars["trajs_address"]
        param_file = "params_" + init_vars["main_function"]
        create_dir_opt_trajs(trajs_address)
        with open(trajs_address+param_file,'w') as f:
            for key, value in init_vars.items():
                f.write('{0}: {1}\n'.format(key, value))
        with open(trajs_address + "init_vars.pkl","wb") as f:
            pickle.dump(init_vars,f)



    elif main_function in ["train", "train_IRL_active", "train_from_inferred_DFA", "train_base_memoryless","train_IRL_bits", "train_base", "train_incomp", "train_non_local_r", "train_BC_v1","train_BC_v2"
            ,"train_BC_3","train_BC_local"] :
        if args_dict["run_mode"] == "continue":
            # 1234567891011 NEEDS TO BE COMPLETED
            pass
            # logs_address, models_address = return_addresses_logs_models(fileName_tr_test)
            # with open(logs_address +'init_vars.pkl', 'rb') as f:
            #     init_vars = pickle.load(f)
            # init_vars["run_mode"] = args_dict["run_mode"]
        else:
            grids_address = return_address_grids(fileName_grids)
            with open(grids_address +'init_vars.pkl', 'rb') as f:
                init_vars = pickle.load(f)

            # you should manually fill these items for any train run
            init_vars["train NOTE"] = ""
            init_vars["fileName_tr_test"] = args_dict["fileName_tr_test"]
            init_vars["fileName_opt_trajs"] = args_dict["fileName_opt_trajs"]
            init_vars["main_function"]  = args_dict["main_function"]
            init_vars["training_function_name"] = args_dict["main_function"]
            init_vars["num_theta_updates"] = args_dict["num_theta_updates"]
            init_vars["device"] = args_dict["device"]
            init_vars["save_model_interval"] = args_dict["save_model_interval"]
            init_vars["num_trajs_used"] = args_dict["num_trajs_used"]
            init_vars["run_mode"] = args_dict["run_mode"] # ["restart", "continue"]
            init_vars["num_runs"] = args_dict["num_runs"]
            init_vars["verbose"] = [False, True][0]
            init_vars["val_period"] = 20

            # traj parameters:
            init_vars["num_optimal_trajs"] = args_dict["num_optimal_trajs"]

            init_vars["thresh_Q_optimal"] = 0.0000001
            init_vars["thresh_y_optimal"] = 0.0000001
            init_vars["random_start"] = [False, True][1] # when deterministic start is picked, it's always (0,0)
                                                         # otherwise some cell on the boundaries is picked
            init_vars["positive_reward"] = 10
            init_vars["negative_reward"] = -10

            # OPTIMAL TRAJECTORY VARIABLES
            init_vars["trajs_address"] = return_address_opt_trajs(grids_address,init_vars["fileName_opt_trajs"], init_vars["fileName_tr_test"])
            init_vars["traj_address"] = args_dict["traj_address"]

            if args_dict["fileName_opt_trajs"] in ["produce_opt_trajs", "produce_opt_trajs_time_limited", "produce_opt_trajs_time_limited_bits"]:
                init_vars["optimal_policy_type"] = ["soft","greedy","random"][0]
                if args_dict["fileName_opt_trajs"] in ["produce_opt_trajs_time_limited", "produce_opt_trajs_time_limited_bits"]:
                    init_vars["time_limit"] = args_dict["time_limit"]
            elif args_dict["fileName_opt_trajs"] in ["produce_non_opt_trajs", "produce_non_opt_trajs_time_limited",
                                "produce_non_opt_trajs_time_limited_no_fail"]:
                init_vars["optimal_policy_type"] = ["soft","greedy","random"][2]
                init_vars["time_limit"] = args_dict["time_limit"]

            # update DFA type if it's provided in the command line
            if args_dict["GT_dfa_address"]:
                init_vars["GT_dfa_address"] = args_dict["GT_dfa_address"]
                init_vars["tr_dfa_address"] =  "./inferred_DFAs/automaton_" + init_vars["fileName_tr_test"] + ".txt"

            if args_dict["tr_dfa_address"]:
                init_vars["tr_dfa_address"] = args_dict["tr_dfa_address"]

            # OPTIMIZATION HYPERPARAMETERS
            init_vars["lr"] = args_dict["lr"]
            init_vars["lam_phi"] = args_dict["lam_phi"] # to balance L_\phi and L_D
            init_vars["lam_reg"] = args_dict["lam_reg"] # to regularize weights of the neural network
            init_vars["gamma"] = args_dict["gamma"]


            if main_function in ["train_BC_v1","train_BC_v2", "train_BC_3", "train_BC_local"]:
                init_vars["policy_net"] = args_dict["policy_net"]
                init_vars["batch_size"] = args_dict["batch_size"]
                init_vars["policy_loss"] = ["CrossEntropyLoss"][0]
                init_vars["policy_optimizer"] = ["Adam", "RMSprop", "SGD"][0]
                init_vars["policy_input_size"] = args_dict["policy_input_size"]


            elif main_function == 'train_non_local_r':
                init_vars["reward_net"] = ["MLP", "CNN", "transformer"][1]
                # note that in this case the whole gridworld is fed into the reward function
            else:
                init_vars["reward_net"] = ["MLP", "CNN", "transformer"][0]
                init_vars["reward_input_size"] = args_dict["reward_input_size"]

            # DP THRESHOLDS
            init_vars["thresh_Q_theta"] = 0.001
            init_vars["thresh_grad_Q_theta"] = 0.0001
            init_vars["thresh_Y_theta"] = 0.0001
            init_vars["thresh_grad_Y_theta"] = 0.000001
            # init_vars["thresh_Q_theta"] = 1e-8
            # init_vars["thresh_grad_Q_theta"] = 1e-8
            # init_vars["thresh_Y_theta"] = 1e-10
            # init_vars["thresh_grad_Y_theta"] = 1e-8
            # ^^^^^^^^ end of init_vars ^^^^^^^^^



            # if main_function in ["produce_opt_trajs_time_limited_bits"]:
            #     grids_address = return_address_grids(fileName_grids)
            #     logs_address, models_address = return_addresses_logs_models(fileName_tr_test)
            #     trajs_address = init_vars["trajs_address"]
            #     param_file = "params_" + init_vars["main_function"]
            #     create_dir_opt_trajs(trajs_address)
            #     with open(trajs_address+param_file,'w') as f:
            #         for key, value in init_vars.items():
            #             f.write('{0}: {1}\n'.format(key, value))
            #     with open(trajs_address + "init_vars.pkl","wb") as f:
            #         pickle.dump(init_vars,f)
            #     return init_vars


            logs_address, models_address = return_addresses_logs_models(fileName_tr_test)
            param_file = "params_" + init_vars["main_function"]
            create_dir_logs_and_models(init_vars)
            with open(logs_address+param_file,'w') as f:
                for key, value in init_vars.items():
                    f.write('{0}: {1}\n'.format(key, value))
            with open(models_address+param_file,'w') as f:
                for key, value in init_vars.items():
                    f.write('{0}: {1}\n'.format(key, value))
            with open(logs_address + "init_vars.pkl","wb") as f:
                pickle.dump(init_vars,f)

    elif main_function in ["test", "test_active", "test_bits", "test_memoryless_baseline","test_non_local_r", "test_base", "test_BC_3",  "test_BC_v1",  "test_BC_v2", "test_BC_local"]:
        logs_address, models_address = return_addresses_logs_models(fileName_tr_test)
        with open(logs_address +'init_vars.pkl', 'rb') as f:
            init_vars = pickle.load(f)
        init_vars["main_function"]  = args_dict["main_function"]
        init_vars["fileName_grids"] = args_dict["fileName_grids"]
        # MDP GEOMETRICAL PARAMETERS
        init_vars["n_dim"] = args_dict["n_dim"] # default is 9
        init_vars["test_type"] = args_dict["test_type"]
        # init_vars["init_type"] = args_dict["init_type"] # ["init_train", "init_det", "init_random", "init_finite_random"][0] # this is for the gridworld, not agent
        # init_vars["imp_obj_idxs_init_det"] = [[1,3,8],[11,1,4]]
        #                             # The first list is for n_dim=9, and second is for 12
        #                             # only used when "init_det"
        # init_vars["obstacle_idxs_det"] = [[4],[2,5,8]] # only used when "init_det"

        create_dir_test(init_vars)
        test_address = return_address_test(init_vars)
        with open(test_address + "init_vars.pkl","wb") as f:
            pickle.dump(init_vars,f)

    init_vars["Transition"] = namedtuple("Transition",
        ["current_state", "dfa_state", "action_idx", "next_current_state", "next_dfa_state",
                "reward", "done"])

    return init_vars


# PRODUCING GRIDS AND TRAJS
# ---------------------------
def produce_grids(init_vars):
    # produces random train and test trajectories corresponding to a specific optimal trajectory set

    mdp_objs_tr = []
    mdp_objs_test = []

    if init_vars["train_grid_init_type"] == "init_det":
        # only one training grid will be produced deterministically
        init_vars["init_type"] = init_vars["train_grid_init_type"]
        print("start making training pre-determined grid")
        mdp = produce_mdp(init_vars)
        mdp_objs_tr.append(copy.deepcopy(mdp)) # test loop
    else:
        # multiple training grids will be produced randomly
        init_vars["init_type"] = init_vars["train_grid_init_type"]
        print("start making training grids")
        for i in range(init_vars["num_grids_tr"]):
            print(f"number: {i}")
            mdp = produce_mdp(init_vars)
            mdp_objs_tr.append(copy.deepcopy(mdp)) # test loop

    print("start making test grids")
    # init_vars["init_type"] = "init_random"
    for i in range(init_vars["num_grids_test"]):
        print(f"number: {i}")
        init_vars["obstacle_idxs_det"][0] = init_vars["obstacle_idxs_det_list"][i]
        init_vars["imp_obj_idxs_init_det"][0] = init_vars["imp_obj_idxs_init_det_list"][i]
        mdp = produce_mdp(init_vars)
        mdp_objs_test.append(copy.deepcopy(mdp))
    grid_address = return_address_grids(init_vars["fileName_grids"])

    with open(grid_address + "mdp_objs_tr.pkl","wb") as f:
        pickle.dump(mdp_objs_tr,f)
    with open(grid_address + "mdp_objs_test.pkl","wb") as f:
        pickle.dump(mdp_objs_test,f)
    print ("Finished writting files")

def produce_opt_trajs(init_vars):
    device = torch.device(init_vars["device"])
    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # creating the folders
    trajs_address = init_vars["trajs_address"]
    all_trajs = []
    all_traces = []

    for mdp_counter, mdp in enumerate(mdp_objs_tr[:1]):
        print (f"### gridworld number: {mdp_counter+1}")
        envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp))
        MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
        agent = produce_agent(envs, init_vars, agent_mode = "trajs")
        game_won_counter = 0
        game_over_counter = 0
        trajs = []
        traces = []
        for traj_number in range(init_vars["num_optimal_trajs"]):
            traj = []
            trace = []
            print("\n\n\n\n### traj_number " + str(traj_number) + "###")
            old_mdp_state, old_dfa_state = PA_obj.reset()

            reward = 0
            terminal = False
            traj_steps = 0

            while not terminal:
                traj_steps += 1
                # print ("current_traj_steps:",  traj_steps)
                if init_vars["optimal_policy_type"] == "soft":
                    action_idx, action = agent.select_optimal_action_soft() # cntr selects an action among permitable actions
                elif init_vars["optimal_policy_type"] == "greedy":
                    action_idx, action = agent.select_optimal_action_greedy()
                elif init_vars["optimal_policy_type"] == "random":
                    action_idx, action = agent.random_action_selection()

                traj.append((reward, old_mdp_state, old_dfa_state, action_idx))

                next_mdp_state, next_dfa_state, reward, _ = PA_obj.step(action_idx)

                # print ("---------- action_probs: {}".format(action_probs))
                # print(str((state,action)) + "; ")

                i,j = next_mdp_state[0], next_mdp_state[1]

                neigh = MDP_obj.neigh_select(i,j)
                match, cl, class_attribution_error = MDP_obj.temp_match(neigh)
                if match:
                    trace.append(cl)

                if init_vars["verbose"]:
                    pass
                    # print ("state before action ----> " + "[" + str(old_mdp_state[0]) +", " +
                    #         str(old_mdp_state[1]) + "]" )
                    # print ("action ----> "  + action)
                    # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                    # print ("---------------------")

                game_over, game_won = PA_obj.is_terminal()
                terminal = game_over or game_won # terminal refers to next state

                if game_over:
                    game_over_counter += 1
                    if init_vars["verbose"]:
                        print("GAME OVER!!!")
                    traj.append("game over")
                    # trace.append("game over")


                if game_won:
                    game_won_counter += 1
                    if init_vars["verbose"]:
                        print("GAME WON!!!")
                    traj.append("game won")
                    # trace.append("game won")


                old_mdp_state, old_dfa_state = copy.deepcopy(next_mdp_state), next_dfa_state
            # THIS IS THE END OF ONE trajectory
            trajs.append(traj)
            traces.append(trace)

        success_ratio = game_won_counter/(game_won_counter+(game_over_counter+time_over))
        if init_vars["verbose"]:
            print("success_ratio: ", success_ratio)
        all_trajs.append(trajs)
        all_traces.append(traces)

    with open(trajs_address + 'all_trajectories.pkl', 'wb') as f:
        pickle.dump(all_trajs, f)

    with open(trajs_address + 'all_traces.pkl', 'wb') as f:
        pickle.dump(all_traces, f)

def produce_opt_trajs_time_limited(init_vars):
    device = torch.device(init_vars["device"])
    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # creating the folders
    trajs_address = init_vars["trajs_address"]
    all_trajs = []
    all_traces = []

    for mdp_counter, mdp in enumerate(mdp_objs_tr[:1]):
        if init_vars["verbose"]:
            print (f"### gridworld number: {mdp_counter+1}")
        envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp))
        MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
        agent = produce_agent(envs, init_vars, agent_mode = "trajs")
        game_won_counter = 0
        game_over_counter = 0
        time_over_counter = 0
        trajs = []
        traces = []
        traj_steps_limit = init_vars["time_limit"]

        for traj_number in range(init_vars["num_optimal_trajs"]):
            traj = []
            trace = []
            if init_vars["verbose"]:
                print("\n\n\n\n### traj_number " + str(traj_number) + "###")
            old_mdp_state, old_dfa_state = PA_obj.reset()

            reward = 0
            terminal = False
            game_over = False
            game_won = False

            # print(agent.y_optimal[:,:,0])
            # fig, ax = plt.subplots()
            # im = ax.imshow(agent.y_optimal[:,:,0], norm=mc.Normalize(vmin=0))
            # cbar = ax.figure.colorbar(im, ax=ax)
            # cbar.ax.set_ylabel('', rotation=-90, va="bottom")
            # plt.show()
            # exit()

            # with open('soft_policy_12_corner_100.pkl', 'wb') as f:
            #     pickle.dump(agent.pi_optimal_soft, f)
            # exit()

            traj_steps = 1
            while not terminal and traj_steps < traj_steps_limit:
                if init_vars["optimal_policy_type"] == "soft":
                    action_idx, action = agent.select_optimal_action_soft() # cntr selects an action among permitable actions
                elif init_vars["optimal_policy_type"] == "greedy":
                    action_idx, action = agent.select_optimal_action_greedy()
                elif init_vars["optimal_policy_type"] == "random":
                    action_idx, action = agent.random_action_selection()

                traj.append((reward, old_mdp_state, old_dfa_state, action_idx))


                next_mdp_state, next_dfa_state, reward, _ = PA_obj.step(action_idx)
                # print(f"old_dfa_state: {old_dfa_state},    next_dfa_state: {next_dfa_state}")
                # print ("---------- action_probs: {}".format(action_probs))
                # print(str((state,action)) + "; ")

                i,j = next_mdp_state[0], next_mdp_state[1]

                neigh = MDP_obj.neigh_select(i,j)
                match, cl, class_attribution_error = MDP_obj.temp_match(neigh)
                if match:
                    trace.append(cl)

                if init_vars["verbose"]:
                    pass
                    # print ("state before action ----> " + "[" + str(old_mdp_state[0]) +", " +
                    #         str(old_mdp_state[1]) + "]" )
                    # print ("action ----> "  + action)
                    # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                    # print ("---------------------")

                game_over, game_won = PA_obj.is_terminal()
                if not terminal:
                    terminal = game_over or game_won # terminal refers to next state

                    if game_over:
                        game_over_counter += 1
                        if init_vars["verbose"]:
                            print("GAME OVER!!!")
                        traj.append("game over")

                        # trace.append("game over")

                    elif game_won:
                        game_won_counter += 1
                        if init_vars["verbose"]:
                            print("GAME WON!!!")
                        traj.append("game won")
                        # trace.append("game won")

                old_mdp_state, old_dfa_state = copy.deepcopy(next_mdp_state), next_dfa_state
                traj_steps += 1

            # THIS IS THE END OF ONE trajectory
            if not terminal and traj_steps >= traj_steps_limit:
                time_over_counter += 1
                if init_vars["verbose"]:
                    print("Time Over!!!")
                traj.append("time over")
            if game_won:
                trajs.append(traj)
                traces.append(trace)


        success_ratio = game_won_counter/(game_won_counter+(game_over_counter+time_over_counter))
        if init_vars["verbose"]:
            print("success_ratio optimal policy: ", success_ratio)
        all_trajs.append(trajs)
        all_traces.append(traces)

    with open(trajs_address + 'success_rate.txt', 'w') as f:
        f.write(f"success_ratio: {success_ratio:.4}\n")

    with open(trajs_address + 'all_trajectories.pkl', 'wb') as f:
        pickle.dump(all_trajs, f)

    with open(trajs_address + 'all_traces.pkl', 'wb') as f:
        pickle.dump(all_traces, f)

def produce_opt_trajs_time_limited_bits(init_vars):

    device = torch.device(init_vars["device"])
    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # creating the folders
    trajs_address = init_vars["trajs_address"]
    all_trajs = []
    all_trajs_bits = []
    all_traces = []

    for mdp_counter, mdp in enumerate(mdp_objs_tr[:1]):
        if init_vars["verbose"]:
            print (f"### gridworld number: {mdp_counter+1}")
        MDP_obj, DFA_obj_bits, PA_obj_bits, DFA_obj_GT, PA_obj_GT = produce_envs_from_MDP_bits(init_vars, copy.deepcopy(mdp))
        envs_GT = MDP_obj, DFA_obj_GT, PA_obj_GT, DFA_obj_GT, PA_obj_GT
        agent = produce_agent(envs_GT, init_vars, agent_mode = "trajs")
        game_won_counter = 0
        game_over_counter = 0
        time_over_counter = 0
        trajs = []
        trajs_bits = []
        traces = []
        traj_steps_limit = init_vars["time_limit"]

        for traj_number in range(init_vars["num_optimal_trajs"]):
            traj = []
            traj_bits = []
            trace = []
            # print("\n\n### traj_number " + str(traj_number) + "###")
            old_mdp_state, old_dfa_state = PA_obj_GT.reset()
            PA_obj_bits.set_MDP_state(old_mdp_state[0], old_mdp_state[1])
            old_mdp_state_bits = PA_obj_bits.MDP.mdp_state
            old_dfa_state_bits_arr = PA_obj_bits.reset_DFA_only()
            old_dfa_state_bits = DFA_obj_bits.convert_dfa_state_to_int(old_dfa_state_bits_arr)

            reward = 0
            terminal = False
            game_over = False
            game_won = False

            traj_steps = 1
            while not terminal and traj_steps < traj_steps_limit:
                if init_vars["optimal_policy_type"] == "soft":
                    action_idx, action = agent.select_optimal_action_soft() # cntr selects an action among permitable actions
                elif init_vars["optimal_policy_type"] == "greedy":
                    action_idx, action = agent.select_optimal_action_greedy()
                elif init_vars["optimal_policy_type"] == "random":
                    action_idx, action = agent.random_action_selection()
                traj.append((reward, old_mdp_state, old_dfa_state, action_idx))
                traj_bits.append((reward, old_mdp_state, old_dfa_state_bits, action_idx))
                next_mdp_state, next_dfa_state, reward, _ = PA_obj_GT.step(action_idx)
                next_mdp_state_bits, next_dfa_state_bits_arr, reward, _ = PA_obj_bits.step(action_idx)
                next_dfa_state_bits = DFA_obj_bits.convert_dfa_state_to_int(next_dfa_state_bits_arr)
                assert next_mdp_state[0] == next_mdp_state_bits[0]
                assert next_mdp_state[1] == next_mdp_state_bits[1]

                # print(f"old_dfa_state: {old_dfa_state},    next_dfa_state: {next_dfa_state}")
                # print ("---------- action_probs: {}".format(action_probs))
                # print(str((state,action)) + "; ")

                i,j = next_mdp_state[0], next_mdp_state[1]
                neigh = MDP_obj.neigh_select(i,j)
                match, cl, class_attribution_error = MDP_obj.temp_match(neigh)
                if match:
                    trace.append(cl)



                if init_vars["verbose"]:
                    pass
                    # print ("state before action ----> " + "[" + str(old_mdp_state[0]) +", " +
                    #         str(old_mdp_state[1]) + "]" )
                    # print ("action ----> "  + action)
                    # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                    # print ("---------------------")

                game_over, game_won = PA_obj_GT.is_terminal()
                if not terminal:
                    terminal = game_over or game_won # terminal refers to next state

                    if game_over:
                        game_over_counter += 1
                        if init_vars["verbose"]:
                            print("GAME OVER!!!")
                        traj.append("game over")
                        traj_bits.append("game over")
                        # trace.append("game over")

                    elif game_won:
                        game_won_counter += 1
                        if init_vars["verbose"]:
                            print("GAME WON!!!")
                        traj.append("game won")
                        traj_bits.append("game won")
                        # trace.append("game won")

                old_mdp_state, old_dfa_state = copy.deepcopy(next_mdp_state), next_dfa_state
                old_dfa_state_bits = next_dfa_state_bits
                old_dfa_state_bits_arr = copy.deepcopy(next_dfa_state_bits_arr)

                traj_steps += 1

            # THIS IS THE END OF ONE trajectory
            if not terminal and traj_steps >= traj_steps_limit:
                time_over_counter += 1
                if init_vars["verbose"]:
                    print("Time Over!!!")
                traj.append("time over")
                traj_bits.append("time over")

            if game_won:
                trajs.append(traj)
                trajs_bits.append(traj_bits)
                traces.append(trace)

        success_ratio = game_won_counter/(game_won_counter+(game_over_counter+time_over_counter))
        if init_vars["verbose"]:
            print("success_ratio optimal policy: ", success_ratio)
        all_trajs.append(trajs)
        all_trajs_bits.append(trajs_bits)
        all_traces.append(traces)

    with open(trajs_address + 'all_trajectories.pkl', 'wb') as f:
        pickle.dump(all_trajs, f)

    with open(trajs_address + 'all_trajectories_bits.pkl', 'wb') as f:
        pickle.dump(all_trajs_bits, f)

    with open(trajs_address + 'all_traces.pkl', 'wb') as f:
        pickle.dump(all_traces, f)

def produce_non_opt_trajs(init_vars):
    device = torch.device(init_vars["device"])
    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)

    # creating the folders
    trajs_address = init_vars["trajs_address"]
    all_trajs = []
    all_traces = []

    for mdp_counter, mdp in enumerate(mdp_objs_tr[:1]):
        print (f"### gridworld number: {mdp_counter+1}")
        envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp))
        MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
        agent = produce_agent(envs, init_vars, agent_mode = "trajs")
        game_won_counter = 0
        game_over_counter = 0
        trajs = []
        traces = []
        for traj_number in range(init_vars["num_optimal_trajs"]):
            traj = []
            trace = []
            print("\n\n\n\n### traj_number " + str(traj_number) + "###")
            old_mdp_state, old_dfa_state = PA_obj.reset()

            reward = 0
            terminal = False
            traj_steps = 0

            while not terminal:
                traj_steps += 1
                # print ("current_traj_steps:",  traj_steps)
                if init_vars["optimal_policy_type"] == "soft":
                    action_idx, action = agent.select_optimal_action_soft() # cntr selects an action among permitable actions
                elif init_vars["optimal_policy_type"] == "greedy":
                    action_idx, action = agent.select_optimal_action_greedy()
                elif init_vars["optimal_policy_type"] == "random":
                    action_idx, action = agent.random_action_selection()

                traj.append((reward, old_mdp_state, old_dfa_state, action_idx))

                next_mdp_state, next_dfa_state, reward, _ = PA_obj.step(action_idx)

                # print ("---------- action_probs: {}".format(action_probs))
                # print(str((state,action)) + "; ")

                i,j = next_mdp_state[0], next_mdp_state[1]

                neigh = MDP_obj.neigh_select(i,j)
                match, cl, class_attribution_error = MDP_obj.temp_match(neigh)
                if match:
                    trace.append(cl)

                if init_vars["verbose"]:
                    pass
                    # print ("state before action ----> " + "[" + str(old_mdp_state[0]) +", " +
                    #         str(old_mdp_state[1]) + "]" )
                    # print ("action ----> "  + action)
                    # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                    # print ("---------------------")

                game_over, game_won = PA_obj.is_terminal()

                if not terminal:
                    terminal = game_over or game_won # terminal refers to next state

                    if game_over:
                        game_over_counter += 1
                        print("GAME OVER!!!")
                        traj.append("game over")
                        trace.append("game over")

                    if game_won:
                        game_won_counter += 1
                        print("GAME WON!!!")
                        traj.append("game won")
                        trace.append("game won")

                old_mdp_state, old_dfa_state = copy.deepcopy(next_mdp_state), next_dfa_state
            # THIS IS THE END OF ONE trajectory
            trajs.append(traj)
            traces.append(trace)

        success_ratio = game_won_counter/(game_won_counter+game_over_counter)
        print("success_ratio: ", success_ratio)
        all_trajs.append(trajs)
        all_traces.append(traces)

    with open(trajs_address + 'all_trajectories_non_opt.pkl', 'wb') as f:
        pickle.dump(all_trajs, f)

    with open(trajs_address + 'all_traces_non_opt.pkl', 'wb') as f:
        pickle.dump(all_traces, f)

def produce_non_opt_trajs_time_limited(init_vars):
    device = torch.device(init_vars["device"])
    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)

    # creating the folders
    trajs_address = init_vars["trajs_address"]
    all_trajs = []
    all_traces = []

    for mdp_counter, mdp in enumerate(mdp_objs_tr[:1]):
        print (f"### gridworld number: {mdp_counter+1}")
        envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp))
        MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
        agent = produce_agent(envs, init_vars, agent_mode = "trajs")
        game_won_counter = 0
        game_over_counter = 0
        time_over_counter = 0

        trajs = []
        traces = []
        for traj_number in range(init_vars["num_optimal_trajs"]):
            win_or_lose = None
            game_won = False

            traj = []
            trace = []
            print("\n\n\n\n### traj_number " + str(traj_number) + "###")
            old_mdp_state, old_dfa_state = PA_obj.reset()

            reward = 0
            terminal = False


            traj_steps_limit = init_vars["time_limit"]
            traj_steps = 1

            while not game_won and traj_steps < traj_steps_limit:

                # print ("current_traj_steps:",  traj_steps)
                if init_vars["optimal_policy_type"] == "soft":
                    action_idx, action = agent.select_optimal_action_soft() # cntr selects an action among permitable actions
                elif init_vars["optimal_policy_type"] == "greedy":
                    action_idx, action = agent.select_optimal_action_greedy()
                elif init_vars["optimal_policy_type"] == "random":
                    action_idx, action = agent.random_action_selection()

                traj.append((reward, old_mdp_state, old_dfa_state, action_idx))

                next_mdp_state, next_dfa_state, reward, _ = PA_obj.step(action_idx)

                # print ("---------- action_probs: {}".format(action_probs))
                # print(str((state,action)) + "; ")

                i,j = next_mdp_state[0], next_mdp_state[1]

                neigh = MDP_obj.neigh_select(i,j)
                match, cl, class_attribution_error = MDP_obj.temp_match(neigh)
                if match:
                    trace.append(cl)

                if init_vars["verbose"]:
                    pass
                    # print ("state before action ----> " + "[" + str(old_mdp_state[0]) +", " +
                    #         str(old_mdp_state[1]) + "]" )
                    # print ("action ----> "  + action)
                    # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                    # print ("---------------------")

                game_over, game_won = PA_obj.is_terminal()


                if not terminal:
                    terminal = game_over or game_won # terminal refers to next state

                    if game_over:
                        game_over_counter += 1
                        # print("GAME OVER!!!")
                        win_or_lose = 'game over'
                        # traj.append("game over")
                        # trace.append("game over")

                    if game_won:
                        game_won_counter += 1
                        # print("GAME WON!!!")
                        win_or_lose = 'game won'
                        # traj.append("game won")
                        # trace.append("game won")

                old_mdp_state, old_dfa_state = copy.deepcopy(next_mdp_state), next_dfa_state
                traj_steps += 1

            # THIS IS THE END OF ONE trajectory
            if not terminal and traj_steps >= traj_steps_limit:
                win_or_lose = 'time over'
                time_over_counter += 1
            print (win_or_lose)
            traj.append(win_or_lose)
            trace.append(win_or_lose)
            if win_or_lose in ['game over', 'time over']:
                trajs.append(traj)
                traces.append(trace)


        success_ratio = game_won_counter/(game_won_counter+game_over_counter+time_over_counter)
        print("success_ratio: ", success_ratio)
        all_trajs.append(trajs)
        all_traces.append(traces)


    with open(trajs_address + 'all_trajectories_non_opt.pkl', 'wb') as f:
        pickle.dump(all_trajs, f)

    with open(trajs_address + 'all_traces_non_opt.pkl', 'wb') as f:
        pickle.dump(all_traces, f)

def produce_non_opt_trajs_time_limited_no_fail(init_vars):
    device = torch.device(init_vars["device"])
    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)

    # creating the folders
    trajs_address = init_vars["trajs_address"]
    all_trajs = []
    all_traces = []

    for mdp_counter, mdp in enumerate(mdp_objs_tr[:1]):
        print (f"### gridworld number: {mdp_counter+1}")
        envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp))
        MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
        agent = produce_agent(envs, init_vars, agent_mode = "trajs")
        game_won_counter = 0
        game_over_counter = 0
        time_over_counter = 0

        trajs = []
        traces = []
        for traj_number in range(init_vars["num_optimal_trajs"]):
            win_or_lose = None
            game_won = False

            traj = []
            trace = []
            print("\n\n\n\n### traj_number " + str(traj_number) + "###")
            old_mdp_state, old_dfa_state = PA_obj.reset()

            reward = 0
            terminal = False


            traj_steps_limit = init_vars["time_limit"]
            traj_steps = 1

            while not game_won and traj_steps < traj_steps_limit:

                # print ("current_traj_steps:",  traj_steps)
                if init_vars["optimal_policy_type"] == "soft":
                    action_idx, action = agent.select_optimal_action_soft() # cntr selects an action among permitable actions
                elif init_vars["optimal_policy_type"] == "greedy":
                    action_idx, action = agent.select_optimal_action_greedy()
                elif init_vars["optimal_policy_type"] == "random":
                    action_idx, action = agent.random_action_selection()

                traj.append((reward, old_mdp_state, old_dfa_state, action_idx))

                next_mdp_state, next_dfa_state, reward, _ = PA_obj.step(action_idx)

                # print ("---------- action_probs: {}".format(action_probs))
                # print(str((state,action)) + "; ")

                i,j = next_mdp_state[0], next_mdp_state[1]

                neigh = MDP_obj.neigh_select(i,j)
                match, cl, class_attribution_error = MDP_obj.temp_match(neigh)
                if match:
                    trace.append(cl)

                if init_vars["verbose"]:
                    pass
                    # print ("state before action ----> " + "[" + str(old_mdp_state[0]) +", " +
                    #         str(old_mdp_state[1]) + "]" )
                    # print ("action ----> "  + action)
                    # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                    # print ("---------------------")

                _, game_won = PA_obj.is_terminal()


                if not terminal:
                    terminal =  game_won # terminal refers to next state

                    # if game_over:
                    #     game_over_counter += 1
                    #     # print("GAME OVER!!!")
                    #     win_or_lose = 'game over'
                    #     # traj.append("game over")
                    #     # trace.append("game over")

                    if game_won:
                        game_won_counter += 1
                        # print("GAME WON!!!")
                        win_or_lose = 'game won'
                        # traj.append("game won")
                        # trace.append("game won")

                old_mdp_state, old_dfa_state = copy.deepcopy(next_mdp_state), next_dfa_state
                traj_steps += 1

            # THIS IS THE END OF ONE trajectory
            if not terminal and traj_steps >= traj_steps_limit:
                win_or_lose = 'time over'
                time_over_counter += 1
            print (win_or_lose)
            traj.append(win_or_lose)
            trace.append(win_or_lose)
            if win_or_lose in ['time over']:
                trajs.append(traj)
                traces.append(trace)

        success_ratio = game_won_counter/(game_won_counter+game_over_counter+time_over_counter)
        print("success_ratio: ", success_ratio)
        all_trajs.append(trajs)
        all_traces.append(traces)
    with open(trajs_address + 'all_trajectories_non_opt_time_limited_no_fail.pkl', 'wb') as f:
        pickle.dump(all_trajs, f)

    with open(trajs_address + 'all_traces_non_opt_time_limited_no_fail.pkl', 'wb') as f:
        pickle.dump(all_traces, f)


# TRAINING
# ---------------------------
def train_old(init_vars):
    """
    this is the old version of the trian funciton wihch has been retained to understand old runs
    This function trains the reward network on a single gridworld and it's corresponding
    set of maxEnt trajectories. That gridworld is exactly the same as the one for which optimal
    trajectories have been produced.
    """
    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]
    envs = produce_envs(init_vars)
    agent = produce_agent(envs, init_vars)
    MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]

    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    agent.logs_address = logs_address


    if init_vars["run_mode"]  == "continue":
        # Load models
        if agent.reward_net_name == "MLP":
            agent.reward_net = agent_rl2.net_MLP(envs[0], n_dim=3, out_dim=1).to(device)
        if agent.reward_net_name == "CNN":
            pass
        agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))

    # save the gridworld to be used for eval later
    with open(logs_address + "MDP_train.pkl", "wb") as f:
        pickle.dump(MDP_obj, f)


    agent.demo_visitation_calculator(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts
    if init_vars["run_mode"] == "restart":
        with open(logs_address + "loss_per_episode.txt", "w") as file:
            file.write("L_D_theta, L_phi, grad_L_D_theta_norm_one, episode_number, episode_time \n")
        with open(logs_address + "output_progress.txt", "w") as file:
            file.write("output progress")
        min_range = 0
        max_range = init_vars["num_theta_updates"]
    elif init_vars["run_mode"] == "continue":
        loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
        min_range = int(loss[-1,3])
        max_range = min_range + init_vars["num_theta_updates"]
        loss = 0

    for epis in range(min_range, max_range):
        start = time.time()
        Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
        agent.calc_pi_theta() # Policy is defined using softmax from pi_theta
        agent.calc_grad_Q_theta(thresh=init_vars["thresh_Q_theta"])
        grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value
        agent.calc_grad_pi_theta()

        # following 4 lines only necessary if we want to modify the cost function
        # using specifications
        # ---------------------
        agent.calc_y_theta(thresh=init_vars["thresh_Y_theta"])
        agent.calc_grad_y_theta(thresh=init_vars["thresh_grad_Y_theta"])
        grad_L_phi_theta = agent.calc_grad_L_phi_theta()
        # ---------------------
        grad_j = grad_L_D_theta - lam_reg * reg_term + lam_phi * grad_L_phi_theta
        agent.update_theta(grad_j)
        L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])
        L_phi, min_y =  agent.calc_L_phi_theta()
        end = time.time()
        epis_time = end-start

        grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
        grad_L_phi_theta_norm_one = np.mean(np.abs(grad_L_phi_theta))
        reg_term_norm_one = np.mean(np.abs(reg_term))

        if init_vars["verbose"] == True:
            print ("****************************************")
            print (f"episode number:  {epis}")
            print (f"L_D_theta: {L_D_theta:.4}")
            print (f"L_phi: {L_phi:.4}")
            print (f"epis_time: {epis_time:.4}")
            print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
            print (f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}")
            print (f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}")
            # print ("reward_net_time: {:.4}".format(reward_net_time))
            # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
            # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
            # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
            # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
            # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
            # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
            # print ("update_theta_TIME: {:.4}".format(s9-s8))

        with open(logs_address + "output_progress.txt", "a") as file2:
            file2.write("****************************************\n")
            file2.write(f"episode number:  {epis}\n")
            file2.write(f"L_D_theta: {L_D_theta:.4}\n")
            file2.write(f"L_phi: {L_phi:.4}\n")
            file2.write(f"epis_time: {epis_time:.4}\n")
            file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
            file2.write(f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}\n")
            file2.write(f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}\n")

        with open(logs_address + "loss_per_episode.txt", "a") as file1:
            file1.write(f'{L_D_theta:>20} {L_phi:>20} {min_y:>15} {grad_L_D_theta_norm_one:.10} {grad_L_phi_theta_norm_one:.10} {reg_term_norm_one:.10} {epis} {epis_time:.4} \n')

        if epis != 0 and epis == 20 or epis % init_vars["save_model_interval"] == 0:
            if init_vars["verbose"] == True:
                print ("SAVING THE MODELS .............")
                print ()
            torch.save(agent.reward_net.state_dict(), models_address + "reward_net.pt")
            # torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + str(epis) + ".pt")

def train(init_vars):
    """
    This function trains the reward network on gridworlds produced by produce_grids and on trajectories
    produced by produce_opt_trajs_mult_grids
    >>> Currently it only trains the first gridworld and ignores "num_training_grids"
    """
    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]
    init_vars["val_period"] = 20

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        mdp_objs_test = pickle.load(f)

    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])

    # save the gridworld to be used for eval later
    with open(logs_address + "MDP_train.pkl", "wb") as f:
        pickle.dump(mdp_objs_tr[0], f)

    for run in range(init_vars['num_runs']):

        envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))
        MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
        agent = produce_agent(envs, init_vars, agent_mode="train")
        agent.logs_address = logs_address



        if init_vars["run_mode"]  == "continue":
            # Load models
            if agent.reward_net_name == "MLP":
                agent.reward_net = agent_rl2.net_MLP(envs[0], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
            if agent.reward_net_name == "CNN":
                pass
            agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))


        with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
            all_trajs = pickle.load(f)
        agent.demo_visitation_calculator(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts

        if init_vars["run_mode"] == "restart":
            with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("L_D_theta, L_phi, validation_L_phi, min_y, grad_L_D_theta_norm_one, grad_L_phi_theta_norm_one, reg_term_norm_one, episode_number, episode_time \n")
            with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("output progress")
            min_range = 0
            max_range = init_vars["num_theta_updates"]

        elif init_vars["run_mode"] == "continue":
            loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
            min_range = int(loss[-1,3])
            max_range = min_range + init_vars["num_theta_updates"]
            loss = 0

        for epis in range(min_range, max_range):
            start = time.time()
            Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
            agent.calc_pi_theta() # Policy is defined using softmax from pi_theta
            agent.calc_grad_Q_theta(thresh=init_vars["thresh_Q_theta"])
            grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value
            agent.calc_grad_pi_theta()

            # following 3 only necessary if we want to modify the cost function
            # using specifications
            # ---------------------
            agent.calc_y_theta(thresh=init_vars["thresh_Y_theta"])
            agent.calc_y_theta_eval(thresh=init_vars["thresh_Y_theta"])

            agent.calc_grad_y_theta(thresh=init_vars["thresh_grad_Y_theta"])
            grad_L_phi_theta = agent.calc_grad_L_phi_theta()

            # ---------------------
            grad_j = grad_L_D_theta - lam_reg * reg_term + lam_phi * grad_L_phi_theta
            agent.update_theta(grad_j)
            L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])
            L_phi, min_y =  agent.calc_L_phi_theta()
            L_phi_eval, min_y_eval = agent.calc_L_phi_theta_eval()
            end = time.time()
            epis_time = end-start

            grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
            grad_L_phi_theta_norm_one = np.mean(np.abs(grad_L_phi_theta))
            reg_term_norm_one = np.mean(np.abs(reg_term))

            if epis % init_vars["val_period"] == 0:
                if init_vars["verbose"] == True:
                    print ("SAVING THE MODELS .............")
                    print ()
                torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + "run_" + str(run) + ".pt")
                validation_L_phi, validation_L_phi_eval = val(init_vars, mdp_objs_test, models_address, run)

                if init_vars["verbose"] == True:
                    print ("****************************************")
                    print (f"episode number:  {epis}")
                    print (f"L_D_theta: {L_D_theta:.4}")
                    print (f"L_phi: {L_phi:.4}")
                    print (f"L_phi_eval: {L_phi_eval:.4}")
                    print (f"validation L_phi: {validation_L_phi:.4}")
                    print (f"validation L_phi eval: {validation_L_phi_eval:.4}")
                    print (f"epis_time: {epis_time:.4}")
                    print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
                    print (f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}")
                    print (f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}")
                # print ("reward_net_time: {:.4}".format(reward_net_time))
                # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
                # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
                # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
                # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
                # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
                # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
                # print ("update_theta_TIME: {:.4}".format(s9-s8))


                with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "a") as file2:
                    file2.write("****************************************\n")
                    file2.write(f"episode number:  {epis}\n")
                    file2.write(f"L_D_theta: {L_D_theta:.4}\n")
                    file2.write(f"L_phi: {L_phi:.4}\n")
                    file2.write(f"L_phi_eval: {L_phi_eval:.4}\n")
                    file2.write(f"validation_L_phi: {validation_L_phi:.4}\n")
                    file2.write(f"validation_L_phi_eval: {validation_L_phi_eval:.4}\n")
                    file2.write(f"epis_time: {epis_time:.4}\n")
                    file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
                    file2.write(f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}\n")
                    file2.write(f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}\n")

                with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "a") as file1:
                    file1.write(f'{L_D_theta:>20} {L_phi:>20} {L_phi_eval:>20} {validation_L_phi:>20} {validation_L_phi_eval:>20} {min_y:>15} {grad_L_D_theta_norm_one:.10} {grad_L_phi_theta_norm_one:.10} {reg_term_norm_one:.10} {epis} {epis_time:.4} \n')


            # if epis % init_vars["save_model_interval"] == 0:
            #     torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + str(epis) + ".pt")

def train_IRL_active(init_vars):
    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)

    _, DFA_obj_GT, _ = produce_envs_only_eval_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))

    bashCommand = "./active-DFA/src/examples/online"
    proc = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    success_thresh = 0.7
    success_ratio = 0.0

    last_read_query = ""
    DFA_counter = 0
    DFA_inference_call_counter = 0
    while success_ratio < success_thresh:
        DFA_counter += 1

        if last_read_query == "equivalent":
            pass
        else:
            query = proc.stdout.readline()

        if query[-1:] == "\n":
            query = query[:-1]
        assert query == "equivalent"

        if DFA_counter == 1:
            ans = "n\n"
        else:
            # produce trajectories for the new DFA
            if init_vars["verbose"]:
                print("producing new trajectories for the new DFA")
            command  = f'python src/main_irl2.py --main_function="produce_opt_trajs_time_limited" --fileName_grids={init_vars["fileName_grids"]} --fileName_tr_test={init_vars["fileName_tr_test"]} '
            produce_trajs = subprocess.call(args = command, shell=True)
            if init_vars["verbose"]:
                print("---- TRAIN IRL --- \n")

            # before calling _train_IRL_active, run_active_DFA_inference needs to be called
            assert DFA_inference_call_counter > 0
            L_phi, success_ratio = _train_IRL_active(init_vars)
            if init_vars["verbose"]:
                print(f"---- success_ratio: {success_ratio}")
            if success_ratio < success_thresh:
                ans = "n\n"
            else:
                ans = "y\n"
        print(query, ans)
        proc.stdin.write(ans)
        proc.stdin.flush()

        assert ans in ["y\n", "n\n"] # this ans corresponds to the equivalent query

        if ans == "n\n":
            # since answer is no,
            ans = provide_counter_ex(init_vars, DFA_counter, copy.deepcopy(mdp_objs_tr[0]))
            if init_vars["verbose"]:
                print(f"Counter Example: {ans}")

            proc.stdin.write(ans)
            proc.stdin.flush()

        # at this point, ans whould either be "y\n" or a counter example

        if ans != "y\n":
            # if the correct DFA is inferred, we don't need to call DFA inference again
            last_read_query = run_active_DFA_inference(init_vars, DFA_obj_GT, proc) # this function writes a new DFA into the corresponding directory
            DFA_inference_call_counter += 1
            command  = f'cp automaton.txt ./inferred_DFAs/automaton_{init_vars["fileName_tr_test"]}.txt '
            subprocess.call(args = command, shell=True)
            command  = f'cp automaton.txt ./inferred_DFAs/automaton_{init_vars["fileName_tr_test"]}_{DFA_inference_call_counter}.txt '
            subprocess.call(args = command, shell=True)
            command  = f'cp outfile.png ./inferred_DFAs/outfile_{init_vars["fileName_tr_test"]}_{DFA_inference_call_counter}.png '
            subprocess.call(args = command, shell=True)

def _train_IRL_active(init_vars):
    """
    This function trains the reward network on gridworlds produced by produce_grids and on trajectories
    produced by produce_opt_trajs_mult_grids
    >>> Currently it only trains the first gridworld and ignores "num_training_grids"
    """

    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        mdp_objs_test = pickle.load(f)

    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    MDP_obj, DFA_obj_GT, PA_obj_GT = produce_envs_only_eval_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))
    _, DFA_obj, PA_obj = produce_envs_only_tr_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))
    envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
    agent = produce_agent(envs, init_vars, agent_mode="train")
    agent.logs_address = logs_address


    # save the gridworld to be used for eval later
    with open(logs_address + "MDP_train.pkl", "wb") as f:
        pickle.dump(MDP_obj, f)
    run = 0 # cause we do a single run in this case
    if init_vars["run_mode"]  == "continue":
        # Load models
        if agent.reward_net_name == "MLP":
            agent.reward_net = agent_env.net_MLP(envs[0], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
        if agent.reward_net_name == "CNN":
            pass
        agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))

    with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
        all_trajs = pickle.load(f)
    agent.demo_visitation_calculator(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts

    if init_vars["run_mode"] == "restart":
        with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "w") as file:
            file.write("success_ratio, L_D_theta, L_phi, validation_L_phi, min_y, grad_L_D_theta_norm_one, grad_L_phi_theta_norm_one, reg_term_norm_one, episode_number, episode_time \n")
        with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "w") as file:
            file.write("output progress")
        min_range = 0
        max_range = init_vars["num_theta_updates"]

    elif init_vars["run_mode"] == "continue":
        loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
        min_range = int(loss[-1,3])
        max_range = min_range + init_vars["num_theta_updates"]
        loss = 0

    for epis in range(min_range, max_range):
        start = time.time()
        Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
        agent.calc_pi_theta() # Policy is defined using softmax from pi_theta
        agent.calc_grad_Q_theta(thresh=init_vars["thresh_Q_theta"])
        grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value
        agent.calc_grad_pi_theta()

        # following 3 only necessary if we want to modify the cost function
        # using specifications
        # ---------------------
        agent.calc_y_theta(thresh=init_vars["thresh_Y_theta"])
        # agent.calc_y_theta_eval(thresh=init_vars["thresh_Y_theta"])

        agent.calc_grad_y_theta(thresh=init_vars["thresh_grad_Y_theta"])
        grad_L_phi_theta = agent.calc_grad_L_phi_theta()

        # ---------------------
        grad_j = grad_L_D_theta - lam_reg * reg_term + lam_phi * grad_L_phi_theta
        agent.update_theta(grad_j)
        L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])
        L_phi, min_y =  agent.calc_L_phi_theta()
        # L_phi_eval, min_y_eval = agent.calc_L_phi_theta_eval()

        end = time.time()
        epis_time = end-start

        grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
        grad_L_phi_theta_norm_one = np.mean(np.abs(grad_L_phi_theta))
        reg_term_norm_one = np.mean(np.abs(reg_term))

        if epis % init_vars["val_period"] == 0:
            if init_vars["verbose"] == True:
                print ("SAVING THE MODELS .............")
                print ()
            torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + "run_" + str(run) + ".pt")
            # validation_L_phi, validation_L_phi_eval = val(init_vars, mdp_objs_test, models_address, run)

            if init_vars["verbose"] == True:
                print ("**** eval game performance")
            num_eval_episodes = 100
            success_ratio = eval_game_performance(init_vars, agent, num_eval_episodes)

            if init_vars["verbose"] == True:
                print ("****************************************")
                print (f"episode number:  {epis}")
                print (f"L_D_theta: {L_D_theta:.4}")
                print (f"L_phi: {L_phi:.4}")
                print (f"success_ratio: {success_ratio:.4}")
                # print (f"validation L_phi: {validation_L_phi:.4}")
                # print (f"validation L_phi eval: {validation_L_phi_eval:.4}")
                print (f"epis_time: {epis_time:.4}")
                print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
                print (f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}")
                print (f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}")
            # print ("reward_net_time: {:.4}".format(reward_net_time))
            # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
            # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
            # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
            # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
            # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
            # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
            # print ("update_theta_TIME: {:.4}".format(s9-s8))

            with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "a") as file2:
                file2.write("****************************************\n")
                file2.write(f"episode number:  {epis}\n")
                file2.write(f"L_D_theta: {L_D_theta:.4}\n")
                file2.write(f"L_phi: {L_phi:.4}\n")
                file2.write(f"success_ratio: {success_ratio:.4}\n")
                # file2.write(f"validation_L_phi: {validation_L_phi:.4}\n")
                # file2.write(f"validation_L_phi_eval: {validation_L_phi_eval:.4}\n")
                file2.write(f"epis_time: {epis_time:.4}\n")
                file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
                file2.write(f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}\n")
                file2.write(f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}\n")

            with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "a") as file1:
                file1.write(f'{success_ratio:>.5} {L_D_theta:>20} {L_phi:>20} {min_y:>15} {grad_L_D_theta_norm_one:.10} {grad_L_phi_theta_norm_one:.10} {reg_term_norm_one:.10} {epis} {epis_time:.4} \n')

    return L_phi, success_ratio
        # if epis % init_vars["save_model_interval"] == 0:
        #     torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + str(epis) + ".pt")

def train_from_inferred_DFA(init_vars):
    """
    This function trains the reward network on gridworlds produced by produce_grids and on trajectories
    produced by produce_opt_trajs_mult_grids
    >>> Currently it only trains the first gridworld and ignores "num_training_grids"
    """
    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]
    print(lam_phi)

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        mdp_objs_test = pickle.load(f)

    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    MDP_obj, DFA_obj_GT, PA_obj_GT = produce_envs_only_eval_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))
    _, DFA_obj, PA_obj = produce_envs_only_tr_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))
    envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
    agent = produce_agent(envs, init_vars, agent_mode="train")
    agent.logs_address = logs_address


    # # save the gridworld to be used for eval later
    with open(logs_address + "MDP_train.pkl", "wb") as f:
        pickle.dump(MDP_obj, f)
    for run in range(init_vars['num_runs']):
        print(agent.MDP.grid[:,33,45])
        if init_vars["run_mode"]  == "continue":
            # Load models
            if agent.reward_net_name == "MLP":
                agent.reward_net = agent_env.net_MLP(envs[0], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
            if agent.reward_net_name == "CNN":
                pass
            agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))


        # produce trajectories for the new DFA
        if init_vars["verbose"]:
            print("producing new trajectories for the new DFA")
        command  = f'python3 src/main_irl2.py --main_function="produce_opt_trajs_time_limited" --fileName_grids={init_vars["fileName_grids"]} --fileName_tr_test={init_vars["fileName_tr_test"]} '
        produce_trajs = subprocess.call(args = command, shell=True)


        with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
            all_trajs = pickle.load(f)
        # all_trajs=[]
        # for i in range(1):
        #     all_trajs.append([(0,[0,4],0,0),(0,[1,4],0,0),(0,[2,4],0,0),(0,[3,4],0,0),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[0,2],0,0),(0,[0,3],0,0),(0,[0,4],0,0),(0,[1,4],0,0),(0,[2,4],0,0),(0,[3,4],0,0),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[0,6],0,0),(0,[0,5],0,0),(0,[0,4],0,0),(0,[1,4],0,0),(0,[2,4],0,0),(0,[3,4],0,0),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[1,2],0,0),(0,[1,3],0,0),(0,[1,4],0,0),(0,[2,4],0,0),(0,[3,4],0,0),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[1,6],0,0),(0,[1,5],0,0),(0,[1,4],0,0),(0,[2,4],0,0),(0,[3,4],0,0),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[2,3],0,0),(0,[2,4],0,0),(0,[3,4],0,0),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[4,0],0,3),(0,[4,1],0,3),(0,[4,2],0,3),(0,[4,3],0,3),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[2,0],0,3),(0,[3,0],0,3),(0,[4,0],0,3),(0,[4,1],0,3),(0,[4,2],0,3),(0,[4,3],0,3),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[6,0],0,3),(0,[5,0],0,3),(0,[4,0],0,3),(0,[4,1],0,3),(0,[4,2],0,3),(0,[4,3],0,3),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[2,1],0,3),(0,[3,1],0,3),(0,[4,1],0,3),(0,[4,2],0,3),(0,[4,3],0,3),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])
        #     all_trajs.append([(0,[6,1],0,3),(0,[5,1],0,3),(0,[4,1],0,3),(0,[4,2],0,3),(0,[4,3],0,3),(0,[4,4],0,3),(0,[4,5],0,3),"game won"])

        if init_vars["traj_address"]:
            all_trajs = get_trajectories(init_vars)
            agent.demo_visitation_calculator(init_vars["num_trajs_used"], all_trajs) # fills in agent.visitation_counts
        else:
            agent.demo_visitation_calculator(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts

        if init_vars["run_mode"] == "restart":
            with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("success_ratio, L_D_theta, L_phi, validation_L_phi, min_y, grad_L_D_theta_norm_one, grad_L_phi_theta_norm_one, reg_term_norm_one, episode_number, episode_time \n")
            with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("output progress")
            min_range = 0
            max_range = init_vars["num_theta_updates"]

        elif init_vars["run_mode"] == "continue":
            loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
            min_range = int(loss[-1,3])
            max_range = min_range + init_vars["num_theta_updates"]
            loss = 0

        L_best = -10000

        for epis in range(min_range, max_range):
            start = time.time()
            Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
            agent.calc_pi_theta() # Policy is defined using softmax from pi_theta
            agent.calc_grad_Q_theta(thresh=init_vars["thresh_Q_theta"])
            grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value
            agent.calc_grad_pi_theta()

            # following 3 only necessary if we want to modify the cost function
            # using specifications
            # ---------------------
            agent.calc_y_theta(thresh=init_vars["thresh_Y_theta"])
            # agent.calc_y_theta_eval(thresh=init_vars["thresh_Y_theta"])

            agent.calc_grad_y_theta(thresh=init_vars["thresh_grad_Y_theta"])
            grad_L_phi_theta = agent.calc_grad_L_phi_theta()

            # ---------------------
            grad_j = grad_L_D_theta - lam_reg * reg_term + lam_phi * grad_L_phi_theta
            agent.update_theta(grad_j)
            L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])
            L_phi, min_y =  agent.calc_L_phi_theta()

            L_phi_eval, min_y_eval = agent.calc_L_phi_theta_eval()

            end = time.time()
            epis_time = end-start

            grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
            grad_L_phi_theta_norm_one = np.mean(np.abs(grad_L_phi_theta))
            reg_term_norm_one = np.mean(np.abs(reg_term))

            if epis % init_vars["val_period"] == 0:
                if init_vars["verbose"] == True:
                    print ("SAVING THE MODELS .............")
                    print ()
                if lam_phi*L_phi + L_D_theta > L_best:
                    print('YES')
                    L_best = lam_phi*L_phi + L_D_theta
                    torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + "run_" + str(run) + ".pt")
                # torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + "run_" + str(run) + ".pt")
                # validation_L_phi, validation_L_phi_eval = val(init_vars, mdp_objs_test, models_address, run)

                if init_vars["verbose"] == True:
                    print ("**** eval game performance")
                num_eval_episodes = 100
                success_ratio, loss_ratio = eval_game_performance(init_vars, agent, num_eval_episodes)

                if init_vars["verbose"] == True:
                    print ("****************************************")
                    print (f"episode number:  {epis}")
                    print (f"L_D_theta: {L_D_theta:.4}")
                    print (f"L_phi: {L_phi:.4}")
                    print (f"L_phi_eval: {L_phi_eval:.4}")

                    print (f"success_ratio: {success_ratio:.4}")
                    # print (f"validation L_phi: {validation_L_phi:.4}")
                    # print (f"validation L_phi eval: {validation_L_phi_eval:.4}")
                    print (f"epis_time: {epis_time:.4}")
                    print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
                    print (f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}")
                    print (f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}")
                # print ("reward_net_time: {:.4}".format(reward_net_time))
                # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
                # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
                # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
                # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
                # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
                # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
                # print ("update_theta_TIME: {:.4}".format(s9-s8))

                with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "a") as file2:
                    file2.write("****************************************\n")
                    file2.write(f"episode number:  {epis}\n")
                    file2.write(f"L_D_theta: {L_D_theta:.4}\n")
                    file2.write(f"L_phi: {L_phi:.4}\n")
                    file2.write(f"min_y: {min_y:.4}\n")
                    file2.write(f"L_phi_eval: {L_phi_eval:.4}\n")

                    file2.write(f"success_ratio: {success_ratio:.4}\n")
                    # file2.write(f"validation_L_phi: {validation_L_phi:.4}\n")
                    # file2.write(f"validation_L_phi_eval: {validation_L_phi_eval:.4}\n")
                    file2.write(f"epis_time: {epis_time:.4}\n")
                    file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
                    file2.write(f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}\n")
                    file2.write(f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}\n")

                with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "a") as file1:
                    file1.write(f'{success_ratio:>.5} {L_D_theta:>20} {L_phi:>20} {min_y:>15} {L_phi_eval:>20} {grad_L_D_theta_norm_one:.10} {grad_L_phi_theta_norm_one:.10} {reg_term_norm_one:.10} {epis} {epis_time:.4} \n')
            print(epis)
            print(agent.y_theta[:,:,0])
        reward_from_NN = agent.eval_reward()
        print(reward_from_NN[:,:,0,0])
        # agent.y_theta[2,2:6,0] = 0
        # agent.y_theta[2,7:9,0] = 0
        # agent.y_theta[3,:11,0] = 0
        # agent.y_theta[5,9:,0] = 0
        # agent.y_theta[6,9:11,0] = 0
        # agent.y_theta[7,8:,0] = 0
        # agent.y_theta[8,0,0] = 0
        # agent.y_theta[8,2:4,0] = 0
        # agent.y_theta[8,6:8,0] = 0
        # agent.y_theta[8,9,0] = 0
        # agent.y_theta[9,9,0] = 0
        # agent.y_theta[10,0,0] = 0
        # agent.y_theta[10,2,0] = 0
        # agent.y_theta[11,0,0] = 0
        # agent.y_theta[9,3:,0] = 1
        # agent.y_theta[10,3:,0] = 1
        # agent.y_theta[11,3:,0] = 1
        fig, ax = plt.subplots()
        im = ax.imshow(agent.y_theta[:,:,0])
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('', rotation=-90, va="bottom")
        plt.show()
    return
            # if epis % init_vars["save_model_interval"] == 0:
            #     torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + str(epis) + ".pt")

def train_non_local_r(init_vars):
    """
    This function trains the reward network on gridworlds produced by produce_grids and on trajectories
    produced by produce_opt_trajs_mult_grids
    >>> Currently it only trains the first gridworld and ignores "num_training_grids"
    """
    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]
    init_vars["val_period"] = 20

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)

    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])


    for run in range(init_vars['num_runs']):
        # random.seed(run)
        envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0][0]))
        agent = produce_agent_RL_non_local_r(envs, init_vars, agent_mode="train")
        agent.logs_address = logs_address
        MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]

        # # save the gridworld to be used for eval later
        # with open(logs_address + "MDP_train.pkl", "wb") as f:
        #     pickle.dump(MDP_obj, f)

        if init_vars["run_mode"]  == "continue":
            # Load models
            pass
            # if agent.reward_net_name == "MLP":
            #     agent.reward_net = agent_env.net_MLP(envs[0], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
            # if agent.reward_net_name == "CNN":
            #     pass
            # agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))


        with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
            all_trajs = pickle.load(f)
        agent.demo_visitation_calculator(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts

        if init_vars["run_mode"] == "restart":
            with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("L_D_theta, L_phi, validation_L_phi, min_y, grad_L_D_theta_norm_one, grad_L_phi_theta_norm_one, reg_term_norm_one, episode_number, episode_time \n")
            with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("output progress")
            min_range = 0
            max_range = init_vars["num_theta_updates"]

        elif init_vars["run_mode"] == "continue":
            loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
            min_range = int(loss[-1,3])
            max_range = min_range + init_vars["num_theta_updates"]
            loss = 0

        for epis in range(min_range, max_range):
            start = time.time()
            Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
            agent.calc_pi_theta() # Policy is defined using softmax from pi_theta
            agent.calc_grad_Q_theta(thresh=init_vars["thresh_Q_theta"])
            grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value
            agent.calc_grad_pi_theta()
            # following 4 lines only necessary if we want to modify the cost function
            # using specifications
            # ---------------------
            agent.calc_y_theta(thresh=init_vars["thresh_Y_theta"])
            agent.calc_grad_y_theta(thresh=init_vars["thresh_grad_Y_theta"])
            grad_L_phi_theta = agent.calc_grad_L_phi_theta()
            # ---------------------
            grad_j = grad_L_D_theta - lam_reg * reg_term + lam_phi * grad_L_phi_theta
            agent.update_theta(grad_j)
            L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])
            L_phi, min_y =  agent.calc_L_phi_theta()
            end = time.time()
            epis_time = end-start

            grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
            grad_L_phi_theta_norm_one = np.mean(np.abs(grad_L_phi_theta))
            reg_term_norm_one = np.mean(np.abs(reg_term))


            if epis % init_vars["val_period"] == 0:
                if init_vars["verbose"] == True:
                    print ("SAVING THE MODELS .............")
                    print ()
                torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + "run_" + str(run) + ".pt")
                validation_L_phi = val_non_local_r(init_vars, envs_objs_test, models_address, run)

                if init_vars["verbose"] == True:
                    print ("****************************************")
                    print (f"episode number:  {epis}")
                    print (f"L_D_theta: {L_D_theta:.4}")
                    print (f"L_phi: {L_phi:.4}")
                    print (f"validation L_phi: {validation_L_phi:.4}")
                    print (f"epis_time: {epis_time:.4}")
                    print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
                    print (f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}")
                    print (f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}")
                # print ("reward_net_time: {:.4}".format(reward_net_time))
                # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
                # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
                # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
                # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
                # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
                # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
                # print ("update_theta_TIME: {:.4}".format(s9-s8))


                with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "a") as file2:
                    file2.write("****************************************\n")
                    file2.write(f"episode number:  {epis}\n")
                    file2.write(f"L_D_theta: {L_D_theta:.4}\n")
                    file2.write(f"L_phi: {L_phi:.4}\n")
                    file2.write(f"validation_L_phi: {validation_L_phi:.4}\n")
                    file2.write(f"epis_time: {epis_time:.4}\n")
                    file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
                    file2.write(f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}\n")
                    file2.write(f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}\n")

                with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "a") as file1:
                    file1.write(f'{L_D_theta:>20} {L_phi:>20} {validation_L_phi:>20} {min_y:>15} {grad_L_D_theta_norm_one:.10} {grad_L_phi_theta_norm_one:.10} {reg_term_norm_one:.10} {epis} {epis_time:.4} \n')


            # if epis % init_vars["save_model_interval"] == 0:
            #     torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + str(epis) + ".pt")

# ----- baselines ------

def train_BC(init_vars):
    """
    Training behavioral cloning agent
    This function trains the reward network on gridworlds produced by produce_grids and on trajectories
    produced by produce_opt_trajs_mult_grids
    >>> Currently it only trains the first gridworld and ignores "num_training_grids"
    """

    init_vars["val_period"] = 20
    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)
    envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0][0]))
    agent = produce_agent_BC(envs, init_vars, agent_mode = "train")

    MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    agent.logs_address = logs_address

    if init_vars["run_mode"]  == "continue":
        # Load models
        pass
        # if agent.reward_net_name == "CNN":
        #     agent.reward_net = agent_env.net_CNN(envs[0], n_dim=envs[0].n_dim, out_dim=1).to(device)
        # if agent.reward_net_name == "att":
        #     agent.reward_net = agent_env.net_att(envs[0], n_dim=envs[0].n_dim, out_dim=1).to(device)
        # agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))

    # save the gridworld to be used for eval later
    with open(logs_address + "MDP_train.pkl", "wb") as f:
        pickle.dump(MDP_obj, f)

    with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
        all_trajs = pickle.load(f)


    if init_vars["run_mode"] == "restart":
        with open(logs_address + "loss_per_episode.txt", "w") as file:
            file.write("L_D_theta, L_phi, grad_L_D_theta_norm_one, episode_number, episode_time \n")
        with open(logs_address + "output_progress.txt", "w") as file:
            file.write("output progress")
        min_range = 0
        max_range = init_vars["num_theta_updates"]
    elif init_vars["run_mode"] == "continue":
        loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
        min_range = int(loss[-1,3])
        max_range = min_range + init_vars["num_theta_updates"]
        loss = 0

    agent.demo_training_set(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts
    agent.calc_all_mdp_dfa_inputs()
    for epis in range(min_range, max_range):
        start = time.time()
        loss = agent.update_theta()
        agent.calc_pi_theta() # this fills in the policy table from the policy network
        # grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value
        # agent.calc_grad_pi_theta()

        # following 4 lines only necessary if we want to modify the cost function
        # using specifications
        # ---------------------
        agent.calc_y_theta(thresh=init_vars["thresh_Y_theta"])
        # agent.calc_grad_y_theta(thresh=init_vars["thresh_grad_Y_theta"])
        # L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])
        L_phi, min_y =  agent.calc_L_phi_theta()
        end = time.time()
        epis_time = end-start

        # grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
        # grad_L_phi_theta_norm_one = np.mean(np.abs(grad_L_phi_theta))


        if epis % init_vars["val_period"] == 0:
            if init_vars["verbose"] == True:
                print ("SAVING THE MODELS .............")
                print ()
            torch.save(agent.policy_net.state_dict(), models_address + "policy_net.pt")
            validation_L_phi = val_BC(init_vars, envs_objs_test, models_address)

            if init_vars["verbose"] == True:
                print ("****************************************")
                print (f"loss: {loss}")
                print (f"episode number:  {epis}")
                # print (f"L_D_theta: {L_D_theta:.4}")
                print (f"L_phi: {L_phi:.4}")
                print (f"validation_L_phi: {validation_L_phi:.4}")
            # print (f"epis_time: {epis_time:.4}")
            # print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
            # print (f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}")


            # print ("reward_net_time: {:.4}".format(reward_net_time))
            # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
            # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
            # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
            # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
            # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
            # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
            # print ("update_theta_TIME: {:.4}".format(s9-s8))

            with open(logs_address + "output_progress.txt", "a") as file2:
                file2.write("****************************************\n")
                file2.write(f"episode number:  {epis}\n")
                file2.write(f"L_D_theta: {loss:.4}\n")
                file2.write(f"L_phi: {L_phi:.4}\n")
                file2.write(f"validation_L_phi: {validation_L_phi:.4}\n")
                # file2.write(f"epis_time: {epis_time:.4}\n")
                # file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
                # file2.write(f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}\n")

            with open(logs_address + "loss_per_episode.txt", "a") as file1:
                file1.write(f'{loss:>20} {L_phi:>20} {validation_L_phi:>20} {epis} \n')


        if epis % init_vars["save_model_interval"] == 0:
            torch.save(agent.policy_net.state_dict(), models_address + "policy_net_" + str(epis) + ".pt")

def train_BC_3(init_vars):
    """
    Training behavioral cloning agent
    This function trains the reward network on gridworlds produced by produce_grids and on trajectories
    produced by produce_opt_trajs_mult_grids
    >>> Currently it only trains the first gridworld and ignores "num_training_grids"
    """


    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0][0]))
    agent = produce_agent_BC_3(envs, init_vars, agent_mode = "train")
    MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    agent.logs_address = logs_address

    if init_vars["run_mode"]  == "continue":
        # Load models
        pass
        # if agent.reward_net_name == "CNN":
        #     agent.reward_net = agent_env.net_CNN(envs[0], n_dim=envs[0].n_dim, out_dim=1).to(device)
        # if agent.reward_net_name == "att":
        #     agent.reward_net = agent_env.net_att(envs[0], n_dim=envs[0].n_dim, out_dim=1).to(device)
        # agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))

    # save the gridworld to be used for eval later
    with open(logs_address + "MDP_train.pkl", "wb") as f:
        pickle.dump(MDP_obj, f)

    with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
        all_trajs = pickle.load(f)

    agent.demo_training_set(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts
    agent.calc_all_mdp_inputs()
    if init_vars["run_mode"] == "restart":
        with open(logs_address + "loss_per_episode.txt", "w") as file:
            file.write("L_D_theta, L_phi, grad_L_D_theta_norm_one, episode_number, episode_time \n")
        with open(logs_address + "output_progress.txt", "w") as file:
            file.write("output progress")
        min_range = 0
        max_range = init_vars["num_theta_updates"]
    elif init_vars["run_mode"] == "continue":
        loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
        min_range = int(loss[-1,3])
        max_range = min_range + init_vars["num_theta_updates"]
        loss = 0

    for epis in range(min_range, max_range):
        start = time.time()
        loss = agent.update_theta()
        agent.calc_pi_theta() # this fills in the policy table from the policy network
        # grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value
        # agent.calc_grad_pi_theta()

        # following 4 lines only necessary if we want to modify the cost function
        # using specifications
        # ---------------------
        agent.calc_y_theta(thresh=init_vars["thresh_Y_theta"])
        # agent.calc_grad_y_theta(thresh=init_vars["thresh_grad_Y_theta"])
        # L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])
        L_phi, min_y =  agent.calc_L_phi_theta()
        end = time.time()
        epis_time = end-start

        # grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
        # grad_L_phi_theta_norm_one = np.mean(np.abs(grad_L_phi_theta))

        if init_vars["verbose"] == True:
            print ("****************************************")
            print (f"loss: {loss}")
            print (f"episode number:  {epis}")
            # print (f"L_D_theta: {L_D_theta:.4}")
            print (f"L_phi: {L_phi:.4}")
            # print (f"epis_time: {epis_time:.4}")
            # print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
            # print (f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}")


            # print ("reward_net_time: {:.4}".format(reward_net_time))
            # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
            # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
            # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
            # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
            # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
            # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
            # print ("update_theta_TIME: {:.4}".format(s9-s8))

        with open(logs_address + "output_progress.txt", "a") as file2:
            file2.write("****************************************\n")
            file2.write(f"episode number:  {epis}\n")
            file2.write(f"L_D_theta: {loss:.4}\n")
            file2.write(f"L_phi: {L_phi:.4}\n")
            # file2.write(f"epis_time: {epis_time:.4}\n")
            # file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
            # file2.write(f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}\n")

        with open(logs_address + "loss_per_episode.txt", "a") as file1:
            file1.write(f'{loss:>20} {L_phi:>20} {epis} \n')

        if epis != 0 and epis % 20 == 0:
            if init_vars["verbose"] == True:
                print ("SAVING THE MODELS .............")
                print ()
            torch.save(agent.policy_net0.state_dict(), models_address + "policy_net0.pt")
            torch.save(agent.policy_net1.state_dict(), models_address + "policy_net1.pt")
            torch.save(agent.policy_net2.state_dict(), models_address + "policy_net2.pt")

        if epis % init_vars["save_model_interval"] == 0:
            torch.save(agent.policy_net0.state_dict(), models_address + "policy_net0_" + str(epis) + ".pt")
            torch.save(agent.policy_net1.state_dict(), models_address + "policy_net1_" + str(epis) + ".pt")
            torch.save(agent.policy_net2.state_dict(), models_address + "policy_net2_" + str(epis) + ".pt")

def train_base(init_vars):
    # training memoryless agent, i.e. the agent is not equipped with an automaton
    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)

    envs = produce_envs_from_MDP_baseline(init_vars, mdp_objs_tr[0][0], 'DFA_base')
    agent = produce_agent_baseline(envs, init_vars, agent_mode = "train")
    MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]

    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])

    if init_vars["run_mode"]  == "continue":
        # Load models
        if agent.reward_net_name == "MLP":
            agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
        if agent.reward_net_name == "CNN":
            pass
        agent.reward_net.load_state_dict(torch.load(models_address + "reward_net_baseline.pt"))

    # save the gridworld to be used for eval later
    with open(logs_address + "MDP_train.pkl", "wb") as f:
        pickle.dump(MDP_obj, f)

    with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
        all_trajs = pickle.load(f)

    agent.demo_visitation_calculator(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts

    if init_vars["run_mode"] == "restart":
        with open(logs_address + "loss_per_episode.txt", "w") as file:
            file.write("L_D_theta, L_phi, grad_L_D_theta_norm_one, episode_number, episode_time \n")
        with open(logs_address + "output_progress.txt", "w") as file:
            file.write("output progress")
        min_range = 0
        max_range = init_vars["num_theta_updates"]
    elif init_vars["run_mode"] == "continue":
        pass
        # loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
        # min_range = int(loss[-1,3])
        # max_range = min_range + init_vars["num_theta_updates"]
        # loss = 0

    for epis in range(min_range, max_range):
        start = time.time()
        Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
        agent.calc_pi_theta() # Policy is defined using softmax from pi_theta
        agent.calc_grad_Q_theta(thresh=init_vars["thresh_Q_theta"])
        grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value

        # following 4 lines only necessary if we want to modify the cost function
        # using specifications
        # ---------------------
        # agent.calc_y_theta(thresh=init_vars["thresh_Y_theta"])
        # agent.calc_grad_y_theta(thresh=init_vars["thresh_grad_Y_theta"])
        # grad_L_phi_theta = agent.calc_grad_L_phi_theta()
        # ---------------------
        grad_j = grad_L_D_theta - lam_reg * reg_term
        agent.update_theta(grad_j)
        L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])
        L_phi_theta, min_y =  agent.calc_L_phi_theta()

        end = time.time()
        epis_time = end-start

        grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
        reg_term_norm_one = np.mean(np.abs(reg_term))

        if init_vars["verbose"] == True:
            print ("****************************************")
            print (f"episode number:  {epis}")
            print (f"L_D_theta: {L_D_theta:.4}")
            print (f"L_phi_theta: {L_phi_theta:.4}")

            print (f"epis_time: {epis_time:.4}")
            print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
            # print ("reward_net_time: {:.4}".format(reward_net_time))
            # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
            # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
            # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
            # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
            # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
            # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
            # print ("update_theta_TIME: {:.4}".format(s9-s8))

        with open(logs_address + "output_progress.txt", "a") as file2:
            file2.write("****************************************\n")
            file2.write(f"episode number:  {epis}\n")
            file2.write(f"L_D_theta: {L_D_theta:.4}\n")
            file2.write(f"L_phi_theta: {L_phi_theta:.4}\n")
            file2.write(f"epis_time: {epis_time:.4}\n")
            file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
            file2.write(f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}\n")

        with open(logs_address + "loss_per_episode.txt", "a") as file1:
            file1.write(f'{L_D_theta:>20} {L_phi_theta:>20} {grad_L_D_theta_norm_one:.10}  {reg_term_norm_one:.10} {epis} {epis_time:.4} \n')

        if epis != 0 and epis % 20 == 0:
            if init_vars["verbose"] == True:
                print ("SAVING THE MODELS .............")
                print ()
            torch.save(agent.reward_net.state_dict(), models_address + "reward_net.pt")

        if epis % init_vars["save_model_interval"] == 0:
            torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + str(epis) + ".pt")

def train_base_memoryless(init_vars):
    """
    This is for IJCAI submission
    """
    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        mdp_objs_test = pickle.load(f)

    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    MDP_obj, DFA_obj_GT, PA_obj_GT = produce_envs_only_eval_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))
    _, DFA_obj, PA_obj = produce_envs_only_tr_from_MDP_memoryless(init_vars, copy.deepcopy(mdp_objs_tr[0]))
    envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
    agent = produce_agent(envs, init_vars, agent_mode="train")
    agent.logs_address = logs_address


    # # save the gridworld to be used for eval later
    with open(logs_address + "MDP_train.pkl", "wb") as f:
        pickle.dump(MDP_obj, f)
    for run in range(init_vars['num_runs']):
        if init_vars["run_mode"]  == "continue":
            # Load models
            if agent.reward_net_name == "MLP":
                agent.reward_net = agent_env.net_MLP(envs[0], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
            if agent.reward_net_name == "CNN":
                pass
            agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))


        # produce trajectories for the new DFA
        if init_vars["verbose"]:
            print("producing new trajectories for the new DFA")
        command  = f'python src/main_irl2.py --main_function="produce_opt_trajs_time_limited" --fileName_grids={init_vars["fileName_grids"]} --fileName_tr_test={init_vars["fileName_tr_test"]} '
        produce_trajs = subprocess.call(args = command, shell=True)


        with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
            all_trajs = pickle.load(f)
        agent.demo_visitation_calculator(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts

        if init_vars["run_mode"] == "restart":
            with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("success_ratio, L_D_theta, L_phi, validation_L_phi, min_y, grad_L_D_theta_norm_one, grad_L_phi_theta_norm_one, reg_term_norm_one, episode_number, episode_time \n")
            with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("output progress")
            min_range = 0
            max_range = init_vars["num_theta_updates"]

        elif init_vars["run_mode"] == "continue":
            loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
            min_range = int(loss[-1,3])
            max_range = min_range + init_vars["num_theta_updates"]
            loss = 0

        for epis in range(min_range, max_range):
            start = time.time()
            Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
            agent.calc_pi_theta() # Policy is defined using softmax from pi_theta
            agent.calc_grad_Q_theta(thresh=init_vars["thresh_Q_theta"])
            grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value
            agent.calc_grad_pi_theta()

            # following 3 only necessary if we want to modify the cost function
            # using specifications
            # ---------------------
            agent.calc_y_theta(thresh=init_vars["thresh_Y_theta"])
            # agent.calc_y_theta_eval(thresh=init_vars["thresh_Y_theta"])

            agent.calc_grad_y_theta(thresh=init_vars["thresh_grad_Y_theta"])
            grad_L_phi_theta = agent.calc_grad_L_phi_theta()

            # ---------------------
            grad_j = grad_L_D_theta - lam_reg * reg_term + lam_phi * grad_L_phi_theta
            agent.update_theta(grad_j)
            L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])
            L_phi, min_y =  agent.calc_L_phi_theta()
            L_phi_eval, min_y_eval = agent.calc_L_phi_theta_eval()

            end = time.time()
            epis_time = end-start

            grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
            grad_L_phi_theta_norm_one = np.mean(np.abs(grad_L_phi_theta))
            reg_term_norm_one = np.mean(np.abs(reg_term))

            if epis % init_vars["val_period"] == 0:
                if init_vars["verbose"] == True:
                    print ("SAVING THE MODELS .............")
                    print ()
                torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + "run_" + str(run) + ".pt")
                # validation_L_phi, validation_L_phi_eval = val(init_vars, mdp_objs_test, models_address, run)

                if init_vars["verbose"] == True:
                    print ("**** eval game performance")
                num_eval_episodes = 100
                success_ratio = eval_game_performance(init_vars, agent, num_eval_episodes)

                if init_vars["verbose"] == True:
                    print ("****************************************")
                    print (f"episode number:  {epis}")
                    print (f"L_D_theta: {L_D_theta:.4}")
                    print (f"L_phi: {L_phi:.4}")
                    print (f"L_phi_eval: {L_phi_eval:>6}")
                    print (f"success_ratio: {success_ratio:>4}")
                    # print (f"validation L_phi: {validation_L_phi:.4}")
                    # print (f"validation L_phi eval: {validation_L_phi_eval:.4}")
                    print (f"epis_time: {epis_time:.4}")
                    print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
                    print (f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}")
                    print (f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}")
                # print ("reward_net_time: {:.4}".format(reward_net_time))
                # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
                # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
                # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
                # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
                # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
                # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
                # print ("update_theta_TIME: {:.4}".format(s9-s8))

                with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "a") as file2:
                    file2.write("****************************************\n")
                    file2.write(f"episode number:  {epis}\n")
                    file2.write(f"L_D_theta: {L_D_theta:.4}\n")
                    file2.write(f"L_phi: {L_phi:.4}\n")
                    file2.write(f"L_phi_eval: {L_phi_eval:.4}\n")
                    file2.write(f"success_ratio: {success_ratio:.4}\n")
                    # file2.write(f"validation_L_phi: {validation_L_phi:.4}\n")
                    # file2.write(f"validation_L_phi_eval: {validation_L_phi_eval:.4}\n")
                    file2.write(f"epis_time: {epis_time:.4}\n")
                    file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
                    file2.write(f"grad_L_phi_theta_norm_one times lam_phi: {lam_phi * grad_L_phi_theta_norm_one:.6}\n")
                    file2.write(f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}\n")

                with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "a") as file1:
                    file1.write(f'{success_ratio:>.5} {L_D_theta:>20} {L_phi:>20} {min_y:>15} {L_phi_eval:>20} {min_y_eval:>15} {grad_L_D_theta_norm_one:.10} {grad_L_phi_theta_norm_one:.10} {reg_term_norm_one:.10} {epis} {epis_time:.4} \n')

    return
            # if epis % init_vars["save_model_interval"] == 0:
            #     torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + str(epis) + ".pt")

def train_IRL_bits(init_vars):
    """
    This function trains the reward network on gridworlds produced by produce_grids and on trajectories
    produced by produce_opt_trajs_mult_grids
    >>> Currently it only trains the first gridworld and ignores "num_training_grids"
    """
    if init_vars["verbose"]:
        print("producing new trajectories for the new DFA")
    command  = f'python src/main_irl2.py --main_function="produce_opt_trajs_time_limited_bits" --fileName_grids={init_vars["fileName_grids"]} --fileName_tr_test={init_vars["fileName_tr_test"]} '
    produce_trajs = subprocess.call(args = command, shell=True)
    if init_vars["verbose"]:
        print("---- TRAIN IRL --- \n")

    device = torch.device(init_vars["device"])
    lam_phi = init_vars["lam_phi"]
    lam_reg = init_vars["lam_reg"]


    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        mdp_objs_test = pickle.load(f)

    # retrieving the folder addresses
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = produce_envs_from_MDP_bits(init_vars, copy.deepcopy(mdp_objs_tr[0]))
    envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
    agent = produce_agent_bits(envs, init_vars, agent_mode="train")

    agent.logs_address = logs_address


    # # save the gridworld to be used for eval later
    # # with open(logs_address + "MDP_train.pkl", "wb") as f:
    # #     pickle.dump(MDP_obj, f)
    for run in range(init_vars['num_runs']):

        # if init_vars["run_mode"]  == "continue":
        #     # Load models
        #     if agent.reward_net_name == "MLP":
        #         agent.reward_net = agent_env.net_MLP(envs[0], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
        #     if agent.reward_net_name == "CNN":
        #         pass
        #     agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))

        with open(init_vars["trajs_address"] + "all_trajectories_bits.pkl", "rb") as f:
            all_trajs = pickle.load(f)
        agent.demo_visitation_calculator(init_vars["num_trajs_used"], all_trajs[0]) # fills in agent.visitation_counts

        if init_vars["run_mode"] == "restart":
            with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("success_ratio, L_D_theta, grad_L_D_theta_norm_one, reg_term_norm_one, episode_number, episode_time \n")
            with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "w") as file:
                file.write("output progress")
            min_range = 0
            max_range = init_vars["num_theta_updates"]

        elif init_vars["run_mode"] == "continue":
            loss = np.loadtxt(logs_address +"loss_per_episode.txt",skiprows=1)
            min_range = int(loss[-1,3])
            max_range = min_range + init_vars["num_theta_updates"]
            loss = 0

        for epis in range(min_range, max_range):
            start = time.time()
            Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
            agent.calc_pi_theta() # Policy is defined using softmax from pi_theta
            agent.calc_grad_Q_theta(thresh=init_vars["thresh_Q_theta"])
            grad_L_D_theta, reg_term = agent.calc_grad_L_D_theta(init_vars["num_trajs_used"]) # returns regularized value
            agent.calc_grad_pi_theta()
            # following 3 only necessary if we want to modify the cost function
            # using specifications
            # ---------------------

            # ---------------------
            grad_j = grad_L_D_theta - lam_reg * reg_term
            agent.update_theta(grad_j)
            L_D_theta = agent.calc_L_D_theta(init_vars["num_trajs_used"])

            end = time.time()
            epis_time = end-start

            grad_L_D_theta_norm_one = np.mean(np.abs(grad_L_D_theta))
            reg_term_norm_one = np.mean(np.abs(reg_term))

            if epis % init_vars["val_period"] == 0:
                if init_vars["verbose"] == True:
                    print ("SAVING THE MODELS .............")
                    print ()
                torch.save(agent.reward_net.state_dict(), models_address + "reward_net_run_" + str(run) + ".pt")
                # validation_L_phi, validation_L_phi_eval = val(init_vars, mdp_objs_test, models_address, run)

                if init_vars["verbose"] == True:
                    print ("**** eval game performance")
                num_eval_episodes = 100
                success_ratio = eval_game_performance_bits(init_vars, agent, num_eval_episodes)

                if init_vars["verbose"] == True:
                    print ("****************************************")
                    print (f"episode number:  {epis}")
                    print (f"L_D_theta: {L_D_theta:.4}")
                    # print (f"success_ratio: {success_ratio:.4}")
                    # print (f"validation L_phi: {validation_L_phi:.4}")
                    # print (f"validation L_phi eval: {validation_L_phi_eval:.4}")
                    print (f"epis_time: {epis_time:.4}")
                    print (f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}")
                    print (f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}")
                    print (f"success ratio ---- {success_ratio}")
                # print ("reward_net_time: {:.4}".format(reward_net_time))
                # print ("calc_Q_theta_TIME: {:.4}".format(s2-s0))
                # print ("Q_theta_counter: {:4}".format(Q_theta_counter))
                # print ("calc_grad_Q_theta_TIME: {:.4}".format(s3-s2))
                # print ("calc_y_theta_TIME: {:.4}".format(s6-s4))
                # print ("calc_grad_y_theta_TIME: {:.4}".format(s7-s6))
                # print ("calc_grad_L_phi_theta_TIME: {:.4}".format(s8-s7))
                # print ("update_theta_TIME: {:.4}".format(s9-s8))


                with open(logs_address + "output_progress_" + "run_" + str(run) + ".txt", "a") as file2:
                    file2.write("****************************************\n")
                    file2.write(f"episode number:  {epis}\n")
                    file2.write(f"L_D_theta: {L_D_theta:.4}\n")
                    # file2.write(f"success_ratio: {success_ratio:.4}\n")
                    # file2.write(f"validation_L_phi: {validation_L_phi:.4}\n")
                    # file2.write(f"validation_L_phi_eval: {validation_L_phi_eval:.4}\n")
                    file2.write(f"epis_time: {epis_time:.4}\n")
                    file2.write(f"grad_L_D_theta_norm_one: {grad_L_D_theta_norm_one:.6}\n")
                    file2.write(f"reg_term_norm_one times lam_reg: {lam_reg * reg_term_norm_one:.6}\n")
                    file2.write(f"success_ratio: {success_ratio:.6}\n")


                with open(logs_address + "loss_per_episode_" + "run_" + str(run) + ".txt", "a") as file1:
                    file1.write(f'{success_ratio:>6} {L_D_theta:>20} {grad_L_D_theta_norm_one:.10} {reg_term_norm_one:.10} {epis} {epis_time:.4} \n')


    return
        # if epis % init_vars["save_model_interval"] == 0:
        #     torch.save(agent.reward_net.state_dict(), models_address + "reward_net_" + str(epis) + ".pt")



# VALIDATION
# ---------------------------
def val(init_vars, mdp_objs_test, models_address, run):
    device = torch.device(init_vars["device"])
    L_phi_list = []
    L_phi_eval_list = []
    for mdp_counter, mdp_read in enumerate(mdp_objs_test[15:]):
        # print (f"### gridworld number: {mdp_counter+1}")
        MDP_obj = copy.deepcopy(mdp_read)
        envs = produce_envs_from_MDP(init_vars,MDP_obj)
        agent = produce_agent(envs, init_vars, agent_mode = "val")

        # Load models
        if agent.reward_net_name == "MLP":
            agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
        if agent.reward_net_name == "CNN":
            pass
        agent.reward_net.load_state_dict(torch.load(models_address + "reward_net_" + "run_" + str(run) + ".pt"))
        agent.reward_net.eval()
        # reward_theta = agent.eval_reward()
        L_phi, min_y, L_phi_eval, min_y_eval = agent.eval_L_phi( init_vars["num_trajs_used"], init_vars["thresh_Q_theta"],
                                                init_vars["thresh_Y_theta"] )

        L_phi_list.append(L_phi)
        L_phi_eval_list.append(L_phi_eval)

        # uncomment following line if you want real world game performance evaluation
        # eval_game_performance(init_vars, agent)
    return np.mean(L_phi_list), np.mean(L_phi_eval_list)

def val_non_local_r(init_vars, envs_objs_test, models_address, run):
    device = torch.device(init_vars["device"])
    L_phi_list = []
    for mdp_counter, mdp_read in enumerate(envs_objs_test[15:]):
        # print (f"### gridworld number: {mdp_counter+1}")
        MDP_obj = copy.deepcopy(envs_read[0])
        envs = produce_envs_from_MDP(init_vars,MDP_obj)
        agent = produce_agent_RL_non_local_r(envs, init_vars, agent_mode = "val")

        # Load models
        if agent.reward_net_name == "MLP":
            pass
        if agent.reward_net_name == "CNN":
            agent.reward_net = agent_env.net_CNN_non_local_r(envs[0], out_dim=1).to(device)

        agent.reward_net.load_state_dict(torch.load(models_address + "reward_net_" + "run_" + str(run) + ".pt"))
        agent.reward_net.eval()
        # reward_theta = agent.eval_reward()
        L_phi, min_y = agent.eval_L_phi( init_vars["num_trajs_used"], init_vars["thresh_Q_theta"],
                                                init_vars["thresh_Y_theta"] )

        L_phi_list.append(L_phi)
        # uncomment following line if you want real world game performance evaluation
        # eval_game_performance(init_vars, agent)
    return np.mean(L_phi_list)

def val_BC(init_vars, envs_objs_test, models_address):
    device = torch.device(init_vars["device"])
    main_function = init_vars["main_function"]
    L_phi_list = []
    for mdp_counter, mdp in enumerate(envs_objs_test[10:]):
        # print (f"### gridworld number: {mdp_counter+1}")
        agent = produce_agent_BC(envs, init_vars, agent_mode = "test")
        MDP_obj = copy.deepcopy(mdp)

        # Load models
        if agent.policy_net_name == "MLP":
            if main_function == "test_BC_v1":
                agent.policy_net = agent_env.net_MLP_pol_v1(envs[0], out_dim=4).to(device)
            elif main_function == "test_BC_v2":
                agent.policy_net = agent_env.net_MLP_pol_v2(envs[0], envs[1], out_dim=4).to(device)
            elif main_function == "test_BC_local":
                agent.policy_net = agent_env.net_MLP_pol_local(envs[0], envs[1], n_dim=agent.policy_input_size, out_dim=4).to(device)
        elif agent.policy_net_name == "CNN":
            if main_function == "test_BC_v1":
                agent.policy_net = agent_env.net_CNN_pol_v1(envs[0], out_dim=4).to(device)
            elif main_function == "test_BC_v2":
                agent.policy_net = agent_env.net_CNN_pol_v2(envs[0], out_dim=4).to(device)
        agent.policy_net.load_state_dict(torch.load(models_address + "policy_net.pt"))
        agent.policy_net.eval()

        L_phi, min_y = agent.eval_L_phi( init_vars["num_trajs_used"], init_vars["thresh_Y_theta"] )

        L_phi_list.append(L_phi)
    return np.mean(L_phi_list)


# TESTING
# ---------------------------

def test(init_vars):
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    test_address = return_address_test(init_vars)
    device = torch.device(init_vars["device"])

    grids_address = return_address_grids(init_vars["fileName_grids"])

    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # test_type=["train_grid", "single_test_grid", "multiple_test_grids"]

    # init_vars["thresh_Q_theta"] = 1e-4
    # init_vars["thresh_grad_Q_theta"] = 1e-4
    # init_vars["thresh_Y_theta"] = 1e-9
    # init_vars["thresh_grad_Y_theta"] = 1e-4
    # init_vars["thresh_Q_theta"] = 1e-4
    # init_vars["thresh_grad_Q_theta"] = 1e-4
    init_vars["thresh_Y_theta"] = 1e-3
    # init_vars["thresh_grad_Y_theta"] = 1e-4

    if not ("reward_input_size" in init_vars.keys()):
        init_vars["reward_input_size"] = 3

    init_vars['num_runs'] = 1
    for run in range(init_vars['num_runs']):
        print ("---- run number {} ----".format( run))
        if init_vars["test_type"] in ["train_grid", "single_test_grid"]:
            if init_vars["test_type"] == "train_grid":
                envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))
            elif init_vars["test_type"] == "single_test_grid":
                envs = produce_envs_from_MDP(init_vars, copy.deepcopy(envs_objs_test[0]))
            agent = produce_agent(envs, init_vars, agent_mode = "test")

            # print(agent.MDP.grid[:,7,4])
            # print(agent.DFA.positive_reward)
            # exit()

            # Load models
            if agent.reward_net_name == "MLP":
                agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
            if agent.reward_net_name == "CNN":
                pass
            agent.reward_net.load_state_dict(torch.load(models_address + "reward_net" +"_run_" + str(run) + ".pt"))
            agent.reward_net.eval()
            reward_theta = agent.eval_reward()
            # print(reward_theta[:,:,0,0])
            # exit()


            # with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
            #     all_trajs = pickle.load(f)

            all_trajs = get_trajectories(init_vars)


            L_phi, min_y, y_theta, Q_theta, pi_theta, vis_count_opt, pi_opt_soft, pi_opt_greedy = agent.eval_task_performance(
                   init_vars["num_trajs_used"], init_vars["thresh_Q_theta"], init_vars["thresh_Y_theta"], all_trajs)
            num_eval_episodes = 5
            success_ratio, lose_ratio = eval_game_performance(init_vars, agent, num_eval_episodes)
            print(success_ratio)
            print(lose_ratio)
            print(agent.y_theta[:,:,0])
            agent.y_theta[4,6:,0] = 0
            agent.y_theta[4,:3,0] = 0
            # agent.y_theta[6,3:6,0] = 1
            # agent.y_theta[7,3:6,0] = 1
            # agent.y_theta[8,3:6,0] = 1
            # agent.y_theta[8,4,0] = 1
            fig, ax = plt.subplots()
            im = ax.imshow(agent.y_theta[:,:,0])
            # im = ax.imshow(agent.V_theta[:,:,0], norm=mc.Normalize(vmax=1))
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('', rotation=-90, va="bottom")
            plt.show()
            # exit()

            with open('pi_theta_9_roady_100_hor_20.pkl', 'wb') as f:
                pickle.dump(agent.pi_theta, f)
            # exit()

            print("SAVING THE FILES ....")
            np.save(test_address+"reward_theta"+"_run_"+str(run),reward_theta)
            np.save(test_address+"y_theta"+"_run_"+str(run),y_theta)
            np.save(test_address+"Q_theta"+"_run_"+str(run),Q_theta)
            np.save(test_address+"pi_theta"+"_run_"+str(run),pi_theta)
            np.save(test_address+"vis_count_opt"+"_run_"+str(run),vis_count_opt)
            np.save(test_address+"pi_opt_soft"+"_run_"+str(run),pi_opt_soft)
            np.save(test_address+"pi_opt_greedy"+"_run_"+str(run),pi_opt_greedy)


        elif init_vars["test_type"] == "multiple_test_grids":
            L_phi_list = []
            success_ratio_list = []
            for mdp_counter, mdp in enumerate(envs_objs_test[:10]):
                print (f"### gridworld number: {mdp_counter+1}")
                agent = produce_agent(envs, init_vars, agent_mode = "test")
                MDP_obj = copy.deepcopy(mdp)

                # Load models
                if agent.reward_net_name == "MLP":
                    agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
                if agent.reward_net_name == "CNN":
                    pass
                agent.reward_net.load_state_dict(torch.load(models_address + "reward_net_run_" + str(run) + ".pt"))
                agent.reward_net.eval()
                reward_theta = agent.eval_reward()

                L_phi, min_y = agent.eval_L_phi( init_vars["num_trajs_used"], init_vars["thresh_Q_theta"],
                                                        init_vars["thresh_Y_theta"] )
                num_eval_episodes = 10
                success_ratio = eval_game_performance(init_vars, agent, num_eval_episodes)
                success_ratio_list.appen(success_ratio)

                L_phi_list.append([L_phi, min_y])
                # uncomment following line if you want real world game performance evaluation
                # eval_game_performance(init_vars, agent)
            print("SAVING THE FILES ....")
            with open(test_address+"L_phi_list_run_"+str(run)+ ".pkl", "wb") as f:
                pickle.dump(L_phi_list, f)
            with open(test_address+"success_ratio_list_run_"+str(run)+ ".pkl", "wb") as f:
                pickle.dump(success_ratio_list, f)

            # np.save(test_address+"reward_theta",reward_theta)
            # np.save(test_address+"y_theta",y_theta)
            # np.save(test_address+"Q_theta",Q_theta)
            # np.save(test_address+"pi_theta",pi_theta)
            # np.save(test_address+"vis_count_opt",vis_count_opt)
            # np.save(test_address+"pi_opt_soft",pi_opt_soft)
            # np.save(test_address+"pi_opt_greedy",pi_opt_greedy)

def test_active(init_vars):
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    test_address = return_address_test(init_vars)
    device = torch.device(init_vars["device"])

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # test_type=["train_grid", "single_test_grid", "multiple_test_grids"]

    if not ("reward_input_size" in init_vars.keys()):
        init_vars["reward_input_size"] = 3
    init_vars['num_runs'] = 10
    for run in range(init_vars['num_runs']):
        print ("---- run number {} ----".format( run))
        if init_vars["test_type"] in ["train_grid", "single_test_grid"]:
            if init_vars["test_type"] == "train_grid":
                MDP_obj, DFA_obj_GT, PA_obj_GT = produce_envs_only_eval_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))
                _, DFA_obj, PA_obj = produce_envs_only_tr_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[1]))
                envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
            elif init_vars["test_type"] == "single_test_grid":
                MDP_obj, DFA_obj_GT, PA_obj_GT = produce_envs_only_eval_from_MDP(init_vars, copy.deepcopy(envs_objs_test[0][0]))
                _, DFA_obj, PA_obj = produce_envs_only_tr_from_MDP(init_vars, copy.deepcopy(envs_objs_test[1][0]))
                envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
            agent = produce_agent(envs, init_vars, agent_mode = "test")


            # Load models
            if agent.reward_net_name == "MLP":
                agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
            if agent.reward_net_name == "CNN":
                pass
            agent.reward_net.load_state_dict(torch.load(models_address + "reward_net" +"_run_" + str(run) + ".pt"))
            agent.reward_net.eval()
            reward_theta = agent.eval_reward()


            with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
                all_trajs = pickle.load(f)


            L_phi, min_y, y_theta, Q_theta, pi_theta, vis_count_opt, pi_opt_soft, pi_opt_greedy = agent.eval_task_performance(
                   init_vars["num_trajs_used"], init_vars["thresh_Q_theta"], init_vars["thresh_Y_theta"], all_trajs[0])
            num_eval_episodes = 500
            success_ratio = eval_game_performance(init_vars, agent, num_eval_episodes)

            print("SAVING THE FILES ....")
            np.save(test_address+"reward_theta"+"_run_"+str(run),reward_theta)
            np.save(test_address+"y_theta"+"_run_"+str(run),y_theta)
            np.save(test_address+"Q_theta"+"_run_"+str(run),Q_theta)
            np.save(test_address+"pi_theta"+"_run_"+str(run),pi_theta)
            np.save(test_address+"vis_count_opt"+"_run_"+str(run),vis_count_opt)
            np.save(test_address+"pi_opt_soft"+"_run_"+str(run),pi_opt_soft)
            np.save(test_address+"pi_opt_greedy"+"_run_"+str(run),pi_opt_greedy)


        elif init_vars["test_type"] == "multiple_test_grids":
            L_phi_list = []
            success_ratio_list = []
            for mdp_counter, mdp in enumerate(envs_objs_test[:10]):
                print (f"### gridworld number: {mdp_counter+1}")
                MDP_obj, DFA_obj_GT, PA_obj_GT = produce_envs_only_eval_from_MDP(init_vars, mdp)
                _, DFA_obj, PA_obj = produce_envs_only_tr_from_MDP(init_vars, mdp)
                envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
                agent = produce_agent(envs, init_vars, agent_mode = "train")
                MDP_obj = copy.deepcopy(mdp)


                # Load models
                if agent.reward_net_name == "MLP":
                    agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
                if agent.reward_net_name == "CNN":
                    pass
                agent.reward_net.load_state_dict(torch.load(models_address + "reward_net_run_" + str(run) + ".pt"))
                agent.reward_net.eval()
                reward_theta = agent.eval_reward()

                L_phi, min_y, _, _ = agent.eval_L_phi( init_vars["num_trajs_used"], init_vars["thresh_Q_theta"],
                                                        init_vars["thresh_Y_theta"] )
                L_phi_list.append([L_phi, min_y])
                num_eval_episodes = 50
                success_ratio = eval_game_performance(init_vars, agent, num_eval_episodes)
                success_ratio_list.append([success_ratio])

            print("SAVING THE FILES ....")
            with open(test_address+"L_phi_list_run_"+str(run)+ ".pkl", "wb") as f:
                pickle.dump(L_phi_list, f)
            with open(test_address+"success_ratio_list_run_"+str(run)+ ".pkl", "wb") as f:
                pickle.dump(success_ratio_list, f)

            # np.save(test_address+"reward_theta",reward_theta)
            # np.save(test_address+"y_theta",y_theta)
            # np.save(test_address+"Q_theta",Q_theta)
            # np.save(test_address+"pi_theta",pi_theta)
            # np.save(test_address+"vis_count_opt",vis_count_opt)
            # np.save(test_address+"pi_opt_soft",pi_opt_soft)
            # np.save(test_address+"pi_opt_greedy",pi_opt_greedy)

def test_memoryless_baseline(init_vars):
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    test_address = return_address_test(init_vars)
    device = torch.device(init_vars["device"])

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # test_type=["train_grid", "single_test_grid", "multiple_test_grids"]


    if not ("reward_input_size" in init_vars.keys()):
        init_vars["reward_input_size"] = 3
    init_vars['num_runs'] = 10
    for run in range(init_vars['num_runs']):
        print ("---- run number {} ----".format( run))
        if init_vars["test_type"] in ["train_grid", "single_test_grid"]:
            if init_vars["test_type"] == "train_grid":
                MDP_obj, DFA_obj_GT, PA_obj_GT = produce_envs_only_eval_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0]))
                _, DFA_obj, PA_obj = produce_envs_only_tr_from_MDP_memoryless(init_vars, copy.deepcopy(mdp_objs_tr[1]))
                envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
            elif init_vars["test_type"] == "single_test_grid":
                MDP_obj, DFA_obj_GT, PA_obj_GT = produce_envs_only_eval_from_MDP(init_vars, copy.deepcopy(envs_objs_test[0][0]))
                _, DFA_obj, PA_obj = produce_envs_only_tr_from_MDP_memoryless(init_vars, copy.deepcopy(envs_objs_test[1][0]))
                envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
            agent = produce_agent(envs, init_vars, agent_mode = "test")

            # Load models
            if agent.reward_net_name == "MLP":
                agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
            if agent.reward_net_name == "CNN":
                pass
            agent.reward_net.load_state_dict(torch.load(models_address + "reward_net" +"_run_" + str(run) + ".pt"))
            agent.reward_net.eval()
            reward_theta = agent.eval_reward()


            with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
                all_trajs = pickle.load(f)


            L_phi, min_y, y_theta, Q_theta, pi_theta, vis_count_opt, pi_opt_soft, pi_opt_greedy = agent.eval_task_performance(
                   init_vars["num_trajs_used"], init_vars["thresh_Q_theta"], init_vars["thresh_Y_theta"], all_trajs[0])
            num_eval_episodes = 500000
            success_ratio = eval_game_performance(init_vars, agent, num_eval_episodes)
            # print(success_ratio)

            print("SAVING THE FILES ....")
            np.save(test_address+"reward_theta"+"_run_"+str(run),reward_theta)
            np.save(test_address+"y_theta"+"_run_"+str(run),y_theta)
            np.save(test_address+"Q_theta"+"_run_"+str(run),Q_theta)
            np.save(test_address+"pi_theta"+"_run_"+str(run),pi_theta)
            np.save(test_address+"vis_count_opt"+"_run_"+str(run),vis_count_opt)
            np.save(test_address+"pi_opt_soft"+"_run_"+str(run),pi_opt_soft)
            np.save(test_address+"pi_opt_greedy"+"_run_"+str(run),pi_opt_greedy)


        elif init_vars["test_type"] == "multiple_test_grids":
            L_phi_list = []
            success_ratio_list = []
            for mdp_counter, mdp in enumerate(envs_objs_test[:10]):
                print (f"### gridworld number: {mdp_counter+1}")
                MDP_obj, DFA_obj_GT, PA_obj_GT = produce_envs_only_eval_from_MDP(init_vars, mdp)
                _, DFA_obj, PA_obj = produce_envs_only_tr_from_MDP(init_vars, mdp)
                envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
                agent = produce_agent(envs, init_vars, agent_mode = "test")
                MDP_obj = copy.deepcopy(mdp)

                # Load models
                if agent.reward_net_name == "MLP":
                    agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
                if agent.reward_net_name == "CNN":
                    pass
                agent.reward_net.load_state_dict(torch.load(models_address + "reward_net_run_" + str(run) + ".pt"))
                agent.reward_net.eval()
                reward_theta = agent.eval_reward()

                L_phi, min_y, _, _ = agent.eval_L_phi( init_vars["num_trajs_used"], init_vars["thresh_Q_theta"],
                                                        init_vars["thresh_Y_theta"] )
                L_phi_list.append([L_phi, min_y])
                num_eval_episodes = 500
                success_ratio = eval_game_performance(init_vars, agent, num_eval_episodes)
                success_ratio_list.append([success_ratio])

            print("SAVING THE FILES ....")
            with open(test_address+"L_phi_list_run_"+str(run)+ ".pkl", "wb") as f:
                pickle.dump(L_phi_list, f)
            with open(test_address+"success_ratio_list_run_"+str(run)+ ".pkl", "wb") as f:
                pickle.dump(success_ratio_list, f)

            # np.save(test_address+"reward_theta",reward_theta)
            # np.save(test_address+"y_theta",y_theta)
            # np.save(test_address+"Q_theta",Q_theta)
            # np.save(test_address+"pi_theta",pi_theta)
            # np.save(test_address+"vis_count_opt",vis_count_opt)
            # np.save(test_address+"pi_opt_soft",pi_opt_soft)
            # np.save(test_address+"pi_opt_greedy",pi_opt_greedy)

def test_bits(init_vars):
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    test_address = return_address_test(init_vars)
    device = torch.device(init_vars["device"])

    grids_address = return_address_grids(init_vars["fileName_grids"])

    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # test_type=["train_grid", "single_test_grid", "multiple_test_grids"]

    if not ("reward_input_size" in init_vars.keys()):
        init_vars["reward_input_size"] = 3

    if 'num_runs' in init_vars.keys():
        num_runs = init_vars['num_runs']
    else:
        num_runs = 1

    for run in range(num_runs):
        print ("---- run number {} ----".format(run))
        if init_vars["test_type"] in ["train_grid", "single_test_grid"]:
            if init_vars["test_type"] == "train_grid":
                envs = produce_envs_from_MDP_bits(init_vars, copy.deepcopy(mdp_objs_tr[0]))
            elif init_vars["test_type"] == "single_test_grid":
                envs = produce_envs_from_MDP_bits(init_vars, copy.deepcopy(mdp_objs_tr[1]))
            agent = produce_agent_bits(envs, init_vars, agent_mode="train")

            # Load models
            if agent.reward_net_name == "MLP":
                agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
            if agent.reward_net_name == "CNN":
                pass

            agent.reward_net.load_state_dict(torch.load(models_address + "reward_net" +"_run_" + str(run) + ".pt"))
            agent.reward_net.eval()
            reward_theta = agent.eval_reward()


            with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
                all_trajs = pickle.load(f)



            print("SAVING THE FILES ....")
            np.save(test_address+"reward_theta"+"_run_"+str(run),reward_theta)
            np.save(test_address+"Q_theta"+"_run_"+str(run),Q_theta)
            np.save(test_address+"pi_theta"+"_run_"+str(run),pi_theta)
            np.save(test_address+"vis_count_opt"+"_run_"+str(run),vis_count_opt)
            np.save(test_address+"pi_opt_soft"+"_run_"+str(run),pi_opt_soft)
            np.save(test_address+"pi_opt_greedy"+"_run_"+str(run),pi_opt_greedy)


        elif init_vars["test_type"] == "multiple_test_grids":
            L_phi_list = []
            success_ratio_list = []
            for mdp_counter, mdp in enumerate(envs_objs_test[:10]):
                print (f"### gridworld number: {mdp_counter+1}")
                # envs = produce_envs_from_MDP_bits(init_vars, mdp)
                # agent = produce_agent_bits(envs, init_vars, agent_mode="train")
                # MDP_obj = copy.deepcopy(mdp)

                # retrieving the folder addresses
                MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = produce_envs_from_MDP_bits(init_vars, copy.deepcopy(mdp_objs_tr[0]))
                envs = MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT
                agent = produce_agent_bits(envs, init_vars, agent_mode="train")

                # Load models
                if agent.reward_net_name == "MLP":
                    agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
                elif agent.reward_net_name == "CNN":
                    pass

                agent.reward_net.load_state_dict(torch.load(models_address + "reward_net_run_" + str(run) + ".pt"))
                agent.reward_net.eval()
                reward_theta = agent.eval_reward()
                num_eval_episodes = 100
                success_ratio = eval_game_performance_bits(init_vars, agent, num_eval_episodes)
                success_ratio_list.append(success_ratio)

            print("SAVING THE FILES ....")
            with open(test_address+"success_ratio_list_run_"+str(run)+ ".pkl", "wb") as f:
                pickle.dump(success_ratio_list, f)

def test_non_local_r(init_vars):
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    test_address = return_address_test(init_vars)
    device = torch.device(init_vars["device"])

    grids_address = return_address_grids(init_vars["fileName_grids"])

    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # test_type=["train_grid", "single_test_grid", "multiple_test_grids"]

    if not ("reward_input_size" in init_vars.keys()):
        init_vars["reward_input_size"] = 3


    for run in range(init_vars['num_runs']):
        print ("---- run number {} ----".format( run))
        if init_vars["test_type"] in ["train_grid", "single_test_grid"]:
            if init_vars["test_type"] == "train_grid":
                envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0][0]))
            elif init_vars["test_type"] == "single_test_grid":
                envs = produce_envs_from_MDP(init_vars, copy.deepcopy(envs_objs_test[1][0]))

            agent = produce_agent_RL_non_local_r(envs, init_vars, agent_mode="test")

            # Load models
            if agent.reward_net_name == "MLP":
                # agent.reward_net = agent_env.net_MLP(envs[0], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
                pass
            if agent.reward_net_name == "CNN":
                agent.reward_net = agent_env.net_CNN_non_local_r(envs[0], out_dim=1).to(device)

            agent.reward_net.load_state_dict(torch.load(models_address + "reward_net" +"_run_" + str(run) + ".pt"))
            agent.reward_net.eval()
            reward_theta = agent.eval_reward()

            with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
                all_trajs = pickle.load(f)

            L_phi, min_y, y_theta, Q_theta, pi_theta, vis_count_opt, pi_opt_soft, pi_opt_greedy = agent.eval_task_performance(
                   init_vars["num_trajs_used"], init_vars["thresh_Q_theta"], init_vars["thresh_Y_theta"], all_trajs[0])

            print("SAVING THE FILES ....")
            np.save(test_address+"reward_theta"+"_run_"+str(run),reward_theta)
            np.save(test_address+"y_theta"+"_run_"+str(run),y_theta)
            np.save(test_address+"Q_theta"+"_run_"+str(run),Q_theta)
            np.save(test_address+"pi_theta"+"_run_"+str(run),pi_theta)
            np.save(test_address+"vis_count_opt"+"_run_"+str(run),vis_count_opt)
            np.save(test_address+"pi_opt_soft"+"_run_"+str(run),pi_opt_soft)
            np.save(test_address+"pi_opt_greedy"+"_run_"+str(run),pi_opt_greedy)


        elif init_vars["test_type"] == "multiple_test_grids":
            L_phi_list = []
            for mdp_counter, mdp in enumerate(envs_objs_test[:10]):
                print (f"### gridworld number: {mdp_counter+1}")
                agent = produce_agent_RL_non_local_r(envs, init_vars, agent_mode="test")
                MDP_obj = copy.deepcopy(mdp)

                # Load models
                if agent.reward_net_name == "MLP":
                    agent.reward_net = agent_env.net_MLP(envs[0], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
                if agent.reward_net_name == "CNN":
                    agent.reward_net = agent_env.net_CNN_non_local_r(envs[0], out_dim=1).to(device)
                agent.reward_net.load_state_dict(torch.load(models_address + "reward_net_run_" + str(run) + ".pt"))
                agent.reward_net.eval()
                reward_theta = agent.eval_reward()

                L_phi, min_y = agent.eval_L_phi( init_vars["num_trajs_used"], init_vars["thresh_Q_theta"],
                                                        init_vars["thresh_Y_theta"] )

                L_phi_list.append([L_phi, min_y])
                # uncomment following line if you want real world game performance evaluation
                # eval_game_performance(init_vars, agent)
            print("SAVING THE FILES ....")
            with open(test_address+"L_phi_list_run_"+str(run)+ ".pkl", "wb") as f:
                pickle.dump(L_phi_list, f)


            # np.save(test_address+"reward_theta",reward_theta)
            # np.save(test_address+"y_theta",y_theta)
            # np.save(test_address+"Q_theta",Q_theta)
            # np.save(test_address+"pi_theta",pi_theta)
            # np.save(test_address+"vis_count_opt",vis_count_opt)
            # np.save(test_address+"pi_opt_soft",pi_opt_soft)
            # np.save(test_address+"pi_opt_greedy",pi_opt_greedy)

def test_BC(init_vars):
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    test_address = return_address_test(init_vars)
    device = torch.device(init_vars["device"])
    main_function = init_vars["main_function"]
    grids_address = return_address_grids(init_vars["fileName_grids"])

    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # test_type=["train_grid", "single_test_grid", "multiple_test_grids"]

    if init_vars["test_type"] in ["train_grid", "single_test_grid"]:
        if init_vars["test_type"] == "train_grid":
            envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0][0]))
        elif init_vars["test_type"] == "single_test_grid":
            envs = produce_envs_from_MDP(init_vars, copy.deepcopy(envs_objs_test[1][0]))
        agent = produce_agent_BC(envs, init_vars, agent_mode = "test")


        # Load models
        if agent.policy_net_name == "MLP":
            if main_function == "test_BC_v1":
                agent.policy_net = agent_env.net_MLP_pol_v1(envs[0], out_dim=4).to(device)
            elif main_function == "test_BC_v2":
                agent.policy_net = agent_env.net_MLP_pol_v2(envs[0], envs[1], out_dim=4).to(device)
            elif main_function == "test_BC_local":
                agent.policy_net = agent_env.net_MLP_pol_local(envs[0], envs[1], n_dim=agent.policy_input_size, out_dim=4).to(device)
        elif agent.policy_net_name == "CNN":
            if main_function == "test_BC_v1":
                agent.policy_net = agent_env.net_CNN_pol_v1(envs[0], out_dim=4).to(device)
            elif main_function == "test_BC_v2":
                agent.policy_net = agent_env.net_CNN_pol_v2(envs[0], envs[1], out_dim=4).to(device)
            elif main_function == "test_BC_local":
                agent.policy_net = agent_env.net_CNN_pol_local(envs[0], envs[1], n_dim=agent.policy_input_size, out_dim=4).to(device)
        agent.policy_net.load_state_dict(torch.load(models_address + "policy_net.pt"))
        agent.policy_net.eval()

        with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
            all_trajs = pickle.load(f)

        L_phi, min_y, y_theta, pi_theta = agent.eval_task_performance( init_vars["num_trajs_used"],
                            init_vars["thresh_Q_theta"], init_vars["thresh_Y_theta"], all_trajs[0])


        print("SAVING THE FILES ....")
        np.save(test_address+"y_theta",y_theta)
        np.save(test_address+"pi_theta",pi_theta)


    elif init_vars["test_type"] == "multiple_test_grids":
        L_phi_list = []
        for mdp_counter, mdp in enumerate(envs_objs_test[:10]):
            print (f"### gridworld number: {mdp_counter+1}")
            agent = produce_agent_BC(envs, init_vars, agent_mode = "test")
            MDP_obj = copy.deepcopy(mdp)

            # Load models
            if agent.policy_net_name == "MLP":
                if main_function == "test_BC_v1":
                    agent.policy_net = agent_env.net_MLP_pol_v1(envs[0], out_dim=4).to(device)
                elif main_function == "test_BC_v2":
                    agent.policy_net = agent_env.net_MLP_pol_v2(envs[0], envs[1], out_dim=4).to(device)
                elif main_function == "test_BC_local":
                    agent.policy_net = agent_env.net_MLP_pol_local(envs[0], envs[1], n_dim=agent.policy_input_size, out_dim=4).to(device)
            elif agent.policy_net_name == "CNN":
                if main_function == "test_BC_v1":
                    agent.policy_net = agent_env.net_CNN_pol_v1(envs[0], out_dim=4).to(device)
                elif main_function == "test_BC_v2":
                    agent.policy_net = agent_env.net_CNN_pol_v2(envs[0], out_dim=4).to(device)
                elif main_function == "test_BC_local":
                    agent.policy_net = agent_env.net_CNN_pol_local(envs[0], envs[1], n_dim=agent.policy_input_size, out_dim=4).to(device)
            agent.policy_net.load_state_dict(torch.load(models_address + "policy_net.pt"))
            agent.policy_net.eval()

            L_phi, min_y = agent.eval_L_phi( init_vars["num_trajs_used"], init_vars["thresh_Y_theta"] )

            L_phi_list.append([L_phi, min_y])
            # uncomment following line if you want real world game performance evaluation
            # eval_game_performance(init_vars, agent)
        print("SAVING THE FILES ....")
        with open(test_address+"L_phi_list.pkl", "wb") as f:
            pickle.dump(L_phi_list, f)

def test_BC_3(init_vars):
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    test_address = return_address_test(init_vars)
    device = torch.device(init_vars["device"])

    grids_address = return_address_grids(init_vars["fileName_grids"])

    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    # test_type=["train_grid", "single_test_grid", "multiple_test_grids"]



    if init_vars["test_type"] in ["train_grid", "single_test_grid"]:
        if init_vars["test_type"] == "train_grid":
            envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0][0]))
        elif init_vars["test_type"] == "single_test_grid":
            envs = produce_envs_from_MDP(init_vars, copy.deepcopy(envs_objs_test[1][0]))

        agent = produce_agent_BC_3(envs, init_vars, agent_mode = "test")

        # Load models
        if agent.policy_net_name in ["MLP", "MLP_pol"]:
            agent.policy_net0 = agent_env.net_MLP_pol_ind(envs[0], out_dim=4).to(device)
            agent.policy_net1 = agent_env.net_MLP_pol_ind(envs[0], out_dim=4).to(device)
            agent.policy_net2 = agent_env.net_MLP_pol_ind(envs[0], out_dim=4).to(device)

        elif agent.policy_net_name == "CNN":
            agent.policy_net0 = agent_env.net_CNN_pol_ind(envs[0], out_dim=4).to(device)
            agent.policy_net1 = agent_env.net_CNN_pol_ind(envs[0], out_dim=4).to(device)
            agent.policy_net2 = agent_env.net_CNN_pol_ind(envs[0], out_dim=4).to(device)

        agent.policy_net0.load_state_dict(torch.load(models_address + "policy_net0.pt"))
        agent.policy_net1.load_state_dict(torch.load(models_address + "policy_net1.pt"))
        agent.policy_net2.load_state_dict(torch.load(models_address + "policy_net2.pt"))

        agent.policy_net0.eval()
        agent.policy_net1.eval()
        agent.policy_net2.eval()


        with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
            all_trajs = pickle.load(f)

        L_phi, min_y, y_theta, pi_theta = agent.eval_task_performance( init_vars["num_trajs_used"],
                            init_vars["thresh_Q_theta"], init_vars["thresh_Y_theta"], all_trajs[0])

        print("SAVING THE FILES ....")
        np.save(test_address+"y_theta",y_theta)
        np.save(test_address+"pi_theta",pi_theta)

    elif init_vars["test_type"] == "multiple_test_grids":
        L_phi_list = []
        for mdp_counter, mdp in enumerate(envs_objs_test[:10]):
            print (f"### gridworld number: {mdp_counter+1}")
            agent = produce_agent_BC_3(envs, init_vars, agent_mode = "test")
            MDP_obj = copy.deepcopy(mdp)

            # Load models
            if agent.policy_net_name in ["MLP", "MLP_pol"]:
                agent.policy_net0 = agent_env.net_MLP_pol_ind(envs[0], out_dim=4).to(device)
                agent.policy_net1 = agent_env.net_MLP_pol_ind(envs[0], out_dim=4).to(device)
                agent.policy_net2 = agent_env.net_MLP_pol_ind(envs[0], out_dim=4).to(device)

            elif agent.policy_net_name == "CNN":
                agent.policy_net0 = agent_env.net_CNN_pol_ind(envs[0], out_dim=4).to(device)
                agent.policy_net1 = agent_env.net_CNN_pol_ind(envs[0], out_dim=4).to(device)
                agent.policy_net2 = agent_env.net_CNN_pol_ind(envs[0], out_dim=4).to(device)

            agent.policy_net0.load_state_dict(torch.load(models_address + "policy_net0.pt"))
            agent.policy_net1.load_state_dict(torch.load(models_address + "policy_net1.pt"))
            agent.policy_net2.load_state_dict(torch.load(models_address + "policy_net2.pt"))

            agent.policy_net0.eval()
            agent.policy_net1.eval()
            agent.policy_net2.eval()

            L_phi, min_y = agent.eval_L_phi( init_vars["num_trajs_used"], init_vars["thresh_Y_theta"] )

            L_phi_list.append([L_phi, min_y])
            # uncomment following line if you want real world game performance evaluation
            # eval_game_performance(init_vars, agent)
        print("SAVING THE FILES ....")
        with open(test_address+"L_phi_list.pkl", "wb") as f:
            pickle.dump(L_phi_list, f)

def test_base(init_vars):
    logs_address, models_address = return_addresses_logs_models(init_vars["fileName_tr_test"])
    test_address = return_address_test(init_vars)
    device = torch.device(init_vars["device"])

    grids_address = return_address_grids(init_vars["fileName_grids"])
    with open(grids_address + "mdp_objs_test.pkl","rb") as f:
        envs_objs_test = pickle.load(f)
    with open(grids_address + "mdp_objs_tr.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)

    if not ("reward_input_size" in init_vars.keys()):
        init_vars["reward_input_size"] = 3

    if init_vars["test_type"] in ["train_grid", "single_test_grid"]:
        if init_vars["test_type"] == "train_grid":
            envs = produce_envs_from_MDP(init_vars, copy.deepcopy(mdp_objs_tr[0][0]))
        elif init_vars["test_type"] == "single_test_grid":
            envs = produce_envs_from_MDP(init_vars, copy.deepcopy(envs_objs_test[1][0]))
        agent = produce_agent(envs, init_vars, agent_mode = "test")

        # Load models
        if agent.reward_net_name == "MLP":
            agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
        if agent.reward_net_name == "CNN":
            pass
        agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))
        agent.reward_net.eval()
        reward_theta = agent.eval_reward()

        with open(init_vars["trajs_address"] + "all_trajectories.pkl", "rb") as f:
            all_trajs = pickle.load(f)


        L_phi, min_y, y_theta, Q_theta, pi_theta, vis_count_opt, pi_opt_soft, pi_opt_greedy = agent.eval_task_performance_base(
               init_vars["num_trajs_used"], init_vars["thresh_Q_theta"], init_vars["thresh_Y_theta"], all_trajs[0])

        print("SAVING THE FILES ....")
        np.save(test_address+"reward_theta",reward_theta)
        np.save(test_address+"y_theta",y_theta)
        np.save(test_address+"Q_theta",Q_theta)
        np.save(test_address+"pi_theta",pi_theta)
        np.save(test_address+"vis_count_opt",vis_count_opt)
        np.save(test_address+"pi_opt_soft",pi_opt_soft)
        np.save(test_address+"pi_opt_greedy",pi_opt_greedy)

    elif init_vars["test_type"] == "multiple_test_grids":
        L_phi_list = []
        for mdp_counter, mdp in enumerate(envs_objs_test[:10]):
            print (f"### gridworld number: {mdp_counter+1}")
            agent = produce_agent(envs, init_vars, agent_mode = "test")

            # Load models
            if agent.reward_net_name == "MLP":
                agent.reward_net = agent_env.net_MLP(envs[0], envs[1], n_dim=init_vars["reward_input_size"], out_dim=1).to(device)
            if agent.reward_net_name == "CNN":
                pass
            agent.reward_net.load_state_dict(torch.load(models_address + "reward_net.pt"))
            agent.reward_net.eval()
            reward_theta = agent.eval_reward()

            L_phi, min_y = agent.eval_L_phi_base( init_vars["num_trajs_used"], init_vars["thresh_Q_theta"],
                                                    init_vars["thresh_Y_theta"] )


            L_phi_list.append((L_phi, min_y))

            # uncomment following line if you want real world game performance evaluation
            # eval_game_performance(init_vars, agent)

        print("SAVING THE FILES ....")
        with open(test_address+"L_phi_list.pkl", "wb") as f:
            pickle.dump(L_phi_list, f)

def eval_game_performance_base(init_vars, original_MDP_obj, envs_base, agent_base):
    # this function produces trajecories based on the learned reward (extract a policy from
    # the reward on the given environment first)
    assert not original_MDP_obj is envs_base[0]
    Q_theta_counter, reward_net_time = agent_base.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
    agent_base.calc_pi_theta() # Policy is defined using softmax from pi_theta
    device = torch.device(init_vars["device"])
    envs = produce_envs_from_MDP(init_vars,original_MDP_obj)
    agent = produce_agent(envs, init_vars, agent_mode = "val")
    MDP_obj, DFA_obj, PA_obj, DFA_obj_GT, PA_obj_GT = envs[0], envs[1], envs[2], envs[3], envs[4]
    MDP_obj_base, PA_obj_base = envs_base[0], envs_base[2]
    assert not MDP_obj is MDP_obj_base
    game_won_counter = 0
    game_over_counter = 0
    trajs = []
    for traj_number in range(1000):
        traj = []
        print("\n\n\n\n### traj_number " + str(traj_number) + "###")
        old_mdp_state, old_dfa_state = PA_obj.reset()
        PA_obj_base.reset()
        MDP_obj_base.mdp_state[0] = old_mdp_state[0]
        MDP_obj_base.mdp_state[1] = old_mdp_state[1]
        assert (MDP_obj.mdp_state == MDP_obj_base.mdp_state).all()
        reward = 0
        terminal = False
        traj_steps = 0
        while not terminal:
            traj_steps += 1
            # print ("current_traj_steps:",  traj_steps)
            action_idx, action = agent_base.select_action_soft_pi_theta() # cntr selects an action among permitable actions
            traj.append((reward, old_mdp_state, old_dfa_state, action_idx))
            next_mdp_state, next_dfa_state, reward, _ = PA_obj.step(action_idx)
            PA_obj_base.step(action_idx)
            assert (MDP_obj.mdp_state == MDP_obj_base.mdp_state).all()

            # print ("---------- action_probs: {}".format(action_probs))
            # print(str((state,action)) + "; ")

            i,j = next_mdp_state[0], next_mdp_state[1]
            if init_vars["verbose"]:
                pass
                # print ("state before action ----> " + "[" + str(mdp_state[0]) +", " +
                #         str(mdp_state[1]) + "]" )
                # print ("action ----> "  + action)
                # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                # print ("---------------------")
            game_over, game_won = PA_obj.is_terminal()
            terminal = game_over or game_won # terminal refers to next state

            if game_over:
                game_over_counter += 1
                print("GAME OVER!!!")
                traj.append("game over")


            if game_won:
                game_won_counter += 1
                print("GAME WON!!!")
                traj.append("game won")


            old_mdp_state, old_dfa_state = copy.deepcopy(next_mdp_state), next_dfa_state
        # THIS IS THE END OF ONE trajectory

        trajs.append(traj)
    success_ratio = game_won_counter/(game_won_counter+game_over_counter)
    print("success_ratio: ", success_ratio)

    with open(trajs_address + 'trajectories.pkl', 'wb') as f:
        pickle.dump(trajs, f)

def eval_game_performance(init_vars, agent, num_episodes):
    device = torch.device(init_vars["device"])
    Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
    agent.calc_pi_theta() # Policy is defined using softmax from pi_theta

    MDP_obj, DFA_obj, PA_obj, DFA_GT, PA_GT = agent.MDP, agent.DFA, agent.PA, agent.DFA_GT, agent.PA_GT
    game_won_counter = 0
    game_over_counter = 0
    time_over_counter = 0
    traj_steps_limit = init_vars["time_limit"]
    traj_steps_limit = 170
    print(traj_steps_limit)
    trajs = []
    for traj_number in range(num_episodes):
        traj = []
        traj_mdp = []
        # print("\n\n\n\n### traj_number " + str(traj_number) + "###")
        old_mdp_state, old_dfa_state = PA_obj.reset()
        _, old_DFA_GT_state = PA_GT.reset()

        PA_GT.set_MDP_state(old_mdp_state[0], old_mdp_state[1])
        reward = 0
        terminal = False
        traj_steps = 0

        while not terminal and traj_steps < traj_steps_limit:
            traj_steps += 1
            # print ("current_traj_steps:",  traj_steps)
            action_idx, action = agent.select_action_soft_pi_theta() # cntr selects an action among permitable actions
            # action_idx, action = agent.select_optimal_action_greedy()

            traj.append((reward, old_mdp_state, old_dfa_state, old_DFA_GT_state, action_idx))
            traj_mdp.append(tuple(old_mdp_state))
            next_mdp_state, next_dfa_state, _, _ = PA_obj.step(action_idx)
            next_mdp_state_GT, next_DFA_GT_state, reward, _ = PA_GT.step(action_idx)
            assert next_mdp_state[0] == next_mdp_state_GT[0]
            assert next_mdp_state[1] == next_mdp_state_GT[1]



            # print ("---------- action_probs: {}".format(action_probs))
            # print(str((state,action)) + "; ")

            i,j = next_mdp_state[0], next_mdp_state[1]
            if init_vars["verbose"]:
                pass
                # print ("state before action ----> " + "[" + str(mdp_state[0]) +", " +
                #         str(mdp_state[1]) + "]" )
                # print ("action ----> "  + action)
                # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                # print ("---------------------")
            game_over, game_won = PA_obj.is_terminal()
            terminal = game_over or game_won # terminal refers to next state

            if game_over:
                game_over_counter += 1
                # print("GAME OVER!!!")
                traj_mdp.append(tuple(next_mdp_state))
                # print(traj_mdp)
                # print(traj)
                traj.append("game over")


            if game_won:
                game_won_counter += 1
                traj_mdp.append(tuple(next_mdp_state))
                # print(traj_mdp)
                # print("GAME WON!!!")
                # print(MDP_obj.grid[:,i,j])
                print(traj_mdp)
                with open('traj_0506.pkl', 'wb') as f:
                    pickle.dump(traj_mdp, f)
                # print(PA_obj.DFA.accepting_states)
                # print(PA_obj.DFA.dfa_state)
                traj.append("game won")


            old_mdp_state, old_dfa_state = copy.deepcopy(next_mdp_state), next_dfa_state
            old_DFA_GT_state = next_DFA_GT_state
        # THIS IS THE END OF ONE trajectory


        if not terminal and traj_steps >= traj_steps_limit:
            time_over_counter += 1
            # print("Time Over!!!")
            # print(traj_mdp)
            traj.append("time over")
        trajs.append(traj)

    success_ratio = game_won_counter/(game_won_counter+(game_over_counter+time_over_counter))
    lose_ratio = game_over_counter/(game_won_counter+(game_over_counter+time_over_counter))
    return success_ratio,lose_ratio

    with open(trajs_address + 'trajectories.pkl', 'wb') as f:
        pickle.dump(trajs, f)

def eval_game_performance_bits(init_vars, agent, num_episodes):
    device = torch.device(init_vars["device"])
    Q_theta_counter, reward_net_time = agent.calc_Q_theta(thresh=init_vars["thresh_Q_theta"]) # calculate Q function for the current parameters
    agent.calc_pi_theta() # Policy is defined using softmax from pi_theta


    MDP_obj, DFA_obj, PA_obj, DFA_GT, PA_GT = agent.MDP, agent.DFA, agent.PA, agent.DFA_GT, agent.PA_GT
    game_won_counter = 0
    game_over_counter = 0
    time_over_counter = 0
    traj_steps_limit = init_vars["time_limit"]
    trajs = []
    for traj_number in range(num_episodes):
        traj = []
        # print("\n\n\n\n### traj_number " + str(traj_number) + "###")
        old_mdp_state, old_dfa_state_bits_arr = PA_obj.reset()
        old_dfa_state = DFA_obj.convert_dfa_state_to_int(old_dfa_state_bits_arr)
        _, old_DFA_GT_state = PA_GT.reset()
        PA_GT.set_MDP_state(old_mdp_state[0], old_mdp_state[1])

        reward = 0
        terminal = False
        traj_steps = 0
        while not terminal and traj_steps < traj_steps_limit:
            traj_steps += 1
            # print ("current_traj_steps:",  traj_steps)
            action_idx, action = agent.select_action_soft_pi_theta() # cntr selects an action among permitable actions

            traj.append((reward, old_mdp_state, old_dfa_state, old_DFA_GT_state, action_idx))

            next_mdp_state, next_dfa_state_bits_arr, _, _ = PA_obj.step(action_idx)
            next_dfa_state = DFA_obj.convert_dfa_state_to_int(next_dfa_state_bits_arr)
            _, next_DFA_GT_state, reward, _ = PA_GT.step(action_idx)



            # print ("---------- action_probs: {}".format(action_probs))
            # print(str((state,action)) + "; ")

            i,j = next_mdp_state[0], next_mdp_state[1]
            if init_vars["verbose"]:
                pass
                # print ("state before action ----> " + "[" + str(mdp_state[0]) +", " +
                #         str(mdp_state[1]) + "]" )
                # print ("action ----> "  + action)
                # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                # print ("---------------------")
            game_over, game_won = PA_GT.is_terminal()
            terminal = game_over or game_won # terminal refers to next state

            if game_over:
                game_over_counter += 1
                # print("GAME OVER!!!")
                traj.append("game over")


            if game_won:
                game_won_counter += 1
                # print("GAME WON!!!")
                traj.append("game won")


            old_mdp_state, old_dfa_state = copy.deepcopy(next_mdp_state), next_dfa_state
            old_DFA_GT_state = next_DFA_GT_state
        # THIS IS THE END OF ONE trajectory


        if not terminal and traj_steps >= traj_steps_limit:
            time_over_counter += 1
            # print("Time Over!!!")
            traj.append("time over")
        trajs.append(traj)

    success_ratio = game_won_counter/(game_won_counter+(game_over_counter+time_over_counter))
    return success_ratio

    # with open(trajs_address + 'trajectories.pkl', 'wb') as f:
    #     pickle.dump(trajs, f)


# MAIN FUNCTION
# ---------------------------
def main(args):

    init_vars = init(args)
    if init_vars["main_function"] == "train" :
        train(init_vars)
    elif init_vars["main_function"] == "train_IRL_active" :
        train_IRL_active(init_vars)
    elif init_vars["main_function"] == "train_from_inferred_DFA" :
        train_from_inferred_DFA(init_vars)
    elif init_vars["main_function"] == "train_base_memoryless" :
        train_base_memoryless(init_vars)
    elif init_vars["main_function"] == "train_IRL_bits" :
        train_IRL_bits(init_vars)
    elif init_vars["main_function"] == "train_incomp" :
        train_incomp(init_vars)
    elif init_vars["main_function"] in ["train_BC_v1", "train_BC_v2", "train_BC_local"] :
        train_BC(init_vars)
    elif init_vars["main_function"] == "train_BC_3" :
        train_BC_3(init_vars)
    elif init_vars["main_function"] == "train_base" :
        train_base(init_vars)
    elif init_vars["main_function"] == "train_non_local_r" :
        train_non_local_r(init_vars)
    elif init_vars["main_function"] == "test" :
        test(init_vars)
    elif init_vars["main_function"] == "test_active" :
        test_active(init_vars)
    elif init_vars["main_function"] == "test_bits" :
        test_bits(init_vars)
    elif init_vars["main_function"] == "test_memoryless_baseline" :
        test_memoryless_baseline(init_vars)
    elif init_vars["main_function"] == "test_non_local_r" :
        test_non_local_r(init_vars)
    elif init_vars["main_function"] == "test_BC_3" :
        test_BC_3(init_vars)
    elif init_vars["main_function"] in ["test_BC_v1", "test_BC_v2", "test_BC_local"] :
        test_BC(init_vars)
    elif init_vars["main_function"] == "test_base" :
        test_base(init_vars)
    elif init_vars["main_function"] == "produce_opt_trajs" :
        produce_opt_trajs(init_vars)
    elif init_vars["main_function"] == "produce_opt_trajs_time_limited" :
        produce_opt_trajs_time_limited(init_vars)
    elif init_vars["main_function"] == "produce_opt_trajs_time_limited_bits" :
        produce_opt_trajs_time_limited_bits(init_vars)
    elif init_vars["main_function"] == "produce_non_opt_trajs" :
        produce_non_opt_trajs(init_vars)
    elif init_vars["main_function"] == "produce_non_opt_trajs_time_limited" :
        produce_non_opt_trajs_time_limited(init_vars)
    elif init_vars["main_function"] == "produce_non_opt_trajs_time_limited_no_fail" :
        produce_non_opt_trajs_time_limited_no_fail(init_vars)
    elif init_vars["main_function"] == "produce_grids" :
        produce_grids(init_vars)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch irl-side')
    # STRINGS
    parser.add_argument('--main_function', type=str,
                        choices=["produce_opt_trajs", "produce_opt_trajs_time_limited", "produce_opt_trajs_time_limited_bits", "produce_non_opt_trajs",
                        "produce_non_opt_trajs_time_limited", "produce_non_opt_trajs_time_limited_no_fail",
                        "train", "train_IRL_active", "train_from_inferred_DFA", "train_base_memoryless","train_IRL_bits", "train_BC_v1", "train_BC_v2", "train_BC_3", "train_BC_local",
                        "train_base", "train_non_local_r","train_incomp", "test", "test_active", "test_bits", "test_memoryless_baseline", "test_non_local_r","test_BC_3", "test_BC_v1",
                        "test_BC_v2", "test_BC_local", "test_base","produce_grids"])
    parser.add_argument('--fileName_tr_test', type=str)
    parser.add_argument('--fileName_opt_trajs', type=str)
    parser.add_argument('--fileName_grids', type=str)
    parser.add_argument('--test_type', type=str, choices=["train_grid", "single_test_grid", "multiple_test_grids"])
    parser.add_argument('--run_mode', type=str, choices=["restart","continue"], default="restart")
    parser.add_argument('--device', type=str, choices=["cpu","cuda","cuda:0","cuda:1","cuda:2","cuda:3"], default="cpu")
    parser.add_argument('--train_grid_init_type', type=str, choices=["init_det","init_random"])
    parser.add_argument('--obj_type', type=str, choices=["monotone","non_monotone", "non_monotone_2"])
    parser.add_argument('--policy_net', type=str, choices=["MLP", "CNN", "att"])
    # parser.add_argument('--reward_net', type=str, choices=["MLP", "CNN", "att"])
    parser.add_argument("--GT_dfa_address", type=str)
    parser.add_argument("--tr_dfa_address", type=str)
    parser.add_argument("--map_address", type=str)
    parser.add_argument("--traj_address", type=str)


    parser.add_argument('--save_model_interval', type=int, default=5000, metavar='N', help='(default: 200)')
    parser.add_argument('--num_trajs_used', type=int, metavar='N')
    parser.add_argument('--num_optimal_trajs', type=int, metavar='N')
    parser.add_argument('--num_grids_tr', type=int, metavar='N')
    parser.add_argument('--num_grids_test', type=int, metavar='N')
    parser.add_argument('--time_limit', type=int, metavar='N')
    parser.add_argument('--num_runs', type=int, metavar='N')

    parser.add_argument('--num_theta_updates', type=int)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lam_phi', type=float, default=0.0, metavar='N', help='(default: 0)')
    parser.add_argument('--lam_reg', type=float, default=0.0, metavar='N', help='(default: 0)')
    parser.add_argument('--gamma', type=float, metavar='N', help='(default: 0.95)')
    parser.add_argument('--gamma_tr', type=float, metavar='N', help='(default: 0.95)')
    parser.add_argument("--batch_size", type=int, default=128, metavar='N')
    parser.add_argument("--n_dim", type=int)
    parser.add_argument("--n_imp_objs", type=int)
    parser.add_argument("--n_obstacles", type=int)
    parser.add_argument("--reward_input_size", type=int)
    parser.add_argument("--policy_input_size", type=int)

    args = parser.parse_args()
    main(args)
