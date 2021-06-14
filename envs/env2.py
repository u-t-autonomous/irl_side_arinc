# AUTHOR: Farzan Memarian
import numpy as np
import math
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import sys, os
from pdb import set_trace

""" NOTES
- in steps, right now if the action is illegal, nothing happens and no reward is returned, this could be modified later
"""

# MDPs ---------
class MDP:

    def __init__(self,
        main_function,
        n_dim,
        n_imp_objs,
        n_obstacles,
        imp_obj_idxs_init_det,
        obstacle_idxs_det,
        obj_type,
        random_start,
        init_type,
        RGB,
        pad_element,
        device,
        ):

        # set device
        self.device = device
        self.main_function = main_function
        # creates a square gridworld
        self.n_dim = n_dim
        self.n_objs = int(n_dim/3)**2
        self.n_imp_objs = n_imp_objs
        self.n_obstacles = n_obstacles
        self.RGB = RGB
        self.pad_element = pad_element
        self.pad_width = 3

        # objs, gridworld and goals
        self.mdp_state = np.zeros(2,dtype=int)
        self.imp_obj_idxs_init_det = imp_obj_idxs_init_det
        self.obstacle_idxs_det = obstacle_idxs_det
        self.grid = None
        self.grid_padded = None

        self.imp_objs_init = None
        self.obstacles_init = None
        self.non_imp_objs_init = None

        self.imp_obj_idxs_all = None
        self.obstacle_idxs_all = None
        self.non_imp_obj_idxs_all = None

        self.obj_type = obj_type
        self.all_colors = None
        self.classes = None
        self.is_grid_created = False

        self.gridworld_finite_store = None
        self.random_start = random_start
        # if init_type == "init_finite_random":
        #     self.gridworld_finite_store = self._create_multiple_gridworlds()

        self.initialize_grid(init_type)
        self._random_start()

        # actions
        self.allowable_actions = {}
        self.allowable_action_idxs = {}
        self.all_actions = ['U','D','R','L']
        self._set_actions()

    def _set_actions(self):
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                if i != 0 and i != self.n_dim-1 and j != 0 and j != self.n_dim-1:
                    self.allowable_actions[(i,j)] = self.all_actions
                    self.allowable_action_idxs[(i,j)] = [0,1,2,3]

                elif i == 0 and j != 0 and j != self.n_dim-1:
                    self.allowable_actions[(i,j)] = ['D','R','L']
                    self.allowable_action_idxs[(i,j)] = [1,2,3]
                elif i == self.n_dim-1 and j != 0 and j != self.n_dim-1:
                    self.allowable_actions[(i,j)] = ['U','R','L']
                    self.allowable_action_idxs[(i,j)] = [0,2,3]
                elif j == 0 and i != 0 and i != self.n_dim-1:
                    self.allowable_actions[(i,j)] = ['U','D','R']
                    self.allowable_action_idxs[(i,j)] = [0,1,2]
                elif j == self.n_dim-1 and i != 0 and i != self.n_dim-1:
                    self.allowable_actions[(i,j)] = ['U','D','L']
                    self.allowable_action_idxs[(i,j)] = [0,1,3]
                elif i == 0 and j == 0:
                    self.allowable_actions[(i,j)] = ['D','R']
                    self.allowable_action_idxs[(i,j)] = [1,2]
                elif i == self.n_dim-1 and j == 0:
                    self.allowable_actions[(i,j)] = ['U','R']
                    self.allowable_action_idxs[(i,j)] = [0,2]
                elif j == self.n_dim-1 and i == 0:
                    self.allowable_actions[(i,j)] = ['D','L']
                    self.allowable_action_idxs[(i,j)] = [1,3]
                elif j == self.n_dim-1 and i == self.n_dim-1:
                    self.allowable_actions[(i,j)] = ['U','L']
                    self.allowable_action_idxs[(i,j)] = [0,3]

    def _create_individual_imp_obj(self,color1,color2,n1,n2,non_imp_colors):
        # create classes out of two colors
        if self.RGB:
            # RGB version
            depth = 3
        else:
            depth = 6
        obj = torch.zeros((depth,3,3),dtype=torch.float32)

        all_idxs = [0,1,2,3,4,5,6,7,8]
        chosen_idxs = np.random.choice(all_idxs, size=n1, replace=False)
        remaining_idxs = list(set(all_idxs) - set(chosen_idxs))
        for i in chosen_idxs:
            x_idx = int(np.floor(i/3))
            y_idx = i % 3
            obj[:,x_idx,y_idx] = color1
        chosen_idxs = np.random.choice(remaining_idxs, size=n2, replace=False)
        remaining_idxs = list(set(remaining_idxs) - set(chosen_idxs))
        for i in chosen_idxs:
            x_idx = int(np.floor(i/3))
            y_idx = i % 3
            obj[:,x_idx,y_idx] = color2
        for i in remaining_idxs:
            x_idx = int(np.floor(i/3))
            y_idx = i % 3
            obj[:,x_idx,y_idx] = non_imp_colors[np.random.choice(len(non_imp_colors))]
        return obj

    def _create_individual_monotone_imp_obj(self,color1):
        # create classes out of two colors
        if self.RGB:
            # RGB version
            depth = 3
        else:
            depth = 6
        obj = torch.zeros((depth,3,3),dtype=torch.float32)
        all_idxs = [0,1,2,3,4,5,6,7,8]
        for i in all_idxs:
            x_idx = int(np.floor(i/3))
            y_idx = i % 3
            obj[:,x_idx,y_idx] = color1
        return obj

    def _create_individual_distractor_obj(self,all_colors=True):
        if self.RGB:
            depth = 3
        else:
            depth = 6
        obj = torch.zeros((depth,3,3),dtype=torch.float32)

        all_idxs = [0,1,2,3,4,5,6,7,8]
        for i in range(3):
            for j in range(3):
                # SHOULD BE MODIFIED 1234567891011
                # obj[:,i,j] = all_colors[np.random.choice(len(all_colors))]
                obj[:,i,j] = all_colors[-1] # all colors gray for now
        return obj

    def _create_individual_imp_obj_2(self, colors, all_colors):
        # create classes
        # does not use color black in the classes as this color is the obstacle
        if self.RGB:
            # RGB version
            depth = 3
        else:
            depth = 6
        obj = torch.zeros((depth,3,3),dtype=torch.float32)
        black_idx = len(colors)-1
        for idx, item in enumerate(colors):
            if item != -1:
                n1 = item
                color_idx = idx
                color1 = copy.deepcopy(all_colors[color_idx])
        remaining_color_idxs = list(set(np.arange(len(all_colors))) - set([color_idx]) - set([black_idx]))
        all_idxs = [0,1,2,3,4,5,6,7,8]
        chosen_idxs = np.random.choice(all_idxs, size=n1, replace=False)
        remaining_idxs = list(set(all_idxs) - set(chosen_idxs))
        for i in chosen_idxs:
            x_idx = int(np.floor(i/3))
            y_idx = i % 3
            obj[:,x_idx,y_idx] = color1
        for i in remaining_idxs:
            x_idx = int(np.floor(i/3))
            y_idx = i % 3
            obj[:,x_idx,y_idx] = all_colors[np.random.choice(remaining_color_idxs)]
        return obj

    def _create_objs(self):

        if self.RGB:
            red   = torch.from_numpy(np.array([255,0,0]))
            green = torch.from_numpy(np.array([0,255,0]))
            blue  = torch.from_numpy(np.array([0,0,255]))
            white = torch.from_numpy(np.array([255,255,255]))
            black  = torch.from_numpy(np.array([128,128,128]))
        else:
            red   = torch.from_numpy(np.array([1,0,0,0,0,0])).type(torch.float32)
            green = torch.from_numpy(np.array([0,1,0,0,0,0])).type(torch.float32)
            blue  = torch.from_numpy(np.array([0,0,1,0,0,0])).type(torch.float32)
            white = torch.from_numpy(np.array([0,0,0,1,0,0])).type(torch.float32)
            black = torch.from_numpy(np.array([0,0,0,0,1,0])).type(torch.float32)


        all_colors = [red, green, blue, white, black]
        classes = {}
        imp_objs = []
        obstacles = []
        distractor_objs = []


        # if self.obj_type == "non_monotone":
        #     # create true task objs
        #     # --------------------------------------
        #     # Class 1
        #     classes["cl1"] = [5,-1,-1,4,-1]
        #     non_imp_colors = [green, blue, black]
        #     color1 = red
        #     color2 = white
        #     n1 = 5
        #     n2 = 4
        #     obj = self._create_individual_imp_obj(color1,color2,n1,n2,non_imp_colors)
        #     imp_objs.append(copy.deepcopy(obj))

        #     # Class 2
        #     classes["cl2"] = [-1,5,-1,4,-1]
        #     non_imp_colors = [red, blue, black]
        #     color1 = green
        #     color2 = white
        #     n1 = 5
        #     n2 = 4
        #     obj = self._create_individual_imp_obj(color1,color2,n1,n2,non_imp_colors)
        #     imp_objs.append(copy.deepcopy(obj))

        #     # Class 3
        #     classes["cl3"] = [-1,-1,5,4,-1]
        #     non_imp_colors = [red, green, black]
        #     color1 = blue
        #     color2 = whitep
        #     n1 = 5
        #     n2 = 4
        #     obj = self._create_individual_imp_obj(color1,color2,n1,n2,non_imp_colors)
        #     imp_objs.append(copy.deepcopy(obj))


        #     # --------------------------------------

        # elif self.obj_type == "monotone":

        #     # Class 1
        #     classes["cl1"] = [9,-1,-1,-1,-1]
        #     color1 = red
        #     obj = self._create_individual_monotone_imp_obj(color1)
        #     imp_objs.append(copy.deepcopy(obj))

        #     # Class 2
        #     classes["cl2"] = [-1,9,-1,-1,-1]
        #     color1 = green
        #     obj = self._create_individual_monotone_imp_obj(color1)
        #     imp_objs.append(copy.deepcopy(obj))

        #     # Class 3
        #     classes["cl3"] = [-1,-1,9,-1,-1]
        #     color1 = blue
        #     obj = self._create_individual_monotone_imp_obj(color1)
        #     imp_objs.append(copy.deepcopy(obj))

        if self.obj_type == "non_monotone_2":
            # create true task objs
            # --------------------------------------
            # Class 1
            classes["cl1"] = [6,-1,-1,-1,-1]
            obj = self._create_individual_imp_obj_2(classes["cl1"], all_colors)
            imp_objs.append(copy.deepcopy(obj))

            # Class 2
            classes["cl2"] = [-1,6,-1,-1,-1]
            obj = self._create_individual_imp_obj_2(classes["cl2"], all_colors)
            imp_objs.append(copy.deepcopy(obj))

            # Class 3
            classes["cl3"] = [-1,-1,6,-1,-1]
            obj = self._create_individual_imp_obj_2(classes["cl3"], all_colors)
            imp_objs.append(copy.deepcopy(obj))

            # Class 3
            classes["cl4"] = [-1,-1,-1,6,-1]
            obj = self._create_individual_imp_obj_2(classes["cl4"], all_colors)
            imp_objs.append(copy.deepcopy(obj))

        elif self.obj_type == "monotone":

            # Class 1
            classes["cl1"] = [9,-1,-1,-1,-1]
            color1 = red
            obj = self._create_individual_monotone_imp_obj(color1)
            imp_objs.append(copy.deepcopy(obj))

            # Class 2
            classes["cl2"] = [-1,9,-1,-1,-1]
            color1 = green
            obj = self._create_individual_monotone_imp_obj(color1)
            imp_objs.append(copy.deepcopy(obj))

            # Class 3
            classes["cl3"] = [-1,-1,9,-1,-1]
            color1 = blue
            obj = self._create_individual_monotone_imp_obj(color1)
            imp_objs.append(copy.deepcopy(obj))

        return imp_objs, obstacles, distractor_objs, all_colors, classes

    def _place_objs(self,random):
        '''
        -places the objs
        -store the obj's indexes
        '''
        n_obj_per_row = int(self.n_dim/3)
        n_obj_per_col = int(self.n_dim/3)
        all_obj_lin_idx = np.arange(self.n_objs)
        import itertools
        all_idxs = list(itertools.product(np.arange(self.n_dim), np.arange(self.n_dim)))
        if random:
            imp_obj_idxs = []
            imp_obj_lin_idxs = np.random.choice(self.n_objs, size=self.n_imp_objs, replace=False)

            # get the index of the center of the important objects
            remaining_idxs = set(copy.deepcopy(all_idxs))
            for item in imp_obj_lin_idxs:
                x_idx = int(np.floor(item/n_obj_per_row))*3+1
                y_idx = (item%n_obj_per_row)*3 +1
                imp_obj_idxs.append((x_idx, y_idx))
                remove_idxs = set(itertools.product(np.arange(x_idx-1,x_idx+2), np.arange(y_idx-1,y_idx+2)))
                remaining_idxs = list(set(remaining_idxs) - remove_idxs)
            obstacle_idxs_temp = np.random.choice(len(remaining_idxs), size=self.n_obstacles, replace=False)
            obstacle_idxs = [remaining_idxs[idx] for idx in obstacle_idxs_temp]
            remaining_idxs = list(set(remaining_idxs) - set(obstacle_idxs))
        else:
            # use the provided self.imp_obj_idxs_init_detand self.obstacle_idxs_det
            if self.n_dim == 12:
                imp_obj_idxs = self.imp_obj_idxs_init_det[1]
                obstacle_idxs = self.obstacle_idxs_det[1]
            elif self.n_dim == 9:
                imp_obj_idxs = self.imp_obj_idxs_init_det[0]
                obstacle_idxs = self.obstacle_idxs_det[0]

            # removing the indexes of obstacles and all important cells from the set of all idxs
            remaining_idxs = set(all_idxs)
            for item in imp_obj_idxs:
                x_idx, y_idx = item
                # remove_idxs = set(itertools.product(np.arange(x_idx-1,x_idx+2), np.arange(y_idx-1,y_idx+2)))
                # remaining_idxs = remaining_idxs - remove_idxs
                remaining_idxs = remaining_idxs - set(imp_obj_idxs)
            remaining_idxs = remaining_idxs - set(obstacle_idxs)

        # non_imp_idxs = list(set(all_idx) - set(imp_obj_idxs) - set(obstacle_idxs))
        for counter, item in enumerate(imp_obj_idxs):
            x_idx, y_idx = item
            # imp_obj_idxs.append((x_idx+1,y_idx+1))
            # self.grid[:,x_idx-1:x_idx+2,y_idx-1:y_idx+2] = self.imp_objs_init[counter]
            self.grid[:,x_idx,y_idx] = self.all_colors[0]

        for counter, item in enumerate(obstacle_idxs):
            x_idx, y_idx = item
            # obstacle_idxs.append((x_idx+1,y_idx+1))
            self.grid[:, x_idx, y_idx] = self.all_colors[-1]

        for counter, item in enumerate(remaining_idxs):
            x_idx, y_idx = item
            color_idx = np.random.choice(np.arange(len(self.all_colors)-1), size=1, replace=True)[0]
            color_idx = 3
                        # above line randomly chooses one color except black
            self.grid[:, x_idx, y_idx] =  self.all_colors[color_idx]

        road_idxs = [(0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4),
                    (4,0),(4,1),(4,2),(4,3),(4,5)]

        # It's activated for the road example
        for item in road_idxs:
            x_idx, y_idx = item
            self.grid[:, x_idx, y_idx] =  self.all_colors[color_idx]

        for counter, item in enumerate(obstacle_idxs):
            x_idx, y_idx = item
            # obstacle_idxs.append((x_idx+1,y_idx+1))
            self.grid[:, x_idx, y_idx] = self.all_colors[-1]

        # after all objects are placed in the grid, place them in the grid_padded as well,
        # all the neighborhoods will be taken from grid_padded
        self.grid_padded[:,self.pad_width:-self.pad_width,self.pad_width:-self.pad_width] = self.grid
        imp_obj_idxs = []
        obstacle_idxs = []
        non_imp_obj_idxs = []
        cells_set = set()
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                neigh = self.neigh_select(i,j)
                match, cl, class_attribution_error = self.temp_match_single(neigh)

                if class_attribution_error == True:
                    return _,_,_,True
                elif match:
                    if cl == "cl5":
                        obstacle_idxs.append((i,j))
                    else:
                        imp_obj_idxs.append((i,j))
                else:
                    non_imp_obj_idxs.append((i,j))

        return imp_obj_idxs, obstacle_idxs, non_imp_obj_idxs, False

    def _create_multiple_gridworlds(self):
        pass
        # 1234567891011 SHOULD BE MODIFIED
        # state_list = []
        # for _ in range(self.num_gridworlds):
        #     if self.RGB:
        #         self.grid = torch.zeros((3, self.n_dim, self.n_dim),dtype=torch.float32, device=self.device)
        #     else:
        #         self.grid = torch.zeros((1, self.n_dim, self.n_dim),dtype=torch.float32, device=self.device)
        #     self.imp_objs_init, self.obstacles_init, self.non_imp_objs_init, self.all_colors, self.classes = self._create_objs()
        #     self.imp_obj_idxs_all, self.non_imp_obj_idxs_all = self.place_objs()
        #     state_list.append([copy.deepcopy(self.grid), copy.deepcopy(self.imp_objs_init),
        #         copy.deepcopy(self.imp_obj_idxs_all)])
        # return state_list

    def initialize_grid(self, init_type):
        if self.main_function == "produce_grids" :

            if self.RGB:
                depth = 3
            else:
                depth = 6

            self.grid = torch.zeros((depth, self.n_dim, self.n_dim),dtype=torch.float32, device=self.device)
            self.grid_padded = torch.zeros((depth, self.n_dim+2*self.pad_width, self.n_dim+2*self.pad_width),
                dtype=torch.float32, device=self.device)

            self.grid_padded[-1,:self.pad_width,:]  = 1
            self.grid_padded[-1,-self.pad_width:,:] = 1
            self.grid_padded[-1,:,:self.pad_width]  = 1
            self.grid_padded[-1,:,-self.pad_width:] = 1
            (self.imp_objs_init, self.obstacles_init, self.non_imp_objs_init,
                self.all_colors, self.classes) = self._create_objs()

            if init_type == "init_det":
                # objects are placed according to the predetermined indexes
                random = False
            elif init_type == "init_random":
                # objects are placed randomly
                random = True

            class_attribution_error = True
            while class_attribution_error == True:
                (self.imp_obj_idxs_all, self.obstacle_idxs_all, self.non_imp_obj_idxs_all,
                    class_attribution_error) = self._place_objs(random=random)

                # sample = random.sample(self.gridworld_finite_store,1)[0]
                # if self.RGB:
            #     self.grid = torch.zeros((3, self.n_dim, self.n_dim),dtype=torch.float32, device=self.device)
            # else:
            #     self.grid = torch.zeros((1, self.n_dim, self.n_dim),dtype=torch.float32, device=self.device)
            # self.imp_objs_init, self.non_imp_objs_init = copy.deepcopy(sample[1])
            # self.imp_obj_idxs_all = copy.deepcopy(sample[2])
        else:
            pass
            # with open(self.trajs_address +'grid_info.pkl', 'rb') as f:
            #     grid_info = pickle.load(f)

            # self.grid = grid_info[0]
            # self.imp_objs_init = grid_info[1]
            # self.obstacles_init = grid_info[2]
            # self.non_imp_objs_init = grid_info[3]

            # self.imp_obj_idxs_all = grid_info[4]
            # self.obstacle_idxs_all = grid_info[5]
            # self.non_imp_obj_idxs_all = grid_info[6]

            # self.all_colors = grid_info[7]
            # self.classes = grid_info[8]

        # self.imp_objs_init = copy.deepcopy(self.imp_objs_init)
        # self.obstacles_init = copy.deepcopy(self.obstacles_init)
        # self.imp_obj_idxs_all = copy.deepcopy(self.imp_obj_idxs_all)
        # self.obstacle_idxs_all = copy.deepcopy(self.obstacle_idxs_all)

    def initialize_agent(self):
        # this function restarts the state and also returns the coordinates of the initialized state
        # and the corresponding neighb
        if self.random_start:
            self.mdp_state[:] = self._random_start()
        else:
            self.mdp_state[:] = self._deterministic_start()

        neigh = self.neigh_select(self.mdp_state[0], self.mdp_state[1])

        return copy.deepcopy(self.mdp_state), copy.deepcopy(neigh)

    def _random_start(self):
        # in random_start, all mdp states are equally probably except the centers of the important
        # objs which are excluded
        start = np.zeros(2,dtype=int)
        i, j = random.choice(self.non_imp_obj_idxs_all)
        # print(self.non_imp_obj_idxs_all)
        # i, j = tuple(np.random.choice(3,(1,2))[0])
        # i = np.random.choice([0,1,2],1)[0]
        # j = np.random.choice([6,7,8],1)[0]
        start[0] = i
        start[1] = j
        return start

    def _deterministic_start(self):
        start = np.zeros(2,dtype=int)
        start[0] = 0
        start[1] = 0
        return start

    def neigh_select(self, i, j):
        # this function always return a 3*3 neighborhood to be used for class matching
        width = 1
        depth,n,m = self.grid.size()
        # wall = torch.from_numpy(np.array([0,0,0,0,0,1])).type(torch.float32)
        ip = i + self.pad_width
        jp = j + self.pad_width
        neigh = torch.zeros((depth,3,3), device=self.device)
        neigh[:,:,:] = self.grid_padded[:, ip-width :ip+width+1, jp-width:jp+width+1]
        return copy.deepcopy(neigh)

    def neigh_select_reward(self, i, j, neigh_size):
        width = int((neigh_size - 1) / 2)
        depth,n,m = self.grid.size()
        # wall = torch.from_numpy(np.array([0,0,0,0,0,1])).type(torch.float32)
        ip = i + self.pad_width
        jp = j + self.pad_width
        neigh = torch.zeros((depth,neigh_size,neigh_size), device=self.device)
        neigh[:,:,:] = self.grid_padded[:, ip-width :ip+width+1, jp-width:jp+width+1]


        # if 0 < i < self.n_dim-1 and 0 < j < self.n_dim-1:
        #     neigh = torch.zeros((depth,3,3), device=self.device)
        #     neigh[:,:,:] = self.grid[:, i-1:i+2, j-1:j+2]

        # elif i == 0 and 0 < j < self.n_dim-1:
        #     neigh = torch.zeros((depth,3,3), device=self.device)
        #     neigh[-1,0,:] = 1
        #     neigh[:,1:,:] = self.grid[:, :2, j-1:j+2]

        # elif i == self.n_dim-1 and 0 < j < self.n_dim-1:
        #     neigh = torch.zeros((depth,3,3), device=self.device)
        #     neigh[-1,2,:] = 1
        #     neigh[:,:2,:] = self.grid[:, -2:, j-1:j+2]

        # elif j == 0 and 0 < i < self.n_dim-1:
        #     neigh = torch.zeros((depth,3,3), device=self.device)
        #     neigh[-1,:,0] =1
        #     neigh[:,:,1:] = self.grid[:, i-1:i+2, :2]

        # elif j == self.n_dim-1 and 0 < i < self.n_dim-1:
        #     neigh = torch.zeros((depth,3,3), device=self.device)
        #     neigh[-1,:,2] = 1
        #     neigh[:,:,:2] = self.grid[:, i-1:i+2, -2:]

        # elif i == 0 and j == 0:
        #     neigh = torch.zeros((depth,3,3), device=self.device)
        #     neigh[-1,:,:] = 1
        #     neigh[:,1:,1:] = self.grid[:, :2, :2]

        # elif i == self.n_dim-1 and j == 0:
        #     neigh = torch.zeros((depth,3,3), device=self.device)
        #     neigh[-1,:,:] = 1
        #     neigh[:,:2,1:] = self.grid[:, -2:, :2]

        # elif j == self.n_dim-1 and i == 0:
        #     neigh = torch.zeros((depth,3,3), device=self.device)
        #     neigh[-1,:,:] = 1
        #     neigh[:,1:,:2] = self.grid[:, :2, -2:]

        # elif j == self.n_dim-1 and i == self.n_dim-1:
        #     neigh = torch.zeros((depth,3,3), device=self.device)
        #     neigh[-1,:,:] = 1
        #     neigh[:,:2,:2] = self.grid[:,-2:,-2:]

        return copy.deepcopy(neigh)

    def step(self, action_idx):
        i = self.mdp_state[0]
        j = self.mdp_state[1]
        next_state, next_neigh =  self.calc_next_state(i,j,action_idx)
        self.mdp_state[0] = next_state[0]
        self.mdp_state[1] = next_state[1]
        return copy.deepcopy(next_state), copy.deepcopy(next_neigh)

    def calc_next_state(self, i,j, action_idx):
        action = self.all_actions[action_idx]
        # check if legal move first, if not, nothing happens!
        if action in self.allowable_actions[(i,j)]:
            if   action == 'U':
                i += -1
            elif action == 'L':
                j += -1
            elif action == 'D':
                i += 1
            elif action == 'R':
                j += 1
        next_neigh = self.neigh_select(i,j)
        next_state = np.zeros(2,dtype=int)
        next_state[0] = i
        next_state[1] = j
        return copy.deepcopy(next_state), copy.deepcopy(next_neigh)

    def temp_match(self, current_neigh):
        """
        TO BE COMPLETED 1234567891011

        params:
        - current_neigh: this comes from the MDP
        - the function can access and use the current state of the DFA so no need
        to pass this directly as input parameter

        output:
        - match: whether any match has been found
        - cl: the class to which the current neigh is matched
        """


        all_colors = self.all_colors

        r_c = 0
        g_c = 0
        b_c = 0
        w_c = 0
        bl_c = 0
        for i in range(3):
            for j in range(3):
                if torch.all(torch.eq(current_neigh[:,i,j], all_colors[0])):
                    r_c+=1
                elif torch.all(torch.eq(current_neigh[:,i,j], all_colors[1])):
                    g_c+=1
                elif torch.all(torch.eq(current_neigh[:,i,j], all_colors[2])):
                    b_c+=1
                elif torch.all(torch.eq(current_neigh[:,i,j], all_colors[3])):
                    w_c+=1
                elif torch.all(torch.eq(current_neigh[:,i,j], all_colors[4])):
                    bl_c+=1
                else:
                    pass

        current_counts = [r_c,g_c,b_c,w_c,bl_c]

        matches = []
        if torch.all(torch.eq(current_neigh[:,1,1], all_colors[-1])):
            # if the center is black, the cell is an obstacle even if the count of colors
            # matches one of the classes
            match = True
            matches.append("cl5")
        else:
            for cl in self.classes:
                match = True
                for current_count, true_color_count in zip(current_counts,self.classes[cl]):
                    if (true_color_count != -1) and (true_color_count != current_count):
                        match = False
                if match:
                    matches.append(cl)



        class_attribution_error = False
        if len(matches) == 1:
            match = True
            cl = matches[0]
        elif len(matches) >= 1:
            match = True
            cl = matches
            class_attribution_error = True
        elif len(matches) == 0:
            match = False
            cl = None
        return match, cl, class_attribution_error

    def temp_match_single(self, current_neigh):
        """
        TO BE COMPLETED 1234567891011

        params:
        - current_neigh: this comes from the MDP
        - the function can access and use the current state of the DFA so no need
        to pass this directly as input parameter

        output:
        - match: whether any match has been found
        - cl: the class to which the current neigh is matched
        """

        all_colors = self.all_colors

        if torch.all(torch.eq(current_neigh[:,1,1], all_colors[0])):
            match = True
            cl = "cl1"
        elif torch.all(torch.eq(current_neigh[:,1,1], all_colors[1])):
            match = False
            cl = "cl2"
        elif torch.all(torch.eq(current_neigh[:,1,1], all_colors[2])):
            match = False
            cl = "cl3"
        elif torch.all(torch.eq(current_neigh[:,1,1], all_colors[3])):
            match = False
            cl = "cl4"
        elif torch.all(torch.eq(current_neigh[:,1,1], all_colors[-1])):
            match = True
            cl = "cl5"
        else:
            pass

        class_attribution_error = False
        return match, cl, class_attribution_error

    def temp_match_old(self, current_neigh):
        """
        TO BE COMPLETED 1234567891011

        params:
        - current_neigh: this comes from the MDP
        - the function can access and use the current state of the DFA so no need
        to pass this directly as input parameter

        output:
        - match: whether any match has been found
        - cl: the class to which the current neigh is matched
        """


        all_colors = self.all_colors

        r_c = 0
        g_c = 0
        b_c = 0
        w_c = 0
        gr_c = 0
        for i in range(3):
            for j in range(3):
                if torch.all(torch.eq(current_neigh[:,i,j], all_colors[0])):
                    r_c+=1
                elif torch.all(torch.eq(current_neigh[:,i,j], all_colors[1])):
                    g_c+=1
                elif torch.all(torch.eq(current_neigh[:,i,j], all_colors[2])):
                    b_c+=1
                elif torch.all(torch.eq(current_neigh[:,i,j], all_colors[3])):
                    w_c+=1
                elif torch.all(torch.eq(current_neigh[:,i,j], all_colors[4])):
                    gr_c+=1
                else:
                    pass

        current_counts = [r_c,g_c,b_c,w_c,gr_c]

        matches = []
        for cl in self.classes:
            match = True
            for current_count, true_color_count in zip(current_counts,self.classes[cl]):
                if (true_color_count != -1) and (true_color_count != current_count):
                    match = False
            if match:
                matches.append(cl)

        class_attribution_error = False
        if len(matches) == 1:
            match = True
            cl = matches[0]
        elif len(matches) >= 1:
            match = True
            cl = matches
            class_attribution_error = True
        elif len(matches) == 0:
            match = False
            cl = None

        return match, cl, class_attribution_error

    def get_to_obj(self, obj):
        pass
        # i, j = [self.imp_obj_idxs_all[item] for  item,obj in enumerate(self.imp_objs_init) if obj == obj][0]
        # self.mdp_state[0] = i
        # self.mdp_state[1] = j
        # return copy.deepcopy(self.mdp_state), copy.deepcopy(self.grid)

    def is_terminal(self, mdp_state):
        # it is terminal either when it's game over or it has solved the game
        pass
        # i = mdp_state[0]
        # j = mdp_state[1]
        # element = self.grid[i,j]
        # game_won = False
        # game_over = False

        # return game_over, game_won

    def set_state(self, i,j):
        self.mdp_state[:]

    def print_grid(self):
        print (self.grid)

    def current_mdp_state(self):
        return self.mdp_state


# DFAs ---------
class DFA_from_raw:
    """
    In this DFA class, we learn to build an DFA from raw representation
    In this class, we are assuming that there is no losing state, only accepting states
    """
    def __init__(self,
                MDP,
                RPNI_output_file_name,
                positive_reward = 1,
                negative_reward = -1,
                device = "cpu"
                ):

        self.MDP = MDP
        self.states = []
        self.alphabet = []
        self.transitions = {}
        self.accepting_states = []
        # self.losing_states = [1]
        self.init_states = []
        self._open_dfa(RPNI_output_file_name)
        self.failing_states = [1]
        self._find_failing_states()
        self.n_states = len(self.states)
        self.non_terminal_states = list(set(self.states)-set(self.accepting_states)-set(self.failing_states))

        self.dfa_state = 0 # this is the initial state
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.positive_reward = 20
        self.negative_reward = -20
        self.device = device

    def _find_failing_states(self):
        for dfa_state in self.states:
            failing = True
            for predicate in self.alphabet:
                new_state = self.transitions[(dfa_state, predicate)]
                if new_state != dfa_state:
                    failing = False
            if failing and dfa_state not in self.accepting_states:
                self.failing_states.append(dfa_state)

    def _open_dfa(self, RPNI_output_file_name):
        with open(RPNI_output_file_name) as RPNI_output_file:
            mode = "general"
            for line in RPNI_output_file:
                if "alphabet size" in line:
                    size = int(line.split("=")[1].strip().strip(';'))
                    self.alphabet = list(range(size))
                if "number of states" in line:
                    num_states = int(line.split("=")[1].strip().strip(';'))
                    self.states = list(range(num_states))
                if "initial states" in line:
                    mode = "init"
                    continue
                if "final states" in line:
                    mode = "final"
                    continue
                if "transitions" in line:
                    mode = "transitions"
                    continue

                if mode == "init":
                    line = line.strip().strip(';')
                    listOfStates = line.split(',')
                    self.init_states = [int(s) for s in listOfStates]
                    if len(self.init_states) > 1:
                        raise ValueError("the automaton has more than 1 initial state")

                if mode == "final":
                    line = line.strip().strip(';')
                    listOfStates = line.split(',')
                    self.accepting_states = list()
                    for s in listOfStates:
                        if s!= '':
                            self.accepting_states.append(int(s))
                    if self.accepting_states=='':
                        self.accepting_states.append(int(random.choice(range(0,51))))

                if mode == "transitions":
                    line = line.strip().strip(';')
                    transition_description = line.split(',')
                    self.transitions[(int(transition_description[0]), int(transition_description[1]))] = int(transition_description[2])


    def step(self, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state, reward = self.calc_next_S_and_R(copy.deepcopy(self.dfa_state), mdp_neigh)

        self.dfa_state = new_state
        return new_state, reward

    def step_from_predicate(self, predicate):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state, reward = self.calc_next_S_and_R_from_predicate(copy.deepcopy(self.dfa_state), predicate)

        self.dfa_state = new_state
        return new_state, reward

    def calc_next_S_and_R(self, dfa_state, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state = dfa_state
        reward = 0

        #  if state is an accepting state ==> no transition from them and zero reward
        if dfa_state in self.accepting_states:
            pass

        else: # if current dfa_state is not accepting
            # match, cl, _ = self.MDP.temp_match(mdp_neigh) # checks to see if there is any match and returns that match
            match, cl, _ = self.MDP.temp_match_single(mdp_neigh)
            predicate = self.predicate_convert(cl)
            if match:
                if (dfa_state, predicate) in self.transitions:
                    new_state = self.transitions[(dfa_state, predicate)]
                    if new_state != dfa_state:
                        if new_state not in self.failing_states:
                            reward = self.positive_reward
                        else:
                            reward = self.negative_reward

        return new_state, reward

    def calc_next_S_and_R_from_predicate(self, dfa_state, predicate):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state = dfa_state
        reward = 0

        #  if state is an accepting state ==> no transition from them and zero reward
        if dfa_state in self.accepting_states:
            pass

        else: # if current dfa_state is not accepting
            if (dfa_state, predicate) in self.transitions:
                new_state = self.transitions[(dfa_state, predicate)]
                if new_state in self.accepting_states and new_state != dfa_state:
                # if new_state != dfa_state:
                    # print(f"old_state: {dfa_state}     new_state: {new_state}")
                    reward = self.positive_reward

        return new_state, reward

    def predicate_convert(self, cl):
        if cl == "cl1":
            predicate = 0
        # elif cl == "cl2":
        #     predicate = 1
        elif cl == "cl3":
            predicate = 2
        elif cl == "cl4":
            predicate = 3
        else:
            predicate = 1
        return predicate

    def reset(self):
        self.dfa_state = self.init_states[0]
        return copy.deepcopy(self.dfa_state)

class DFA_bits:
    """
    In this DFA class, we learn to build an DFA from raw representation
    In this class, we are assuming that there is no losing state, only accepting states
    """
    def __init__(self,
                MDP,
                positive_reward = 1,
                negative_reward = -1,
                device = "cpu"
                ):

        self.MDP = MDP
        self.states = []
        self.init_states = []
        self.accepting_states = []
        self.n_states = self.MDP.n_imp_objs ** 2
        self.dfa_state = np.zeros(self.MDP.n_imp_objs) # this is the initial state
        self.device = device

    def step(self, mdp_neigh):
        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        old_state = copy.deepcopy(self.dfa_state)
        match, cl, _ = self.MDP.temp_match(mdp_neigh) # checks to see if there is any match and returns that match
        reward = 0
        if match:
            predicate = self.predicate_convert(cl)
            if predicate == -1:
                set_trace()
            new_state, reward = self._calc_next_S_and_R_from_predicate(copy.deepcopy(self.dfa_state), predicate)
            self.dfa_state[:] = new_state[:]
        else:
            new_state = old_state


        return new_state, reward

    def calc_next_S_and_R(self, dfa_state, mdp_neigh):
        reward = 0
        match, cl, _ = self.MDP.temp_match(mdp_neigh) # checks to see if there is any match and returns that match
        predicate = self.predicate_convert(cl)
        next_state_arr, _ = self._calc_next_S_and_R_from_predicate(copy.deepcopy(self.dfa_state), predicate)
        return self.convert_dfa_state_to_int(next_state_arr), reward

    def _calc_next_S_and_R_from_predicate(self, dfa_state, predicate):
        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        dfa_state[predicate] = 1
        reward = 0
        # dfa_state_int = self.convert_dfa_state_to_int(dfa_state)
        #  if state is an accepting state ==> no transition from them and zero reward
        return dfa_state, reward

    def convert_dfa_state_to_int(self, dfa_state):
        return int(dfa_state[0] * 1 + dfa_state[1] * 2 + dfa_state[2] * 4 + dfa_state[3] * 8)

    def predicate_convert(self, cl):
        if cl == "cl1":
            predicate = 0
        elif cl == "cl2":
            predicate = 1
        elif cl == "cl3":
            predicate = 2
        elif cl == "cl4":
            predicate = 3
        else:
            predicate = -1
        return predicate

    def reset(self):
        self.dfa_state = np.zeros(self.MDP.n_imp_objs)
        return copy.deepcopy(self.dfa_state)

class DFA_v0:
    """
    the following are the DFA classes
    0 = q0 ---> non of the imp classes have been met
    1 = q1 ---> cl1 has been met
    2 = q2 ---> cl2 has been met after the DFA has been in q1
    4 = qf ---> failure state
    """
    def __init__(self,
                MDP,
                positive_reward = 1,
                negative_reward = -1,
                device = "cpu",
                ):
        self.MDP = MDP
        self.n_states = 5
        self.states = [0,1,2,3,4]
        self.dfa_state = 0
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.device = device

    def step(self, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state, reward = self.calc_next_S_and_R(copy.deepcopy(self.dfa_state), mdp_neigh)
        self.dfa_state = new_state
        return new_state, reward

    def calc_next_S_and_R(self, dfa_state, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state = dfa_state
        reward = 0

        # states 3 and 4 are abxorbing states ==> no transition from them and zero reward
        if dfa_state == 4 or dfa_state == 3:
            pass

        else: # if current dfa_state is anything other than 3 or 4
            match, cl, _ = self.MDP.temp_match(mdp_neigh) # checks to see if there is any match and returns that match
            if match:
                if cl == "cl4":
                    # whenever "cl4" is visited, DFA will transition to qf
                    new_state = 4
                    reward =  self.negative_reward

                elif dfa_state == 0:
                    if cl == "cl1":
                        # transition to q1
                        new_state = 1
                        reward =  self.positive_reward
                    elif cl == "cl2" or cl == "cl3":
                        # transition to qf
                        new_state = 4
                        reward =  self.negative_reward

                elif dfa_state == 1:
                    if cl == "cl1":
                        pass
                        # no transition
                    elif cl == "cl2":
                        # transition to q2
                        new_state = 2
                        reward =  self.positive_reward
                    elif cl == "cl3":
                        # transition to qf
                        new_state = 4
                        reward =  self.negative_reward

                elif dfa_state == 2:
                    if cl == "cl1" or cl == "cl2":
                        # no transition
                        pass
                    elif cl == "cl3":
                        # transition to q3
                        new_state = 3
                        reward =  self.positive_reward

        return new_state, reward


    def reset(self):
        self.dfa_state = 0
        return copy.deepcopy(self.dfa_state)

class DFA_v1:
    """
    In this DFA, there are two ways to win the game, and there is a failure state
    the following are the DFA classes
    0 = q0 ---> non of the imp classes have been met
    1 = q1 ---> pick C1
    2 = q2 ---> pick C2
    3 = q3 ---> pick C1 then C3 (picking C1 again is allowed)
    4 = q4 ---> pick C2 then C4 (picking C2 again is allowed)
    5 = q5 ---> pick (C1 then C3 then C2) or (C2 then C4 then C1)
    6 = q6 ---> losing state, reaching C5 at any time causes transition to this state.
                picking up other wrong objects causes transitioning to q6 as well.
    """
    def __init__(self,
                MDP,
                positive_reward = 1,
                negative_reward = -1,
                device = "cpu",
                ):
        self.MDP = MDP
        self.n_states = 7
        self.states = [0,1,2,3,4,5,6]
        self.dfa_state = 0
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.device = device

    def step(self, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state, reward = self.calc_next_S_and_R(copy.deepcopy(self.dfa_state), mdp_neigh)
        self.dfa_state = new_state
        return new_state, reward

    def calc_next_S_and_R(self, dfa_state, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state = dfa_state
        reward = 0

        # states 5 (winning state) and 6 (losing state) are absorbing states ==> no transition from them and zero reward
        if dfa_state == 5 or dfa_state == 6:
            pass

        else: # if current dfa_state is anything other than 5 or 6
            match, cl, _ = self.MDP.temp_match(mdp_neigh) # checks to see if there is any match and returns that match
            if match:
                if cl == "cl5":
                    # whenever "cl5" (obstacle) is visited, DFA will transition to qf
                    new_state = 6
                    reward =  self.negative_reward

                elif dfa_state == 0:
                    if cl == "cl1":
                        # transition to q1
                        new_state = 1
                        reward =  self.positive_reward
                    elif cl == "cl2":
                        # transition to q1
                        new_state = 2
                        reward =  self.positive_reward
                    elif cl == "cl3" or cl == "cl4":
                        # transition to qf
                        new_state = 6
                        reward =  self.negative_reward

                elif dfa_state == 1:
                    if cl == "cl1":
                        pass
                        # no transition
                    elif cl == "cl3":
                        # transition to q2
                        new_state = 3
                        reward =  self.positive_reward
                    elif cl == "cl2" or cl == "cl4":
                        # transition to qf
                        new_state = 6
                        reward =  self.negative_reward

                elif dfa_state == 2:
                    if cl == "cl2":
                        pass
                        # no transition
                    elif cl == "cl4":
                        # transition to q2
                        new_state = 4
                        reward =  self.positive_reward
                    elif cl == "cl1" or cl == "cl3":
                        # transition to qf
                        new_state = 6
                        reward =  self.negative_reward

                elif dfa_state == 3:
                    if cl == "cl3":
                        pass
                        # no transition
                    elif cl == "cl2":
                        # transition to q2
                        new_state = 5
                        reward =  self.positive_reward
                    elif cl == "cl4":
                        # transition to qf
                        new_state = 6
                        reward =  self.negative_reward

                elif dfa_state == 4:
                    if cl == "cl4":
                        pass
                        # no transition
                    elif cl == "cl1":
                        # transition to q2
                        new_state = 5
                        reward =  self.positive_reward
                    elif cl == "cl3":
                        # transition to qf
                        new_state = 6
                        reward =  self.negative_reward
        return new_state, reward

    def reset(self):
        self.dfa_state = 0
        return copy.deepcopy(self.dfa_state)

class DFA_v2:
    """
    In this DFA, there are two ways to win the game, and there is no failure state
    the following are the DFA classes
    0 = q0 ---> non of the imp classes have been met
    1 = q1 ---> pick C1
    2 = q2 ---> pick C2
    3 = q3 ---> pick C1 then C3 (picking C1 again is allowed)
    4 = q4 ---> pick C2 then C4 (picking C2 again is allowed)
    5 = q5 ---> pick (C1 then C3 then C2) or (C2 then C4 then C1)
                picking up other wrong objects causes transitioning to q6 as well.
    """
    def __init__(self,
                MDP,
                positive_reward = 1,
                negative_reward = -1,
                device = "cpu",
                ):
        self.MDP = MDP
        self.n_states = 6
        self.states = [0,1,2,3,4,5]
        self.dfa_state = 0
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.device = device

    def step(self, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state, reward = self.calc_next_S_and_R(copy.deepcopy(self.dfa_state), mdp_neigh)
        self.dfa_state = new_state
        return new_state, reward

    def calc_next_S_and_R(self, dfa_state, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state = dfa_state
        reward = 0

        # states 5 (winning state)  ==> no transition from them and zero reward
        if dfa_state == 5:
            pass

        else: # if current dfa_state is anything other than 5 or 6
            match, cl, _ = self.MDP.temp_match(mdp_neigh) # checks to see if there is any match and returns that match
            if match:
                if cl == "cl5":
                    # whenever "cl5" is visited, nothing happens, cl5 is to be ignored
                    pass

                elif dfa_state == 0:
                    if cl == "cl1":
                        # transition to q1
                        new_state = 1
                        reward =  self.positive_reward
                    elif cl == "cl2":
                        # transition to q1
                        new_state = 2
                        reward =  self.positive_reward


                elif dfa_state == 1:

                    if cl == "cl3":
                        # transition to q2
                        new_state = 3
                        reward =  self.positive_reward


                elif dfa_state == 2:
                    if cl == "cl4":
                        # transition to q2
                        new_state = 4
                        reward =  self.positive_reward

                elif dfa_state == 3:
                    if cl == "cl2":
                        # transition to q2
                        new_state = 5
                        reward =  self.positive_reward

                elif dfa_state == 4:
                    if cl == "cl1":
                        # transition to q2
                        new_state = 5
                        reward =  self.positive_reward

        return new_state, reward

    def reset(self):
        self.dfa_state = 0
        return copy.deepcopy(self.dfa_state)

class DFA_infered_v1:
    """
    In this DFA, there are two ways to win the game, and there is no failure state
    the following are the DFA classes
    similar to original DFA, except:
    - there is no losing state
    - there is a transition from q3 to q2
    """
    def __init__(self,
                MDP,
                positive_reward = 1,
                negative_reward = -1,
                device = "cpu",
                ):
        self.MDP = MDP
        self.n_states = 6
        self.states = [0,1,2,3,4,5]
        self.dfa_state = 0
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.device = device

    def step(self, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state, reward = self.calc_next_S_and_R(copy.deepcopy(self.dfa_state), mdp_neigh)
        self.dfa_state = new_state
        return new_state, reward

    def calc_next_S_and_R(self, dfa_state, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state = dfa_state
        reward = 0

        # states 5 (winning state)  ==> no transition from them and zero reward
        if dfa_state == 5:
            pass

        else: # if current dfa_state is anything other than 5 or 6
            match, cl, _ = self.MDP.temp_match(mdp_neigh) # checks to see if there is any match and returns that match
            if match:

                if dfa_state == 0:
                    if cl == "cl1":
                        # transition to q1
                        new_state = 1
                        reward =  self.positive_reward
                    elif cl == "cl2":
                        # transition to q1
                        new_state = 2
                        reward =  self.positive_reward


                elif dfa_state == 1:

                    if cl == "cl3":
                        # transition to q2
                        new_state = 3
                        reward =  self.positive_reward


                elif dfa_state == 2:
                    if cl == "cl4":
                        # transition to q2
                        new_state = 4
                        reward =  self.positive_reward

                elif dfa_state == 3:
                    if cl == "cl2":
                        # transition to q2
                        new_state = 5
                        reward =  self.positive_reward
                    elif cl == "cl1":
                        new_state = 2
                        # no reward for this tansition

                elif dfa_state == 4:
                    if cl == "cl1":
                        # transition to q2
                        new_state = 5
                        reward =  self.positive_reward

        return new_state, reward

    def reset(self):
        self.dfa_state = 0
        return copy.deepcopy(self.dfa_state)

class DFA_incomp:
    """
    the following are the DFA classes
    0 = q0 ---> non of the imp classes have been met
    1 = q1 ---> cl1 has been met
    2 = q2 ---> cl2 has been met after the DFA has been in q1
    4 = qf ---> failure state
    """
    def __init__(self,
                MDP,
                positive_reward = 100,
                negative_reward = -100,
                device = "cpu",
                ):
        self.MDP = MDP
        self.n_states = 5
        self.states = [0,1,2,3,4]
        self.dfa_state = 0
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.device = device

    def step(self, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state, reward = self.calc_next_S_and_R(copy.deepcopy(self.dfa_state), mdp_neigh)
        self.dfa_state = new_state
        return new_state, reward

    def calc_next_S_and_R(self, dfa_state, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state = dfa_state
        reward = 0

        # states 3 and 4 are abxorbing states ==> no transition from them and zero reward
        if dfa_state == 4 or dfa_state == 3:
            pass

        else: # if current dfa_state is anything other than 3 or 4
            match, cl, _ = self.MDP.temp_match(mdp_neigh) # checks to see if there is any match and returns that match
            if match:
                if cl == "cl4":
                    # this is where the DFA is incomplete, instead of transitioning to q4, nothing happens
                    pass

                elif dfa_state == 0:
                    if cl == "cl1":
                        # transition to q1
                        new_state = 1
                        reward =  self.positive_reward
                    elif cl == "cl2" or cl == "cl3":
                        # transition to qf
                        new_state = 4
                        reward =  self.negative_reward

                elif dfa_state == 1:
                    if cl == "cl1":
                        pass
                        # no transition
                    elif cl == "cl2":
                        # transition to q2
                        new_state = 2
                        reward =  self.positive_reward
                    elif cl == "cl3":
                        # transition to qf
                        new_state = 4
                        reward =  self.negative_reward

                elif dfa_state == 2:
                    if cl == "cl1" or cl == "cl2":
                        # no transition
                        pass
                    elif cl == "cl3":
                        # transition to q3
                        new_state = 3
                        reward =  self.positive_reward
        return new_state, reward


    def reset(self):
        self.dfa_state = 0
        return copy.deepcopy(self.dfa_state)

class DFA_base:
    """
    the following are the DFA classes
    0 = q0 ---> non of the imp classes have been met
    1 = q1 ---> cl1 has been met
    2 = q2 ---> cl2 has been met after the DFA has been in q1
    4 = qf ---> failure state
    """
    def __init__(self,
                MDP,
                positive_reward = 100,
                negative_reward = -100,
                device = "cpu",
                ):
        self.MDP = MDP
        self.n_dfa_states = 1,
        self.dfa_states = [0],
        self.dfa_state = 0
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.device = device

    def step(self, mdp_neigh):
        return 0,0

    def calc_next_S_and_R(self, dfa_state, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        return 0,0


    def reset(self):
        self.dfa_state = 0
        return copy.deepcopy(self.dfa_state)

class DFA_from_raw_memoryless:
    """
    In this DFA class, we learn to build an DFA from raw representation
    In this class, we are assuming that there is no losing state, only accepting states
    """
    def __init__(self,
                MDP,
                RPNI_output_file_name,
                positive_reward = 1,
                negative_reward = -1,
                device = "cpu"
                ):

        self.MDP = MDP
        self.states = []
        self.alphabet = []
        self.transitions = {}
        self.accepting_states = []
        self.init_states = []
        self._open_dfa(RPNI_output_file_name)
        self.failing_states = []
        self._find_failing_states()
        self.n_states = len(self.states)
        self.non_terminal_states = list(set(self.states)-set(self.accepting_states)-set(self.failing_states))

        self.dfa_state = 0 # this is the initial state
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.device = device

    def _find_failing_states(self):
        for dfa_state in self.states:
            failing = True
            for predicate in self.alphabet:
                new_state = self.transitions[(dfa_state, predicate)]
                if new_state != dfa_state:
                    failing = False
            if failing and dfa_state not in self.accepting_states:
                self.failing_states.append(dfa_state)

    def _open_dfa(self, RPNI_output_file_name):
        with open(RPNI_output_file_name) as RPNI_output_file:
            mode = "general"
            for line in RPNI_output_file:
                if "alphabet size" in line:
                    size = int(line.split("=")[1].strip().strip(';'))
                    self.alphabet = list(range(size))
                if "number of states" in line:
                    num_states = int(line.split("=")[1].strip().strip(';'))
                    self.states = list(range(num_states))
                if "initial states" in line:
                    mode = "init"
                    continue
                if "final states" in line:
                    mode = "final"
                    continue
                if "transitions" in line:
                    mode = "transitions"
                    continue

                if mode == "init":
                    line = line.strip().strip(';')
                    listOfStates = line.split(',')
                    self.init_states = [int(s) for s in listOfStates]
                    if len(self.init_states) > 1:
                        raise ValueError("the automaton has more than 1 initial state")

                if mode == "final":
                    line = line.strip().strip(';')
                    listOfStates = line.split(',')
                    self.accepting_states = list()
                    for s in listOfStates:
                        if s!= '':
                            self.accepting_states.append(int(s))
                    if self.accepting_states=='':
                        self.accepting_states.append(int(random.choice(range(0,51))))

                if mode == "transitions":
                    line = line.strip().strip(';')
                    transition_description = line.split(',')
                    self.transitions[(int(transition_description[0]), int(transition_description[1]))] = int(transition_description[2])


    def step(self, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state, reward = self.calc_next_S_and_R(copy.deepcopy(self.dfa_state), mdp_neigh)

        self.dfa_state = new_state
        return new_state, reward

    def step_from_predicate(self, predicate):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state, reward = self.calc_next_S_and_R_from_predicate(copy.deepcopy(self.dfa_state), predicate)

        self.dfa_state = new_state
        return new_state, reward

    def calc_next_S_and_R(self, dfa_state, mdp_neigh):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state = dfa_state
        reward = 0

        #  if state is an accepting state ==> no transition from them and zero reward
        if dfa_state in self.accepting_states:
            pass

        else: # if current dfa_state is not accepting
            match, cl, _ = self.MDP.temp_match(mdp_neigh) # checks to see if there is any match and returns that match
            predicate = self.predicate_convert(cl)
            if match:
                if (dfa_state, predicate) in self.transitions:
                    new_state = self.transitions[(dfa_state, predicate)]
                    if new_state != dfa_state:
                        if new_state not in self.failing_states:
                            reward = self.positive_reward

        return dfa_state, reward

    def calc_next_S_and_R_from_predicate(self, dfa_state, predicate):

        # the assumption is that one neighborhood never matches more than one class
        # the implementation needs to ensure that this is not violated or else this
        # function won't work properly
        new_state = dfa_state
        reward = 0

        #  if state is an accepting state ==> no transition from them and zero reward
        if dfa_state in self.accepting_states:
            pass

        else: # if current dfa_state is not accepting
            if (dfa_state, predicate) in self.transitions:
                new_state = self.transitions[(dfa_state, predicate)]
                if new_state in self.accepting_states and new_state != dfa_state:
                # if new_state != dfa_state:
                    # print(f"old_state: {dfa_state}     new_state: {new_state}")
                    reward = self.positive_reward

        return dfa_state, reward

    def predicate_convert(self, cl):
        if cl == "cl1":
            predicate = 0
        elif cl == "cl2":
            predicate = 1
        elif cl == "cl3":
            predicate = 2
        elif cl == "cl4":
            predicate = 3
        else:
            predicate = -1
        return predicate

    def reset(self):
        self.dfa_state = self.init_states[0]
        return copy.deepcopy(self.dfa_state)


# PRODUCT AUTOMATAS ---------
class PA:
    # This class is the Product Automaton
    def __init__(self,MDP,DFA):
        self.MDP = MDP
        self.DFA = DFA
        self.transition_table = {}
        self._calc_transitions() # fill in self.transition_table

    def return_current_state(self):
        return copy.deepcopy(self.MDP.mdp_state), self.DFA.dfa_state

    def step(self,action_idx):
        next_mdp_state, next_neigh = self.MDP.step(action_idx)
        next_dfa_state, reward = self.DFA.step(next_neigh)
        return copy.deepcopy(next_mdp_state), copy.deepcopy(next_dfa_state), reward, copy.deepcopy(next_neigh)

    def step_DFA_only(self,action_idx):
        next_mdp_state, next_neigh = self.MDP.step(action_idx)
        next_dfa_state, reward = self.DFA.step(next_neigh)
        return copy.deepcopy(next_mdp_state), copy.deepcopy(next_dfa_state), reward, copy.deepcopy(next_neigh)

    def _calc_transitions(self):
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                for k in range(self.DFA.n_states):
                    for l in range(4): # for the 4 actions
                        next_mdp_state, next_mdp_neigh = self.MDP.calc_next_state(i,j,l)
                        next_dfa_state, reward = self.DFA.calc_next_S_and_R(k, next_mdp_neigh)
                        self.transition_table[(i,j,k,l)] = (next_mdp_state[0],next_mdp_state[1], next_dfa_state, reward)

    def reset(self):
        mdp_state, neigh = self.MDP.initialize_agent()
        dfa_state = self.DFA.reset()
        self.MDP.mdp_state = mdp_state
        self.DFA.dfa_state = dfa_state
        return copy.deepcopy(mdp_state), dfa_state

    def reset_DFA_only(self):
        dfa_state = self.DFA.reset()
        return dfa_state

    def is_terminal(self):
        game_won = False
        game_over = False
        if self.DFA.dfa_state in self.DFA.accepting_states:
            game_won = True
        try:
            if self.DFA.dfa_state in self.DFA.failing_states:
                game_over = True
        except:
            pass
        return game_over, game_won

    def set_MDP_state(self, i, j):
        self.MDP.mdp_state[0] = i
        self.MDP.mdp_state[1] = j

class PA_base:
    # This class is the Product Automaton
    def __init__(self,MDP,DFA):
        self.MDP = MDP
        self.DFA = DFA
        self.transition_table = {}
        self._calc_transitions() # fill in self.transition_table

    def return_current_state(self):
        return copy.deepcopy(self.MDP.mdp_state), self.DFA.dfa_state

    def step(self,action_idx):
        next_mdp_state, next_neigh = self.MDP.step(action_idx)
        next_dfa_state, reward = self.DFA.step(next_neigh)
        return copy.deepcopy(next_mdp_state), copy.deepcopy(next_dfa_state), reward, copy.deepcopy(next_neigh)

    def _calc_transitions(self):
        for i in range(self.MDP.n_dim):
            for j in range(self.MDP.n_dim):
                for k in range(self.DFA.n_states):
                    for l in range(4): # for the 4 actions
                        next_mdp_state, next_mdp_neigh = self.MDP.calc_next_state(i,j,l)
                        next_dfa_state, reward = self.DFA.calc_next_S_and_R(k, next_mdp_neigh)
                        self.transition_table[(i,j,k,l)] = (next_mdp_state[0],next_mdp_state[1],next_dfa_state, reward)

    def reset(self):
        mdp_state, neigh = self.MDP.initialize_agent()
        dfa_state = self.DFA.reset()
        self.MDP.mdp_state = mdp_state
        self.DFA.dfa_state = dfa_state
        return copy.deepcopy(mdp_state), dfa_state

    def is_terminal(self):
        game_won = False
        game_over = False
        if self.DFA.dfa_state == 3:
            game_won = True
        elif self.DFA.dfa_state == 4:
            game_over = True
        return game_over, game_won
