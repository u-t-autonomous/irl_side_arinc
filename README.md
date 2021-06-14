This is an implementation of deep IRL with side info

To run the code, go to src directory and run python main_irl.py. This function calls env.py from env directory and utils.py from utils directory and agent.py from agent directory.

For each run you need to fill in the corresponding parameters in main_irl.init()
	- first you need to choose if the run is for train, test or produce_optimal_trajectories
	- then depending on which one you chose, you need to fill in the corresponding parts in the following if-then cases

## Examples in the slide show

The constructed gridworld files can be found under the `grids` folder. For the following examples you don't need to reconstruct them.

### Door Example

In this example, we constructed 9x9 gridworlds as training and test environments. They contain a goal region and door-shaped varying obstacles in themselves. For the training, we start the all of the demonstrations from the upper-left corner of the environment. To do that you need to modify `_random_start` function in `envs/env2.py` file. It normally initializes trajectories randomly. You need to restrict the initialization region.

To train a reward function model using this gridworld example, you need to run the following command,
```
python3 src/main_irl2.py --main_function="train_from_inferred_DFA" --fileName_tr_test="filename_for_door_model" --fileName_grids="door" --fileName_opt_trajs="produce_opt_trajs_time_limited" --num_trajs_used=8 --num_optimal_trajs=10 --GT_dfa_address=./inferred_DFAs/arinc.txt --tr_dfa_address=./inferred_DFAs/arinc.txt --lr=0.003 --gamma=0.9 --reward_input_size=3 --num_runs=1 --num_theta_updates=100 --time_limit=1000 --lam_phi=100
```

To test the trained model use the following command,
```
python3 src/main_irl2.py --main_function="test" --fileName_tr_test="filename_for_door_model" --fileName_grids="door" --fileName_opt_trajs="produce_opt_trajs_time_limited"  --num_runs=1  --test_type="train_grid"
```
### Floodedground Example

In this example, we discretize a small portion of floodedground environment in Phoenix and feed it to our pipeline as a gridworld.

To train a reward function model using this gridworld example, you need to run the following command,
```
python3 src/main_irl2.py --main_function="train_from_inferred_DFA" --fileName_tr_test="filename_for_phoenix_model" --fileName_grids="phoenix" --fileName_opt_trajs="produce_opt_trajs_time_limited" --num_trajs_used=8 --num_optimal_trajs=10 --GT_dfa_address=./inferred_DFAs/arinc.txt --tr_dfa_address=./inferred_DFAs/arinc.txt --lr=0.003 --gamma=0.9 --reward_input_size=3 --num_runs=1 --num_theta_updates=100 --time_limit=1000 --lam_phi=100
```
To test the trained model use the following command,
```
python3 src/main_irl2.py --main_function="test" --fileName_tr_test="filename_for_phoenix_model" --fileName_grids="phoenix" --fileName_opt_trajs="produce_opt_trajs_time_limited"  --num_runs=1  --test_type="train_grid"
```

### Lejeune Example

In this example, we discretize a big portion of lejeune environment in Phoenix as a 50x50 gridworld and feed it to our pipeline.

To train a reward function model using this gridworld example, you need to run the following command,
```
python3 src/main_irl2.py --main_function="train_from_inferred_DFA" --fileName_tr_test="filename_for_lejeune_model" --fileName_grids="lejeune50" --fileName_opt_trajs="produce_opt_trajs_time_limited" --num_trajs_used=5 --num_optimal_trajs=5 --GT_dfa_address=./inferred_DFAs/arinc.txt --tr_dfa_address=./inferred_DFAs/arinc.txt --traj_address=./trajs --lr=0.003 --gamma=0.9 --reward_input_size=3 --num_runs=1 --num_theta_updates=5000 --time_limit=50 --lam_phi=0
```
To test the trained model use the following command,
```
python3 src/main_irl2.py --main_function="test" --fileName_tr_test="filename_for_lejeune_model" --fileName_grids="lejeune50" --fileName_opt_trajs="produce_opt_trajs_time_limited"  --num_runs=1  --test_type="train_grid"
```
	
