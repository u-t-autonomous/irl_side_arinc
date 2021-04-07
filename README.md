this is an implementation of deep IRL with side info

To run the code, go to src directory and run python main_irl.py. This function calls env.py from env directory and utils.py from utils directory and agent.py from agent directory. 

For each run you need to fill in the corresponding parameters in main_irl.init()
	- first you need to choose if the run is for train, test or produce_optimal_trajectories
	- then depending on which one you chose, you need to fill in the corresponding parts in the following if-then cases
	