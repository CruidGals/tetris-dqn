import sys, os
import copy
import numpy as np
from src.tetris_env import TetrisEnv
import src.mdp.utils as utils
import src.mdp.observations as obs_terms

class HeuristicTetrisEnv(TetrisEnv):
    def __init__(self, frame_rate=250):
        super().__init__(frame_rate, True)

    def fitness_per_move(self, curr_state, blocks, idx):
        fitness_per_action = np.zeros(40, dtype=np.float32)

        # Find best move for each action
        for i in range(40):
            self.grid = curr_state.copy()
            self.controlled_block = copy.deepcopy(blocks[idx])
            _, rew, done = self.step(i)

            if done:
                fitness_per_action[i] = float('-inf')
                continue
            
            # Perform lookahead (only goes to one block)
            if idx < 1:
                fitness_per_action[i] = self.find_best_move(self.grid.copy(), blocks, idx + 1)[1]
            else:
                fitness_per_action[i] = rew

        return fitness_per_action
            
    def find_best_move(self, curr_state, blocks, idx):
        """
        Finds the best move based on the heuristic function given blocks and the block index to check
        """
        fpm =self.fitness_per_move(curr_state, blocks, idx)
        return (fpm.argmax().item(), np.max(fpm).item())
    
    def step(self, action: int):

        # Place the block into position
        if not self._fall(action):
            return self._get_observation(), float("-inf"), True

        # Officially land block!
        self.land()
        lines_cleared, _ = self.clear_rows()

        # Return MDP step
        obs = self._get_observation()
        reward = self._get_reward(lines_cleared, obs)
        term = np.any(obs_terms.height_per_column(utils.bin_grid(self.grid)) >= 20) # If any of heights are 20 or above

        return obs, reward, term

    def _get_observation(self):
        bin_grid = utils.bin_grid(self.grid)

        col_heights = obs_terms.height_per_column(bin_grid)
        aggregate_height = col_heights.sum()
        total_holes = obs_terms.holes_count(bin_grid).sum()
        bumpiness = obs_terms.calc_bumpiness(col_heights)

        return np.concatenate([
            np.array([aggregate_height], dtype=np.float32),
            np.array([total_holes], dtype=np.float32),
            np.array([bumpiness], dtype=np.float32),
        ])

    def _get_reward(self, lines_cleared, obs):
        """
        Distribute the reward after a block lands (counts as one step in this environment),

        Heuristic function inspired by:
            L. Yiyuan. Tetris ai - the (near) perfect bot. https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
        """
        return float(-0.510066 * obs[0] + 0.760666 * lines_cleared - 0.35663 * obs[1] - 0.184483 * obs[2])
    
    def reset(self):
        """
        Resets the environment for the next episode
        """
        self.grid = TetrisEnv.CLEAR_GRID.copy()
        self.done = False
        self.frame_count = 0
        self.block_queue = []
        self.landed_block_count = 0
        self.rows_cleared = 0

        self.start()
        return self._get_observation()