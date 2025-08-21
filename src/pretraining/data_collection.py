import os
import numpy as np
import pygame as py
from src.tetris_env import * 
from src.pretraining.heuristic_env import HeuristicTetrisEnv

env = TetrisEnv(frame_rate=2, headless=True)
heuristic_env = HeuristicTetrisEnv()
heuristic_env.reset()
game_clock = py.time.Clock()

obs = env.reset()
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data.npz'))

running = True
while running:
    for event in py.event.get():
        if event.type == py.QUIT:
            running = False

    # Play what the hueristic env thinks is best
    action = heuristic_env.find_best_move(env.grid.copy(), np.concatenate([[env.controlled_block], list(env.block_queue)]), 0)
    new, rew, done = env.step(action[0], obs)

    if done:
        running=False

    obs = new.copy()
    env.render()
    game_clock.tick(60)

env.close()