import sys
import os
import copy
import pygame as py

# Add path of tetris env
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..') 
sys.path.append(parent_dir)
from tetris_env import * 

env = TetrisEnv(frame_rate=2, headless=False)
heuristic_env = HeuristicTetrisEnv()
game_clock = py.time.Clock()

obs = env.reset()

running = True
while running:
    for event in py.event.get():
        if event.type == py.QUIT:
            running = False

    # Play what the hueristic env thinks is best
    action = heuristic_env.find_best_move(env.grid.copy(), copy.deepcopy(env.controlled_block))
    new, rew, done = env.step(action, obs)
    print(rew)
    if done:
        running=False

    obs = new.copy()
    env.render()
    game_clock.tick(1)

env.close()