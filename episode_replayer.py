import numpy as np
import argparse
import pygame as py
from tetris_env import Tetris

def replay(arr):
    # Make into iterator for easy access
    arr = iter(arr)
    mapping = {0.0: '.', 0.5: '$', 1.0: '#'}

    # Pygame boilerplate
    py.init()
    screen = py.display.set_mode((300, 600))
    py.display.set_caption('Mini Tetris Environment')

    game_clock = py.time.Clock()

    running = True
    while running:
        screen.fill((0,0,0))

        for event in py.event.get():
            if event.type == py.QUIT:
                running = False
        
        grid = next(arr, np.array([[]]))
        if grid.any():
            grid = np.vectorize(lambda x: mapping[x])(np.reshape(grid, (-1, 10)))
        else:
            running = False
            break
        
        for i in range(20):
            for j in range(10):
                if grid[i][j] == "#":
                    py.draw.rect(screen, (255,255,255), (j * 30, i * 30, 30, 30))
                elif grid[i][j] == '$':
                    py.draw.rect(screen, (255,0,0), (j * 30, i * 30, 30, 30))

        py.display.flip()

        game_clock.tick(30)

    py.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the episode replay")
    args = parser.parse_args()
    arr = np.load(args.input)
    
    replay(arr)