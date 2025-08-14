import numpy as np
import pygame as py
import time
import sys
from collections import deque
from copy import deepcopy

import mdp.utils as utils
import mdp.observations as obs_terms

class Block:
    # Make each block an enum
    O = 0 # 2x2 block
    I = 1 # 1x4 block
    S = 2 # S shaped zigzag
    Z = 3 # Z shaped zigzag
    L = 4 # L shaped block
    J = 5 # Backwards L shaped block
    T = 6 # T shaped block

    def __init__(self, block_type):
        if not (0 <= block_type <= 6):
            raise Exception('Incorrect block type; use the global variables to safely set your block type')

        self.type = block_type

        # 0 deg: 0 | 90 deg: 1 | 180 deg: 2 | 270 deg: 3 (GOING CW)
        self.rotation = 0

        # Tetris is a 22*10 grid (first two rows aren't shown), specify spawn position as the bottom-left corner. Read coordinate left to right, top to bottom
        self._spawn_pos = (4,3) if self.type == Block.O else (3,3)
        self.pos = self._spawn_pos
        

    def flip(self, is_ccw_flip=False):
        # Handle if user flipped piece in the counter clockwise (CCW) direction
        if is_ccw_flip:
            self.rotation = self.rotation + 1 if self.rotation < 3 else 0
        else:
            self.rotation = self.rotation - 1 if self.rotation > 0 else 3

    def get_pixel_pos(self):
        """
        Based on flip status, get the position of each pixel, or block, ina  tertromino block
        """
        match self.type:
            case Block.O:
                # Same regardless of flip status
                return [(self.pos[0], self.pos[1]), (self.pos[0] + 1, self.pos[1]), (self.pos[0], self.pos[1] + 1), (self.pos[0] + 1, self.pos[1] + 1)]
            case Block.I:
                # Only two flip patterns
                if self.rotation % 2 == 0: # 0 or 180 deg
                    return [(self.pos[0] - 1, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0] + 1, self.pos[1]), (self.pos[0] + 2, self.pos[1])]
                
                return [(self.pos[0], self.pos[1] - 1), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1), (self.pos[0], self.pos[1] + 2)]
            case Block.S:
                # 2 flip patterns
                if self.rotation % 2 == 0: # 0 or 180 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0], self.pos[1]), (self.pos[0] - 1, self.pos[1]), (self.pos[0] - 1, self.pos[1] + 1)]
                
                return [(self.pos[0] - 1, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1), (self.pos[0] + 1, self.pos[1] + 1)]
            case Block.Z:
                # 2 flip patterns
                if self.rotation % 2 == 0: # 0 or 180 deg
                    return [(self.pos[0] - 1, self.pos[1] - 1), (self.pos[0], self.pos[1] - 1), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1)]
                
                return [(self.pos[0] + 1, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1), (self.pos[0] - 1, self.pos[1] + 1)]
            case Block.L:
                # 4 flip patterns
                if self.rotation == 0: # 0 deg
                    return [(self.pos[0] - 1, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0] + 1, self.pos[1]), (self.pos[0] + 1, self.pos[1] + 1)]
                elif self.rotation == 1: # 90 deg
                    return [(self.pos[0] + 1, self.pos[1] - 1), (self.pos[0], self.pos[1] - 1), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1)]
                elif self.rotation == 2: # 180 deg
                    return [(self.pos[0] - 1, self.pos[1] - 1), (self.pos[0] - 1, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0] + 1, self.pos[1])]
                else: # 270 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1), (self.pos[0] - 1, self.pos[1] + 1)]
            case Block.J:
                # 4 flip patterns
                if self.rotation == 0: # 0 deg
                    return [(self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0] - 1, self.pos[1])]
                elif self.rotation == 1: # 90 deg
                    return [(self.pos[0] - 1, self.pos[1] - 1), (self.pos[0], self.pos[1] - 1), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1)]
                elif self.rotation == 2: # 180 deg
                    return [(self.pos[0] + 1, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0] - 1, self.pos[1]), (self.pos[0] - 1, self.pos[1] + 1)]
                else: # 270 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1), (self.pos[0] + 1, self.pos[1] + 1)]
            case Block.T:
                # 4 flip patterns
                if self.rotation == 0: # 0 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1), (self.pos[0] - 1, self.pos[1])]
                elif self.rotation == 1: # 90 deg
                    return [(self.pos[0] - 1, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0] + 1, self.pos[1]), (self.pos[0], self.pos[1] + 1)]
                elif self.rotation == 2: # 180 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 1), (self.pos[0] + 1, self.pos[1])]
                else: # 270 deg
                    return [(self.pos[0] - 1, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0] + 1, self.pos[1]), (self.pos[0], self.pos[1] - 1)]


class Tetris:

    CLEAR_GRID = np.array([['.' for i in range(10)] for i in range(22)])

    MOVE_NONE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_CCW = 3
    MOVE_CW = 4

    def __init__(self, frame_rate=250, use_pygame=True):
        self.use_pygame = use_pygame 

        # Screen size is a grid, Denote . as nothing, and # as a filled
        self.grid = Tetris.CLEAR_GRID.copy() # 10 * 20, with 2 unrendered rows at the top
        self.cell_size = 30
        self.screen_size = (10 * self.cell_size, 20 * self.cell_size)

        if self.use_pygame:
            py.init()
            self.screen = py.display.set_mode(self.screen_size)
            py.display.set_caption('Mini Tetris Environment')

            self.game_clock = py.time.Clock()
        
        # Keep track of state of game
        self.done = False
        self.landed_block_count = 0
        self.rows_cleared = 0

    def start(self):
        """
        Begins the tetris environment
        """
        # Keep a queue of blocks
        self.block_queue = deque([Block(np.random.randint(0,7)) for i in range(3)], maxlen=3)

        # Keep track of the controlled block
        self.cycle_block()
        pass

    def cycle_block(self):
        """
        Pops out a block from the block_queue and adds a new random one in
        """
        reward = 0

        self.controlled_block = self.block_queue.popleft()
        self.block_queue.append(Block(np.random.randint(0,7)))

        # If the block spawns within a collision, game over!
        if not self.valid_pos(self.controlled_block.get_pixel_pos()):
            self.done = True


    def move(self, type):
        """
        Make a move in tetris. There are four moves I will allow the environment to make: CCW turn, CW turn, and left and right movement of the block.
        Typically allow a movement per frame, and the blocks will fall one block every 20gi frames.
        In the case of moving while the block is falling, the **falling will process first**, then the movement.

        Args:
            type: the type of movement that will be performed. Use the global variables at the very top
        """

        match type:
            case Tetris.MOVE_CCW:
                self.controlled_block.flip(True)

                # TODO implement wall kicking
                if not self.valid_pos(self.controlled_block.get_pixel_pos()):
                    self.controlled_block.flip()
            case Tetris.MOVE_CW:
                self.controlled_block.flip()

                # TODO implement wall kicking
                if not self.valid_pos(self.controlled_block.get_pixel_pos()):
                    self.controlled_block.flip(True)
            case Tetris.MOVE_LEFT:
                orig_pos = self.controlled_block.pos
                self.controlled_block.pos = (self.controlled_block.pos[0] - 1, self.controlled_block.pos[1])

                if not self.valid_pos(self.controlled_block.get_pixel_pos()):
                    self.controlled_block.pos = orig_pos
            case Tetris.MOVE_RIGHT:
                orig_pos = self.controlled_block.pos
                self.controlled_block.pos = (self.controlled_block.pos[0] + 1, self.controlled_block.pos[1])

                if not self.valid_pos(self.controlled_block.get_pixel_pos()):
                    self.controlled_block.pos = orig_pos
            case Tetris.MOVE_NONE:
                pass #Do nothing
            case _:
                raise Exception(f"Invalid move type (from 0-3). Given type: {type}")

    def fall(self):
        """
        Simulate the block falling by one block. Return True if block has landed
        """
        orig_pos = self.controlled_block.pos
        self.controlled_block.pos = (self.controlled_block.pos[0], self.controlled_block.pos[1] + 1)

        if not self.valid_pos(self.controlled_block.get_pixel_pos()):
            self.controlled_block.pos = orig_pos

            # Land block behavior
            for y, x in self.controlled_block.get_pixel_pos():
                self.grid[x][y] = '#'
            
            self.cycle_block()
            self.landed_block_count += 1
            return True
        
        return False

    def valid_pos(self, positions):
        """
        Given a set of coordinates, return a boolean indicating whether those coordinates don't collide with other blocks

        Args:
            positions: A list of positions to check over
        """

        for pos in positions:
            # If outta bounds
            if not (0 <= pos[0] < 10) or not (0 <= pos[1] < 22):
                return False
            
            # If colliding with another block
            if self.grid[pos[1]][pos[0]] == "#":
                return False
        
        return True
    
    def clear_rows(self):
        """
        Clears any rows that are fully filled, counting them along the way
        """
        
        # Remove any full rows
        # Statement literally gets all indices with full rows, and indexes the grid without those indices
        cleared_grid = self.grid[~np.all(self.grid == '#', axis=1)]

        # Count lines cleared
        lines_cleared = self.grid.shape[0] - cleared_grid.shape[0]

        # Add empty rows at top
        if lines_cleared > 0:
            new_rows = np.full((lines_cleared, cleared_grid.shape[1]), fill_value='.', dtype=cleared_grid.dtype)
            cleared_grid = np.vstack((new_rows, cleared_grid))

        self.grid = cleared_grid
        return lines_cleared

    # Pygame specific    
    def render(self):
        if not self.use_pygame:
            return
        
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                self.use_pygame = False
                return

        self.screen.fill((0,0,0))

        # Draw the rendered rows (rows 2 - 22)
        for i in range(20):
            for j in range(10):
                type = self.grid[i + 2][j]
                if type == "#":
                    #Draw a white square for each #
                    py.draw.rect(self.screen, (255,255,255), (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

        # Draw the position of the controlled block
        for x, y in self.controlled_block.get_pixel_pos():
            py.draw.rect(self.screen, (255,0,0), (x * self.cell_size, (y-2) * self.cell_size, self.cell_size, self.cell_size))

        # Render the screen as shown above and the framerate
        py.display.flip()

        # Framerate and counting
        self.game_clock.tick(60)
 
    def close(self):
        """
        Quit the environment
        """
        if self.use_pygame:
            py.quit()

class TetrisEnv(Tetris):
    CLEAR_GRID = np.array([['.' for i in range(10)] for i in range(22)])

    REWARDS = {
        "survival": 1.0,
        "clear_row_base": 35,
        "clear_row_scale": 2.0,
        "death": 5.0
    }

    def __init__(self, frame_rate=250, headless=False):
        super().__init__(frame_rate, not headless)
        self.grid = TetrisEnv.CLEAR_GRID.copy()

    # Overriden functions
    def clear_rows(self):
        cleared_row_idx = np.where(np.all(self.grid == '#', axis=1))[0]
        lines_cleared = super().clear_rows()
        self.rows_cleared += lines_cleared

        return lines_cleared, cleared_row_idx
    
    def land(self):
        """
        Modified fall function that lands the block immediately.
        Returns the landed block positions
        """
        block_positions = np.array(self.controlled_block.get_pixel_pos())
        self.grid[block_positions[:, 0], block_positions[:, 1]] = '#'
        self.landed_block_count += 1
        self.cycle_block()
        
        return block_positions


    def step(self, action: int):
        """ 
        Perform the action, and return a ndarray in the form:
        (obs, reward, terminated)

        In TetrisEnv, an action is performed as picking a rotation for the block's orientation and a column for the block to fall in
        Action is taken from a 40 dim one-hot vector where indices:
        - 0-9: Column # with rotation 0
        - 10-19: Column # with rotation 1
        - 20-29: Column # with rotation 2
        - 30-39: Column # with rotation 3
        """

        rows, cols = self.grid.shape

        # Parse action information: Tens digit is rotation, Ones digit is column number
        action_temp = str(action)
        rotation = int(action_temp[0]) if len(action_temp) > 1 else 0
        column = int(action_temp[1]) if len(action_temp) > 1 else int(action_temp[0])
        
        # Position the block
        self.controlled_block.rotation = rotation
        self.controlled_block.pos = (self.controlled_block.pos[0], column)

        # Handle case: block goes out of bound if placed in certain column
        block_position_cols = np.array(self.controlled_block.get_pixel_pos(), dtype=int)[:, 1]
        overflow = min(0, block_position_cols.min()) + max(0, block_position_cols.max() - (cols - 1))
        self.controlled_block.pos = (self.controlled_block.pos[0], int(self.controlled_block.pos[1] - overflow))

        # Update block position
        block_positions = np.array(self.controlled_block.get_pixel_pos())

        # Get highest row pos for each column the block is in
        cols = block_positions[:,  1]
        unique_cols = np.unique(cols)
        idx = [np.where(cols == c)[0][np.argmax(block_positions[cols == c, 0])] for c in unique_cols]
        unique_pos = block_positions[idx]

        # Get col heights (inverted)
        selected_columns = self.grid[:, unique_cols[0]:unique_cols[-1] + 1] == '#'
        col_heights = np.where(np.any(selected_columns, axis=0), np.argmax(selected_columns, axis=0), rows)

        # If any blocks collide with already landed ones, terminate
        if np.isin(block_positions[:, 0], col_heights).any():
            return self._get_observation(block_positions), TetrisEnv.REWARDS["death"], True
        
        dy = min(col_heights - unique_pos[:, 0]) - 1
        self.controlled_block.pos = (int(self.controlled_block.pos[0] + dy), self.controlled_block.pos[1])

        # Officially land block!
        landed_block_positions = self.land()
        lines_cleared, cleared_row_idx = self.clear_rows()

        # Return MDP step
        obs = self._get_observation(block_positions=landed_block_positions, lines_cleared=lines_cleared, cleared_row_idx=cleared_row_idx)
        reward = self._get_reward(lines_cleared)

        return obs, reward, False

    # Environment specific functions
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
        return self._get_observation(np.array(self.controlled_block.get_pixel_pos()))

    def _get_observation(self, block_positions, lines_cleared=0, cleared_row_idx=np.array([])):
        """
        Returns the visual part of the grid, denoting '.' as 0, '#' as .5, and '$' as 1

        Also contains various other observations like falling block type, rotation, position
        """
        rows, cols = self.grid.shape
        bin_grid = utils.bin_grid(self.grid)

        # Get all terms (normalize if need be)

        col_heights = obs_terms.height_per_column(bin_grid)
        col_heights_norm = col_heights / 20
        holes_per_col = (obs_terms.holes_count(bin_grid) / col_heights if col_heights.all() > 0 else np.zeros(10))
        bumpiness = obs_terms.calc_bumpiness(col_heights) / (rows * (cols - 1))
        row_transitions, col_transitions = obs_terms.count_transitions(bin_grid)
        row_transitions /= (rows * (cols + 1))
        col_transitions /= ((rows + 1) * cols)
        eroded_cells = obs_terms.eroded_cells(lines_cleared, cleared_row_idx, block_positions, rows)
        num_wells = obs_terms.cumulative_wells(bin_grid) / (cols * (rows * (rows + 1) // 2)) # Normalized by max number of wells
        landing_height = obs_terms.landing_height(block_positions, rows) / rows

        # Return observation concatenated altogether
        return np.concatenate([
            col_heights_norm.astype(np.float32),
            holes_per_col.astype(np.float32),
            np.array([bumpiness], dtype=np.float32),
            np.array([row_transitions], dtype=np.float32),
            np.array([col_transitions], dtype=np.float32),
            np.array([eroded_cells], dtype=np.float32),
            np.array([num_wells], dtype=np.float32),
            np.array([landing_height], dtype=np.float32),
        ])

    
    def _get_reward(self, lines_cleared):
        """
        Distribute the reward after a block lands (counts as one step in this environment),
        """
        return TetrisEnv.REWARDS['survival'] + TetrisEnv.REWARDS['clear_row_base'] * (lines_cleared ** TetrisEnv.REWARDS['clear_row_scale'])

# For if the user would like to play on their own
if __name__ == "__main__":
    # # Test code
    # env = Tetris(use_pygame=False)

    tetris_grid = [
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # row 0 (top)
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 1
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 2
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 3
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 4
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 5
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 6
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 7
        ['#', '.', '#', '.', '.', '.', '.', '.', '.', '.'],  # 8
        ['#', '#', '#', '.', '.', '#', '.', '.', '.', '.'],  # 9
        ['#', '.', '#', '.', '#', '#', '.', '.', '.', '.'],  # 10
        ['#', '.', '#', '#', '#', '#', '#', '#', '#', '.'],  # 11
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '.'],  # 12
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '.'],
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '.'],
        ['#', '#', '.', '#', '#', '#', '.', '.', '#', '.'],
        ['#', '#', '.', '#', '#', '#', '.', '.', '#', '.'],
        ['#', '#', '#', '#', '#', '#', '.', '.', '#', '.'],
        ['#', '#', '#', '#', '#', '#', '.', '.', '#', '.'],
        ['#', '#', '#', '#', '#', '#', '.', '.', '#', '.'],
        ['#', '#', '#', '#', '#', '#', '#', '.', '#', '.'],
        ['#', '#', '#', '#', '#', '#', '#', '.', '#', '.'],  # row 21 (bottom)
    ]

    env = TetrisEnv(headless=True)
    env.grid = np.array(tetris_grid)
    env.start()
    env.controlled_block = Block(Block.J)
    print(env.grid)

    obs, rew, term = env.step(9)
    print(env.grid)
    print(obs, rew, term)

    # env.grid = tetris_grid
    
    # seen_blocks = set()
    # hole_blocks = set()

    # for i, row in enumerate(env.grid[9:]):
    #     if '.' not in row:
    #         continue

    #     first_empty = row.index('.')
    #     if (first_empty, 9 + i) in seen_blocks:
    #         continue

    #     propogated_blocks = set()

    #     if env.is_hole(first_empty, 9 + i, 9, propogated_blocks):
    #         hole_blocks.update(propogated_blocks)
            
    #     seen_blocks.update(propogated_blocks)

    # print(seen_blocks)
    # print(hole_blocks)

    # env = Tetris(use_pygame=True)
    # env.start()

    # running = True
    # while running:
    #     for event in py.event.get():
    #         if event.type == py.QUIT:
    #             running = False
    #         elif event.type == py.KEYDOWN:
    #             if event.key == py.K_q:
    #                 env.move(Tetris.MOVE_CCW)
    #             if event.key == py.K_e:
    #                 env.move(Tetris.MOVE_CW)
    #             if event.key == py.K_a:
    #                 env.move(Tetris.MOVE_LEFT)
    #             if event.key == py.K_d:
    #                 env.move(Tetris.MOVE_RIGHT)

    #     # Falling behavior (every 4 frames)
    #     if env.clock.frame_count % 4 == 0:
    #         env.fall()
    #         env.clear_rows()

    #     if env.done:
    #         running=False

    #     env.render()
    
    # env.close()