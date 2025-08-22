import numpy as np
import pygame as py
from collections import deque

import mdp.env as env_funcs
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

    def __init__(self, frame_rate=250, headless=False):
        super().__init__(frame_rate, not headless)
        self.grid = TetrisEnv.CLEAR_GRID.copy()

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

        Previous observation is also taken to calculate the delta for rewards
        """

        # Make the block fall in the column
        if not env_funcs.fall(action, self.grid, self.controlled_block):
            return env_funcs._get_observation(self.grid, np.array(self.controlled_block.get_pixel_pos()), self.controlled_block.type),-5.0, True

        # Officially land block!
        landed_block_positions, landed_block_type = env_funcs.land(self.grid, self.controlled_block)
        lines_cleared, cleared_row_idx = env_funcs.clear_rows(self.grid)
        self.cycle_block()
        self.landed_block_count += 1
        self.rows_cleared += lines_cleared

        # Return MDP step
        obs = env_funcs._get_observation(grid=self.grid, block_positions=landed_block_positions, block_type=landed_block_type, lines_cleared=lines_cleared, cleared_row_idx=cleared_row_idx)
        reward = env_funcs._get_reward(lines_cleared, obs)
        term = np.any(obs_terms.height_per_column(utils.bin_grid(self.grid)) >= 20) # If any of heights are 20 or above

        return obs, reward, term

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
        return env_funcs._get_observation(self.grid, np.array(self.controlled_block.get_pixel_pos()), self.controlled_block.type)


# For if the user would like to play on their own
if __name__ == "__main__":
    env = Tetris(use_pygame=True)
    env.start()

    running = True
    while running:
        for event in py.event.get():
            if event.type == py.QUIT:
                running = False
            elif event.type == py.KEYDOWN:
                if event.key == py.K_q:
                    env.move(Tetris.MOVE_CCW)
                if event.key == py.K_e:
                    env.move(Tetris.MOVE_CW)
                if event.key == py.K_a:
                    env.move(Tetris.MOVE_LEFT)
                if event.key == py.K_d:
                    env.move(Tetris.MOVE_RIGHT)

        # Falling behavior (every 4 frames)
        if env.clock.frame_count % 4 == 0:
            env.fall()
            env.clear_rows()

        if env.done:
            running=False

        env.render()
    
    env.close()