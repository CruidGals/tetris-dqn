import numpy as np
import pygame as py
import time
import sys
from collections import deque

class Clock:
    def __init__(self, frame_rate):
        self.frame_rate = frame_rate
        self.prev_time = time.time()
        self.frame_count = 0

    def tick(self):
        current_time = time.time()
        
        if current_time - self.prev_time >= (1/self.frame_rate):
            self.frame_count += 1
            self.prev_time = current_time

        if self.frame_count == self.frame_rate:
            self.frame_count = 0
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

        # 0 deg: 0 | 90 deg: 1 | 180 deg: 2 | 270 deg: 3 (GOING CCW)
        self.flipped_status = 0

        # Tetris is a 22*10 grid (first two rows aren't shown), specify spawn position as the bottom-left corner. Read coordinate left to right, top to bottom
        self._spawn_pos = (4,3) if self.type == Block.O else (3,3)
        self.pos = self._spawn_pos
        

    def flip(self, is_ccw_flip=False):
        # Handle if user flipped piece in the counter clockwise (CCW) direction
        if is_ccw_flip:
            self.flipped_status = self.flipped_status + 1 if self.flipped_status < 3 else 0
        else:
            self.flipped_status = self.flipped_status - 1 if self.flipped_status > 0 else 3

    def get_pixel_pos(self):
        """
        Based on flip status, get the position of each pixel, or block, ina  tertromino block
        """
        match self.type:
            case Block.O:
                # Same regardless of flip status
                return [(self.pos[0], self.pos[1]), (self.pos[0] + 1, self.pos[1]), (self.pos[0], self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 1)]
            case Block.I:
                # Only two flip patterns
                if self.flipped_status % 2 == 0: # 0 or 180 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1), (self.pos[0] + 3, self.pos[1] - 1)]
                
                return [(self.pos[0] + 2, self.pos[1]), (self.pos[0] + 2, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 2), (self.pos[0] + 2, self.pos[1] - 3)]
            case Block.S:
                # 2 flip patterns
                if self.flipped_status % 2 == 0: # 0 or 180 deg
                    return [(self.pos[0], self.pos[1]), (self.pos[0] + 1, self.pos[1]), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1)]
                
                return [(self.pos[0] + 1, self.pos[1] - 2), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1])]
            case Block.Z:
                # 2 flip patterns
                if self.flipped_status % 2 == 0: # 0 or 180 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1]), (self.pos[0] + 2, self.pos[1])]
                
                return [(self.pos[0] + 1, self.pos[1]), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 2)]
            case Block.L:
                # 4 flip patterns
                if self.flipped_status == 0: # 0 deg
                    return [(self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1)]
                elif self.flipped_status == 1: # 90 deg
                    return [(self.pos[0] + 1, self.pos[1] - 2), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1]), (self.pos[0] + 2, self.pos[1])]
                elif self.flipped_status == 2: # 180 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 2)]
                else: # 270 deg
                    return [(self.pos[0], self.pos[1] - 2), (self.pos[0] + 1, self.pos[1] - 2), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1])]
            case Block.J:
                # 4 flip patterns
                if self.flipped_status == 0: # 0 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1])]
                elif self.flipped_status == 1: # 90 deg
                    return [(self.pos[0] + 1, self.pos[1]), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 2), (self.pos[0] + 2, self.pos[1] - 2)]
                elif self.flipped_status == 2: # 180 deg
                    return [(self.pos[0], self.pos[1] - 2), (self.pos[0], self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1)]
                else: # 270 deg
                    return [(self.pos[0], self.pos[1]), (self.pos[0] + 1, self.pos[1]), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 2)]
            case Block.T:
                # 4 flip patterns
                if self.flipped_status == 0: # 0 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1])]
                elif self.flipped_status == 1: # 90 deg
                    return [(self.pos[0] + 1, self.pos[1] - 2), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1]), (self.pos[0] + 2, self.pos[1] - 1)]
                elif self.flipped_status == 2: # 180 deg
                    return [(self.pos[0], self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 2, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1] - 2)]
                else: # 270 deg
                    return [(self.pos[0] + 1, self.pos[1] - 2), (self.pos[0] + 1, self.pos[1] - 1), (self.pos[0] + 1, self.pos[1]), (self.pos[0], self.pos[1] - 1)]


class Tetris:

    MOVE_NONE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_CCW = 3
    MOVE_CW = 4

    def __init__(self, frame_rate=250, use_pygame=True):
        self.use_pygame = use_pygame 

        # Screen size is a grid, Denote . as nothing, and # as a filled
        self.grid = [['.' for i in range(10)] for i in range(22)] # 10 * 20, with 2 unrendered rows at the top
        self.cell_size = 30
        self.screen_size = (10 * self.cell_size, 20 * self.cell_size)

        # Set up a clock to control falling
        self.clock = Clock(frame_rate=frame_rate)

        if self.use_pygame:
            py.init()
            self.screen = py.display.set_mode(self.screen_size)
            py.display.set_caption('Mini Tetris Environment')

            self.game_clock = py.time.Clock()
        
        # Keep track of state of game
        self.done = False
        self.landed_block_count = 0

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
        
        return self._get_observation()

    def fall(self):
        """
        Simulate the block falling by one block
        """
        orig_pos = self.controlled_block.pos
        self.controlled_block.pos = (self.controlled_block.pos[0], self.controlled_block.pos[1] + 1)
        reward = 0

        if not self.valid_pos(self.controlled_block.get_pixel_pos()):
            self.controlled_block.pos = orig_pos

            # Land block behavior
            for y, x in self.controlled_block.get_pixel_pos():
                self.grid[x][y] = '#'
            
            self.cycle_block()
            reward = self.distribute_reward()
            self.clear_rows()
            self.landed_block_count += 1

        return reward

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
        Clears any rows that are fully filled
        If cleared 4 rows at once, give bonus points
        """
        count = 0

        for i in range(len(self.grid)):
            if self.grid[i].count('#') == 10:
                self.grid.pop(i)
                self.grid.insert(0, ['.' for i in range(10)])
                count += 1

    # Environment specific functions
    def reset(self):
        """
        Resets the environment for the next episode
        """
        self.grid = [['.' for i in range(10)] for i in range(22)]
        self.done = False
        self.frame_count = 0
        self.block_queue = []
        self.landed_block_count = 0

        self.start()
        return self._get_observation()

    def _get_observation(self):
        """
        Returns the visual part of the grid, denoting '.' as 0, '#' as .5, and '$' as 1
        """

        transformed_grid = np.array(self.grid).copy()

        # Turn empty spaces '.' into 0, and filled spaces '#' into 1
        mapping = {'.': 0, '#': 1}
        transformed_grid = np.vectorize(lambda x: mapping[x])(transformed_grid).astype('float64')

        # Fill in falling block as 0.5
        for y, x in self.controlled_block.get_pixel_pos():
            transformed_grid[x][y] = 0.5

        # Constrin to visual part of grid
        return transformed_grid[2:, :].flatten()
    
    def distribute_reward(self):
        """
        Distribute the reward after a block lands (counts as one step in this environment).
        Features rewards such as: survival (low), clearing rows (big), encouraging flatter structure (moderate), penalizing holes, encouraging lower height
        """
        reward = 0

        # Clear rows reward
        for row in self.grid:
            # A full row is cleared
            if row.count('#') == 10:
                # Decrease or increase based on performance
                reward += 5

        # Penalize height
        highest_row = -1
        for i, row in enumerate(self.grid):
            if '#' in row and highest_row == -1:
                highest_row = i
                break 

        reward -= (21 - highest_row) ** 1.5 * 0.01

        # Find holes and penalize them
        holes = 0

        for col in range(10):
            block_found = False
            for row in range(22):
                if self.grid[row][col] == '#':
                    block_found = True
                elif self.grid[row][col] == '.' and block_found:
                    holes += 1
        
        reward -= holes * 0.05
        

        # Penalize dying early
        if self.done:
            reward -= np.exp(-(self.landed_block_count-10) / 10) * 5

        return reward 

    def is_hole(self, x, y, highest_row, seen_blocks):
        """
        Determines whether or not a space is a "hole" based on the starting location.
        A hole is defined as empty space that cannot be reached without clearing any rows.
        We can find holes by propogating outwards until we reach the very top of the grid
        (or in this case, the top of the highest row that contains a landed block).
        """
        seen_blocks.add((x,y))

        left, right, up, down = True, True, True, True

        if y == highest_row:
            return False

        if (x != 0 and self.grid[y][x-1] != '#') and not (x-1, y) in seen_blocks:
            left = self.is_hole(x-1, y, highest_row, seen_blocks)
        if (x != 9 and self.grid[y][x+1] != '#') and not (x+1, y) in seen_blocks:
            right = self.is_hole(x+1, y, highest_row, seen_blocks)
        if (y != 21 and self.grid[y+1][x] != '#') and not (x, y+1) in seen_blocks:
            down = self.is_hole(x, y+1, highest_row, seen_blocks)
        if (y > highest_row and self.grid[y-1][x] != '#') and not (x, y-1) in seen_blocks:
            up = self.is_hole(x, y-1, highest_row, seen_blocks)

        return left and right and down and up

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

# For if the user would like to play on their own
if __name__ == "__main__":
    # # Test code
    # env = Tetris(use_pygame=False)

    # tetris_grid = [
    #     ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # row 0 (top)
    #     ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 1
    #     ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 2
    #     ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 3
    #     ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 4
    #     ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 5
    #     ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 6
    #     ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 7
    #     ['#', '.', '#', '.', '.', '.', '.', '.', '.', '.'],  # 8
    #     ['#', '#', '#', '.', '.', '#', '.', '.', '.', '.'],  # 9
    #     ['#', '.', '#', '.', '#', '#', '.', '.', '.', '.'],  # 10
    #     ['#', '.', '#', '#', '#', '#', '#', '#', '#', '.'],  # 11
    #     ['#', '#', '#', '#', '#', '#', '.', '.', '#', '.'],  # 12
    #     ['#', '#', '#', '#', '#', '#', '#', '.', '#', '.'],
    #     ['#', '#', '#', '#', '#', '#', '#', '.', '#', '.'],
    #     ['#', '#', '.', '#', '#', '#', '.', '.', '#', '.'],
    #     ['#', '#', '.', '#', '#', '#', '.', '.', '#', '.'],
    #     ['#', '#', '#', '#', '#', '#', '.', '.', '#', '#'],
    #     ['#', '#', '#', '#', '#', '#', '.', '.', '#', '#'],
    #     ['#', '#', '#', '#', '#', '#', '.', '.', '#', '#'],
    #     ['#', '#', '#', '#', '#', '#', '#', '.', '#', '#'],
    #     ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],  # row 21 (bottom)
    # ]

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