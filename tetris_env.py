import numpy as np
import pygame as py
from collections import deque

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

    MOVE_CCW = 0
    MOVE_CW = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3

    def __init__(self):
        py.init()

        # Screen size is a grid, Denote . as nothing, and # as a filled
        self.grid = [['.' for i in range(10)] for i in range(22)] # 10 * 20, with 2 unrendered rows at the top
        self.cell_size = 30
        self.screen_size = (10 * self.cell_size, 20 * self.cell_size)
        self.screen = py.display.set_mode(self.screen_size)
        py.display.set_caption('Mini Tetris Environment')

        # Set a clock for tetris
        self.clock = py.time.Clock()

        # Sprites on the screen have different timings. Use a frame counter
        self.frame_count = 0

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
        self.controlled_block = self.block_queue.popleft()
        self.block_queue.append(Block(np.random.randint(0,7)))

        # If the block spawns within a collision, game over!
        if not self.valid_pos(self.controlled_block.get_pixel_pos()):
            print("Game Over!")
            self.close()

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
            case _:
                raise Exception(f"Invalid move type (from 0-3). Given type: {type}")

    def fall(self):
        """
        Simulate the block falling by one block
        """
        orig_pos = self.controlled_block.pos
        self.controlled_block.pos = (self.controlled_block.pos[0], self.controlled_block.pos[1] + 1)

        if not self.valid_pos(self.controlled_block.get_pixel_pos()):
            self.controlled_block.pos = orig_pos

            # Land block behavior
            for y, x in self.controlled_block.get_pixel_pos():
                self.grid[x][y] = '#'
            
            self.cycle_block()

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
        """

        for i in range(len(self.grid)):
            if self.grid[i].count('#') == 10:
                self.grid.pop(i)
                self.grid.insert(0, ['.' for i in range(10)])
        
    def render(self):
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
        self.clock.tick(60)
        self.frame_count = self.frame_count + 1 if self.frame_count < 59 else 0
 
    def close(self):
        """
        Quit the environment
        """
        py.quit()

if __name__ == "__main__":
    env = Tetris()
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

        # Falling behavior
        if env.frame_count % 10 == 0:
            env.fall()
            env.clear_rows()

        env.render()
    
    env.close()