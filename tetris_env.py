import numpy as np
import pygame as py

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
        self.type = block_type

        # 0 deg: 0 | 90 deg: 1 | 180 deg: 2 | 270 deg: 3 (GOING CCW)
        self.flipped_status = 0

        # Tetris is a 10*20 grid, specify spawn position as the bottom-left corner. Read coordinate left to right, top to bottom
        self._spawn_pos = (4,1) if self.type == Block.O else (3,1)
        self.pos = self._spawn_pos
        

    def flip(self, is_ccw_flip):
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


    def render(self, grid):
        """
        Tetris blocks have a "area of rotation", in which it facilitates rotation of the piece
        More can be seen through this link: https://tetris.fandom.com/wiki/DTET_Rotation_System
        """

        match self.type:
            case Block.O:
                pass
            case Block.I:
                pass
            case Block.S:
                pass
            case Block.Z:
                pass
            case Block.L:
                pass
            case Block.J:
                pass
            case Block.T:
                pass

class Tetris:
    def __init__(self):
        py.init()

        # Screen size is a grid, Denote . as nothing, and # as a block 
        self.grid = [[] for i in range(22)] # 10 * 20, with 2 unrendered rows at the top
        self.grid_size = 30
        self.screen_size = (10 * self.grid_size, 20 * self.grid_size)
        self.screen = py.display.set_mode(self.screen_size)
        py.display.set_caption('Mini Tetris Environment')

        # Set a clock for tetris
        self.clock = py.time.Clock()

        # Sprites on the screen have different timings. Use a frame counter
        self.frame_count = 0

    def valid_pos(self, positions):
        for pos in positions:
            if self.grid[pos[0], pos[1]] == "#":
                return False
        
        return True
        
    def render(self):
        self.screen.fill((255,0,0))

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

    running = True
    while running:
        for event in py.event.get():
            if event.type == py.QUIT:
                running = False

        env.render()
    
    env.close()