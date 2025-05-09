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

    def flip(self, is_ccw_flip):
        # Handle if user flipped piece in the counter clockwise (CCW) direction
        if is_ccw_flip:
            self.flipped_status = self.flipped_status + 1 if self.flipped_status < 2 else 0
        else:
            self.flipped_status = self.flipped_status - 1 if self.flipped_status > 0 else 3

    def render(self):
        """
        Tetris blocks have a "area of rotation", in which it facilitates rotation of the piece
        More can be seen through this link: 
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

        # Screen size is a grid
        self.grid_size = 30
        self.screen_size = (12 * self.grid_size, 24 * self.grid_size)
        self.screen = py.display.set_mode(self.screen_size)
        py.display.set_caption('Mini Tetris Environment')

        # Set a clock for tetris
        self.clock = py.time.Clock()

        # Sprites on the screen have different timings. Use a frame counter
        self.frame_count = 0
        
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