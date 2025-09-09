import numpy as np

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

class BlockFuncs:
    """ Static class for blocks in numpy arr form """
    def serialize(block: Block):
        """
        Serialize a block into a numpy array in form [block_type, rotation, row, col]
        """
        return np.array([block.type, block.rotation, block.pos[0], block.pos[1]], dtype=int)

    def get_pixel_pos(block):
        """
        Based on flip status, get the position of each pixel, or block, ina  tertromino block

        block: Numpy Array in form [block_type, rotation, row, col]
        """
        match block[0]:
            case Block.O:
                # Same regardless of flip status
                return [(block[2], block[3]), (block[2] + 1, block[3]), (block[2], block[3] + 1), (block[2] + 1, block[3] + 1)]
            case Block.I:
                # Only two flip patterns
                if block[1] % 2 == 0: # 0 or 180 deg
                    return [(block[2] - 1, block[3]), (block[2], block[3]), (block[2] + 1, block[3]), (block[2] + 2, block[3])]

                return [(block[2], block[3] - 1), (block[2], block[3]), (block[2], block[3] + 1), (block[2], block[3] + 2)]
            case Block.S:
                # 2 flip patterns
                if block[1] % 2 == 0: # 0 or 180 deg
                    return [(block[2], block[3] - 1), (block[2], block[3]), (block[2] - 1, block[3]), (block[2] - 1, block[3] + 1)]

                return [(block[2] - 1, block[3]), (block[2], block[3]), (block[2], block[3] + 1), (block[2] + 1, block[3] + 1)]
            case Block.Z:
                # 2 flip patterns
                if block[1] % 2 == 0: # 0 or 180 deg
                    return [(block[2] - 1, block[3] - 1), (block[2], block[3] - 1), (block[2], block[3]), (block[2], block[3] + 1)]

                return [(block[2] + 1, block[3]), (block[2], block[3]), (block[2], block[3] + 1), (block[2] - 1, block[3] + 1)]
            case Block.L:
                # 4 flip patterns
                if block[1] == 0: # 0 deg
                    return [(block[2] - 1, block[3]), (block[2], block[3]), (block[2] + 1, block[3]), (block[2] + 1, block[3] + 1)]
                elif block[1] == 1: # 90 deg
                    return [(block[2] + 1, block[3] - 1), (block[2], block[3] - 1), (block[2], block[3]), (block[2], block[3] + 1)]
                elif block[1] == 2: # 180 deg
                    return [(block[2] - 1, block[3] - 1), (block[2] - 1, block[3]), (block[2], block[3]), (block[2] + 1, block[3])]
                else: # 270 deg
                    return [(block[2], block[3] - 1), (block[2], block[3]), (block[2], block[3] + 1), (block[2] - 1, block[3] + 1)]
            case Block.J:
                # 4 flip patterns
                if block[1] == 0: # 0 deg
                    return [(block[2] + 1, block[3] - 1), (block[2] + 1, block[3]), (block[2], block[3]), (block[2] - 1, block[3])]
                elif block[1] == 1: # 90 deg
                    return [(block[2] - 1, block[3] - 1), (block[2], block[3] - 1), (block[2], block[3]), (block[2], block[3] + 1)]
                elif block[1] == 2: # 180 deg
                    return [(block[2] + 1, block[3]), (block[2], block[3]), (block[2] - 1, block[3]), (block[2] - 1, block[3] + 1)]
                else: # 270 deg
                    return [(block[2], block[3] - 1), (block[2], block[3]), (block[2], block[3] + 1), (block[2] + 1, block[3] + 1)]
            case Block.T:
                # 4 flip patterns
                if block[1] == 0: # 0 deg
                    return [(block[2], block[3] - 1), (block[2], block[3]), (block[2], block[3] + 1), (block[2] - 1, block[3])]
                elif block[1] == 1: # 90 deg
                    return [(block[2] - 1, block[3]), (block[2], block[3]), (block[2] + 1, block[3]), (block[2], block[3] + 1)]
                elif block[1] == 2: # 180 deg
                    return [(block[2], block[3] - 1), (block[2], block[3]), (block[2], block[3] + 1), (block[2] + 1, block[3])]
                else: # 270 deg
                    return [(block[2] - 1, block[3]), (block[2], block[3]), (block[2] + 1, block[3]), (block[2], block[3] - 1)]