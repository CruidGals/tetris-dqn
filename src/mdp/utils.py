import numpy as np

def one_hot(num_categories, idx):
    """ Returns a one-hot numpy array given idx and num_categories """
    return np.eye(num_categories)[idx]

def bin_grid(grid):
    """ Turn a grid of '.' and '#' into '0' and '1' """
    return (grid == '#').astype(np.uint8)

