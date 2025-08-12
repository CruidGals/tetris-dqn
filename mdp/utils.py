import numpy as np

def one_hot(num_categories, idx):
    """ Returns a one-hot numpy array given idx and num_categories """
    return np.eye(num_categories)[idx]

def grid_features(grid: np.ndarray):
    rows, cols = grid.shape

    # Column heights
    first_filled_row = np.argmax(grid, axis=0)

    # Handle when column has no blocks
    has_block = grid.any(axis=0)
    col_heights = np.where(has_block, rows - first_filled_row, 0) # Correct heights

    # Holes
    cum_filled = np.cumsum(grid, axis=0) # Starts sum after first filled block
    hole_mask = (grid == 0) & (cum_filled > 0) # Masks the holes
    col_holes = np.sum(hole_mask, axis=0)

    # Bumpiness
    bumpiness = np.abs(np.diff(first_filled_row)).sum()

    # Row transitions
    row_transitions = (np.diff(grid.astype(np.int8), axis=1) != 0).sum()

    # Col transitinos
    col_transitions = (np.diff(grid.astype(np.int8), axis=0) != 0).sum()

    # Normalize and return
    return {
        "col_heights": col_heights / rows,
        "col_holes": (col_holes / col_heights if col_heights.all() > 0 else np.zeros(10)),
        "bump": bumpiness / (rows * (cols - 1)),
        "row_transitions": row_transitions / (rows * (cols + 1)),
        "col_transitions": col_transitions / ((rows + 1) * cols)
    }
