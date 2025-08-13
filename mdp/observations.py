import numpy as np

def holes_count(grid: np.ndarray):
    """
    Given a binary grid, return its the holes in each column
    """
    cum_filled = np.cumsum(grid, axis=0) # Starts sum after first filled block
    hole_mask = (grid == 0) & (cum_filled > 0) # Masks the holes
    return np.sum(hole_mask, axis=0)

def calc_bumpiness(column_heights: np.ndarray):
    """
    Calculate the bumpiness of a certain grid given the column heights

    Args:
        column_heights (ndarray): Array describing the heights of each column in the grid
    """
    return np.abs(np.diff(column_heights)).sum()
    

def landing_height(block_positions: np.ndarray, height: int):
    """
    Calculates the landing height of a certain block.

    Args:
        block_positions: positions of all the blocks of the tetris piece
        height: height of the grid
    """
    
    # Grab all y-values and take mean
    y_vals = block_positions[:, 0]
    dist_from_bottom = (height - 1) - y_vals
    return float(dist_from_bottom.mean())

def cumulative_wells(grid: np.ndarray):
    """
    Counts all the wells in a grid. A 'well' is an empty cell whose left and right neighbors are grid

    Args:
        grid (ndarray): A binary grid
    """

    f = grid.astype(bool)
    height, width = f.shape

    # neighbors "filled or wall"
    left_filled  = np.pad(f[:, :-1], ((0,0),(1,0)), constant_values=True)
    right_filled = np.pad(f[:, 1:],  ((0,0),(0,1)), constant_values=True)
    well_cells = (~f) & left_filled & right_filled  # shape (H,W)

    # Sum triangular numbers of vertical runs per column
    total = 0
    for c in range(width):
        col = well_cells[:, c]
        # find run lengths of True segments
        # indices where runs start/stop
        if not col.any():
            continue
        # Convert to 0/1 for diff trick
        x = col.astype(np.int8)
        starts = np.flatnonzero(np.diff(np.pad(x, (1,1))) == 1)
        ends   = np.flatnonzero(np.diff(np.pad(x, (1,1))) == -1)
        runs = ends - starts  # lengths
        # triangular numbers: n*(n+1)//2
        total += int(np.sum(runs * (runs + 1) // 2))

    return total

def count_transitions(grid: np.ndarray):
    """
    Counts the row and column transitions in a binary grid. Adds imaginary borders at the end as it in 
    tetris feature engineering, a transition from a filled cell to out of bounds counts as a transition

    Args:
        grid (ndarray): A binary grid
    
    Returns:
        tuple: (row_transitions, col_transitions)
    """

    # Row transitions (pad L/R with zeros)
    plr = np.pad(grid, ((0,0),(1,1)), constant_values=0)
    row_transitions = int((plr[:, :-1] != plr[:, 1:]).sum())

    # Col transitions (pad T/B with zeros)
    ptb = np.pad(grid, ((1,1),(0,0)), constant_values=0)
    col_transitions = int((ptb[:-1, :] != ptb[1:, :]).sum())

    return row_transitions, col_transitions

def eroded_cells(lines_cleared, cleared_rows_idx, block_positions, height):
    """
    Eroded cells = (#lines_cleared) * (# of blocks of the placed tetromino that lie in those cleared lines).
    """

    if lines_cleared == 0 or len(cleared_rows_idx) == 0:
        return 0
    # bounds-safe mask for the 4 y's
    ys = block_positions[:, 1]
    in_bounds = (ys >= 0) & (ys < height)
    ys = ys[in_bounds]

    cleared_set = set(int(i) for i in cleared_rows_idx)
    touched = sum(int(y in cleared_set) for y in ys)    # how many piece blocks are in cleared rows
    return int(lines_cleared * touched)

def height_per_column(grid):
    """
    Return the aggregate height of a binary grid
    """

    rows, cols = grid.shape

    # Column heights
    first_filled_row = np.argmax(grid, axis=0)

    # Handle when column has no blocks
    has_block = grid.any(axis=0)
    col_heights = np.where(has_block, rows - first_filled_row, 0) # Correct heights

    return col_heights
