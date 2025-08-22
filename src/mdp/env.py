import numpy as np
import copy
import mdp.utils as utils
import mdp.observations as obs_terms

def clear_rows(grid):
    cleared_row_idx = np.where(np.all(grid == '#', axis=1))[0]
    
    # Remove any full rows
    # Statement literally gets all indices with full rows, and indexes the grid without those indices
    cleared_grid = grid[~np.all(grid == '#', axis=1)]

    # Count lines cleared
    lines_cleared = grid.shape[0] - cleared_grid.shape[0]

    # Add empty rows at top
    if lines_cleared > 0:
        new_rows = np.full((lines_cleared, cleared_grid.shape[1]), fill_value='.', dtype=cleared_grid.dtype)
        cleared_grid = np.vstack((new_rows, cleared_grid))

    grid = cleared_grid

    return lines_cleared, cleared_row_idx

def land( grid: np.ndarray, block):
    """
    Modified fall function that lands the block immediately.
    Returns the landed block positions
    """
    block_type = block.type
    block_positions = np.array(block.get_pixel_pos())
    grid[block_positions[:, 0], block_positions[:, 1]] = '#'
    
    return block_positions, block_type

def _get_observation(grid, block_positions, block_type, lines_cleared=0, cleared_row_idx=np.array([])):
    """
    Returns the visual part of the grid, denoting '.' as 0, '#' as .5, and '$' as 1

    Also contains various other observations like falling block type, rotation, position
    """
    rows, cols = grid.shape
    bin_grid = utils.bin_grid(grid)

    # Get all terms (normalize if need be)
    block_type = utils.one_hot(7, block_type)
    col_heights = obs_terms.height_per_column(bin_grid)
    col_heights_norm = col_heights / 20
    holes_per_col = (obs_terms.holes_count(bin_grid) / col_heights if col_heights.all() > 0 else np.zeros(10))
    bumpiness = obs_terms.calc_bumpiness(col_heights) / (rows * (cols - 1))
    row_transitions, col_transitions = obs_terms.count_transitions(bin_grid)
    row_transitions /= (rows * (cols + 1))
    col_transitions /= ((rows + 1) * cols)
    eroded_cells = obs_terms.eroded_cells(lines_cleared, cleared_row_idx, block_positions, rows) / 16 # Max number of eroded cells possible
    num_wells = obs_terms.cumulative_wells(bin_grid) / (cols * (rows * (rows + 1) // 2)) # Normalized by max number of wells
    landing_height = obs_terms.landing_height(block_positions, rows) / rows

    # Return observation concatenated altogether
    return np.concatenate([
        block_type.astype(np.float32),
        col_heights_norm.astype(np.float32),
        holes_per_col.astype(np.float32),
        np.array([bumpiness], dtype=np.float32),
        np.array([row_transitions], dtype=np.float32),
        np.array([col_transitions], dtype=np.float32),
        np.array([eroded_cells], dtype=np.float32),
        np.array([num_wells], dtype=np.float32),
        np.array([landing_height], dtype=np.float32),
    ])


def _get_reward(lines_cleared, obs):
    """
    Distribute the reward after a block lands (counts as one step in this environment),
    """
    reward = 1.0
    reward += 35 * (lines_cleared ** 2)

    return reward

def get_possible_obs(grid: np.ndarray, block):
    """
    For each possible action, return the observation once the action has been made.
    The index in the observations array correspond to the action made
    """
    observations = np.array([])

    # Actions 0 - 39
    for i in range(40):
        working_grid = grid.copy()
        working_block = copy.deepcopy(block)

        if not fall(i, working_grid, working_block):
            observations[i] = _get_observation(np.array(working_block.get_pixel_pos()), working_block.type)
            continue
        
        landed_block_positions, landed_block_type = land(working_grid, working_block)
        lines_cleared, cleared_row_idx = clear_rows(working_grid)

        observations[i] = _get_observation(block_positions=landed_block_positions, block_type=landed_block_type, lines_cleared=lines_cleared, cleared_row_idx=cleared_row_idx)

    return observations

def fall(action, grid: np.ndarray, block):
    """
    Helper function to position the block correctly based on action and "place" the block into the fallen position

    Returns false if terminated
    """

    rows, cols = grid.shape

    # Parse action information: Tens digit is rotation, Ones digit is column number
    action_temp = str(action)
    rotation = int(action_temp[0]) if len(action_temp) > 1 else 0
    column = int(action_temp[1]) if len(action_temp) > 1 else int(action_temp[0])
    
    # Position the block
    block.rotation = rotation
    block.pos = (block.pos[0], column)

    # Handle case: block goes out of bound if placed in certain column
    block_position_cols = np.array(block.get_pixel_pos(), dtype=int)[:, 1]
    overflow = min(0, block_position_cols.min()) + max(0, block_position_cols.max() - (cols - 1))
    block.pos = (block.pos[0], int(block.pos[1] - overflow))

    # Update block position
    block_positions = np.array(block.get_pixel_pos())

    # Get highest row pos for each column the block is in
    cols = block_positions[:,  1]
    unique_cols = np.unique(cols)
    idx = [np.where(cols == c)[0][np.argmax(block_positions[cols == c, 0])] for c in unique_cols]
    unique_pos = block_positions[idx]

    # Get col heights (inverted)
    selected_columns = grid[:, unique_cols[0]:unique_cols[-1] + 1] == '#'
    col_heights = np.where(np.any(selected_columns, axis=0), np.argmax(selected_columns, axis=0), rows)

    # If any blocks collide with already landed ones, terminate
    if np.any(grid[block_positions[:, 0], block_positions[:, 1]] == '#'):
        return False
    
    dy = min(col_heights - unique_pos[:, 0]) - 1
    block.pos = (int(block.pos[0] + dy), block.pos[1])
    return True