import numpy as np
import csv
import argparse

COOLING_RATE = 0.99
NUM_ITERATIONS = 2000

def valid_rows(grid):
    for i in range(9):
        row = grid[i, :]
        non_zero_values = row[row != 0]
        if len(np.unique(non_zero_values)) != len(non_zero_values):
            return False
    return True

def valid_cols(grid):
    for i in range(9):
        col = grid[:, i]
        non_zero_values = col[col != 0]
        if len(np.unique(non_zero_values)) != len(non_zero_values):
            return False
    return True

def valid_blocks(grid):
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block = grid[i:i+3, j:j+3]
            non_zero_values = block[block != 0]
            if len(np.unique(non_zero_values)) != len(non_zero_values):
                return False
    return True

def valid_sudoku(grid):
    return valid_rows(grid) and valid_cols(grid) and valid_blocks(grid)

def display_grid(grid: np.ndarray):
    for i in range(9):
        if i % 3 == 0:
            print_border()
        for j in range(9):
            if j % 3 == 0:
                print('|', end=' ')

            if grid[i, j] == 0:
                print('.', end=' ')
            else:
                print(grid[i, j], end=' ')
        print('|')
    print_border()

def print_border():
    print('+---+---+---+---+---+---+')

def cost_function(grid: np.ndarray):
    cost = 0
    for i in range(9):
        cost += abs(9 - len(np.unique(grid[i, :]))) + abs(9 - len(np.unique(grid[:, i])))
    return cost

def swap_vals(grid, region, fixed_coords):
    region_x, region_y = region * 3
    block = np.copy(grid[region_x:region_x+3, region_y:region_y+3])

    unseen_coords = np.argwhere(block > 0)

    for coord in fixed_coords:
        block_coord = (coord[0]-region_x, coord[1]-region_y) 
        if block_coord in unseen_coords:
            index_to_remove = np.where((unseen_coords == block_coord).all(axis=1))[0]
            unseen_coords = np.delete(unseen_coords, index_to_remove, axis=0)

    swp_coords = unseen_coords[np.random.choice(len(unseen_coords), 2, replace=False)]

    block[tuple(swp_coords[0])], block[tuple(swp_coords[1])] = block[tuple(swp_coords[1])], block[tuple(swp_coords[0])]

    grid[region_x:region_x+3, region_y:region_y+3] = block


def generate_grid(grid):
    new_grid = grid.copy()
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            seen_vals = set(new_grid[i:i+3, j:j+3].flatten())
            missing_vals = list(set(range(1, 10)) - seen_vals)
            np.random.shuffle(missing_vals)
            for k in range(3):
                for l in range(3):
                    if new_grid[i+k, j+l] == 0:
                        new_grid[i+k, j+l] = missing_vals.pop()
    return new_grid

def init_temperature(grid):
    std_vals = []
    for _ in range(100):
        new_grid = generate_grid(grid)
        std_vals.append(cost_function(new_grid))
    return np.std(std_vals)

def fixed_coords(grid):
    return np.argwhere(grid > 0)

def solve(grid: np.ndarray):
    temperature = init_temperature(grid)
    grid_init = generate_grid(grid)

    fixed = fixed_coords(grid)

    while temperature > 0.1:
        for _ in range(NUM_ITERATIONS):
            region = np.random.randint(0, 3, 2)
            grid_new = grid_init.copy()

            swap_vals(grid_new, region, fixed)

            init_cost = cost_function(grid_init)
            new_cost = cost_function(grid_new)

            acceptance_probability = np.exp((init_cost-new_cost)/temperature)

            if acceptance_probability > np.random.rand():
                grid_init = grid_new
                if new_cost == 0:
                    return grid_new

        temperature *= COOLING_RATE

    return grid_init

def parse_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        grid = []
        for row in reader:
            grid.append(row)
    return np.array(grid, dtype=int)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sudoku Solver")
    parser.add_argument('file_path', type=str, help='Path to the input csv file')
    args = parser.parse_args()

    grid = parse_csv(args.file_path)

    display_grid(grid)
    assert valid_sudoku(grid), 'Invalid Input Sudoku'
    solved_grid = solve(grid)
    display_grid(solved_grid)
    assert valid_sudoku(solved_grid), 'Invalid Solved Sudoku'
