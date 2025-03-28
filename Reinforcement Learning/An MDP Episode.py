import sys, grader, parse
from decimal import Decimal
import random

actions = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1), 'exit': (0, 0)}

direction = {
    'N':['N', 'E', 'W'],
    'S':['S', 'W', 'E'],
    'W':['W', 'N', 'S'],
    'E':['E', 'S','N']
}


def gridchange(grid, action, r, c):
    if action == 'exit':
        return [row[:] for row in grid]

    if action in actions:
        rr, cc = r + actions[action][0], c + actions[action][1]
        if 0 <= rr < len(grid) and 0 <= cc < len(grid[0]) and grid[rr][cc] != '#':
            grid = [row[:] for row in grid]
            grid[rr][cc] = 'P'
        else:
            grid = [row[:] for row in grid]
            grid[r][c] = 'P'
    return grid


def play_episode(problem):
    seed = problem['seed']
    if seed != -1:
        random.seed(seed, version=1)

    score = Decimal('0')
    noise = problem['noise']
    current_policy = ''
    experience = ''
    experience += "Start state:\n"

    grid = [row[:] for row in problem['grid']]

    initialcol, initialrow = next(((row.index('S'), i) for i, row in enumerate(grid) if 'S' in row), (None, None))

    if initialcol is not None and initialrow is not None:
        grid[initialrow][initialcol] = "P"


    format = '\n'.join(''.join(f"{j:>5}" for j in row) for row in grid)

    def cumulativescore(score):
        if float(score).is_integer():
            return f"{float(score):.1f}"
        return f"{score.normalize()}"

    def P_position(grid):
        for row_index in range(len(grid)):
            if 'P' in grid[row_index]:
                return row_index, grid[row_index].index('P')
        return None

    experience += format
    experience += f"\nCumulative reward sum: {cumulativescore(score)}\n"

    while True:
        row, col = P_position(grid)
        action = problem['policy'][row][col]

        if action == 'exit':
            current_policy = 'exit'
        else:
            current_policy = random.choices(population=direction[action], weights=[1 - noise * 2, noise, noise])[0]

        new_grid = gridchange(problem['grid'], current_policy, row, col)
        grid = new_grid

        experience += "-------------------------------------------- \n"

        if current_policy == 'exit':
            score += Decimal(problem['grid'][row][col])
            experience += f"Taking action: exit (intended: exit)\nReward received: " \
                          f"{float(problem['grid'][row][col])}\nNew state:\n"
            format = '\n'.join(''.join(f"{j:>5}" for j in row) for row in grid)
            experience += format
            experience += f"\nCumulative reward sum: {cumulativescore(score)}"
            break
        else:
            score += Decimal(str(problem['livingReward']))
            experience += f"Taking action: {current_policy} (intended: {action})\n"
            experience += f"Reward received: {problem['livingReward']}\nNew state:\n"
            format = '\n'.join(''.join(f"{j:>5}" for j in row) for row in grid)
            experience += format
            experience += f"\nCumulative reward sum: {cumulativescore(score)}\n"

    return experience

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    #test_case_id = 1
    problem_id = 1
    grader.grade(problem_id, test_case_id, play_episode, parse.read_grid_mdp_problem_p1)