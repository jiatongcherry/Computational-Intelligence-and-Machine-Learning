import sys, grader, parse
import random

actions = {
    'N': (-1, 0),
    'E': (0, 1),
    'S': (1, 0),
    'W': (0, -1),
    'exit': (0, 0)
}

direction = {
    'N':['N', 'E', 'W'],
    'S':['S', 'W', 'E'],
    'W':['W', 'N', 'S'],
    'E':['E', 'S','N'],
    'exit':'exit'
}

#judgement
def action_value(grid, value, living_reward, discount, position, action, multiplier):
    row = position[0]
    col = position[1]
    new_row, new_col = row + actions[action][0], col + actions[action][1]

    if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]) and grid[new_row][new_col] != '#':
            return multiplier * (living_reward + discount * value[new_row][new_col])
    else:
        return multiplier * (living_reward + discount * value[row][col])

def Vfunc(matrix, grid, policy, noise, living_reward, discount, position):
    row, col = position
    action = policy[row][col]
    vvalue = 0
    if action == 'exit':
        return grid[row][col]
    else:
        vvalue += action_value(grid, matrix, living_reward, discount, (row,col), action,1 - noise * 2)

        for action_offset in direction[action][1:]:
            vvalue += action_value(grid, matrix, living_reward, discount, (row,col), action_offset, noise)

    return vvalue

#string operation
def switch(matrix):
    output = []
    switch_matrix = ''
    for row in matrix:
        string = ''
        for value in row:
            if value == '#':
                string += "| ##### |"
            else:
                string += f"| {value:6.2f}|"
        output.append(string)

    switch_matrix = '\n'.join(output)

    return switch_matrix


def policy_evaluation(problem):
    matrix = []
    #initialize
    return_value = "V^pi_k=0\n"
    grid, policy, noise, score, discount, iterations = problem['grid'], problem['policy'], problem['noise'], \
                                                       problem['livingReward'], problem['discount'], problem['iterations']

    for row in problem['grid']:
        new_row = []
        for i in row:
            if i == '#':
                new_row.append('#')
            else:
                new_row.append(0)
        matrix.append(new_row)


    return_value += switch(matrix)

    while iterations > 1:
        current_matrix = []
        for row in problem['grid']:
            new_row = []
            for cell in row:
                if cell == '#':
                    new_row.append('#')
                else:
                    new_row.append(0)
            current_matrix.append(new_row)

        for row in range(len(problem['grid'])):
            for col in range(len(problem['grid'][0])):
                if problem['grid'][row][col] == '#':
                    current_matrix[row][col] = "#"
                else:
                    current_matrix[row][col] = Vfunc(matrix, grid, policy, noise, score, discount, (row, col))

        matrix = current_matrix

        iterations = iterations - 1
        return_value += f"\nV^pi_k="
        return_value += f"{problem['iterations'] - iterations}\n"
        return_value += switch(matrix)

    return return_value

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    #test_case_id = -7
    problem_id = 2
    grader.grade(problem_id, test_case_id, policy_evaluation, parse.read_grid_mdp_problem_p2)