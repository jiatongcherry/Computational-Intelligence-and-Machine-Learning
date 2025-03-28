import sys, grader, parse

actions = {
    'N': (-1, 0),
    'E': (0, 1),
    'S': (1, 0),
    'W': (0, -1)
}

direction = {
    'x':'x',
    'N':['N', 'E', 'W'],
    'S':['S', 'W', 'E'],
    'W':['W', 'N', 'S'],
    'E':['E', 'S','N']
}

choices = ['N', 'W', 'E', 'S']

def initialize(problem):
    value_matrix = []
    policy_matrix = []
    for row in problem['grid']:
        new_row = []
        for cell in row:
            if cell == '#':
                new_row.append('#')
            elif isinstance(cell, (int, float)) or str(cell).lstrip('-').replace('.', '').isdigit():
                new_row.append('x')
            else:
                new_row.append('N')
        policy_matrix.append(new_row)

    for row in problem['grid']:
        new_row = []
        for cell in row:
            new_row.append('#' if cell == '#' else 0)
        value_matrix.append(new_row)



    return value_matrix, policy_matrix


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

def action_value(grid, value, living_reward, discount, position, action, multiplier):
    row, col = position
    new_row, new_col = row + actions[action][0], col + actions[action][1]

    if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]) and grid[new_row][new_col] != '#':
        return multiplier * (living_reward + discount * value[new_row][new_col])
    else:
        return multiplier * (living_reward + discount * value[row][col])

def Vfunc(grid, position, policy, noise, value, living_reward, discount):
    row, col = position
    action = policy[row][col]
    vvalue = 0

    if action == 'x':
        return grid[row][col]
    else:
        vvalue += action_value(grid, value, living_reward, discount, (row, col), action, 1 - noise * 2)

        for other_action in direction[action][1:]:
            vvalue += action_value(grid, value, living_reward, discount, (row, col), other_action, noise)

        return vvalue


def choice_func(grid, position, policy, value, noise, living_reward, discount):
    row, col = position
    V_k = float('-inf')
    step = ""

    if policy[row][col] == 'x':
        V_k = grid[row][col]
        step = 'x'
        return step, V_k

    for choice in choices:
        policy[row][col] = choice
        v_ = Vfunc(grid, (row, col), policy, noise, value, living_reward, discount)

        V_k, step = (v_, choice) if v_ > V_k else (V_k, step)

    policy[row][col] = step
    return step, V_k


def value_iteration(problem):

    # return_value = ''
    return_value = 'V_k=0\n'
    V_k = None
    step = None

    grid, noise, score, discount, iterations = problem['grid'], problem['noise'], problem['livingReward'], \
                                               problem['discount'], problem['iterations']
    value, policy = initialize(problem)

    return_value += switch(value)

    i = iterations
    for _ in range(i - 1):
        gridchange = [[0] * len(problem['grid'][0]) for _ in problem['grid']]

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                step, V_k = ("#", "#") if grid[row][col] == '#' else choice_func(grid, (row, col), policy, value,
                                                                                 noise, score,discount)
                policy[row][col] = step
                gridchange[row][col] = V_k

        i -= 1
        gap = iterations - i
        value = gridchange


        return_value += f"\nV_k={gap}\n"
        return_value += switch(value)
        return_value += f"\npi_k={gap}\n"

        strchange = [
            ''.join(f"| {val} |" if val != '#' else "| # |" for val in p)
            for p in policy
        ]
        return_value += '\n'.join(strchange)

    return return_value


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    #test_case_id = -4
    problem_id = 3
    grader.grade(problem_id, test_case_id, value_iteration, parse.read_grid_mdp_problem_p3)