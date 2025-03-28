import sys, parse
import time, os, copy
import random
from collections import deque

#if crash, return false, otherwise retrn true
def free_check(direction, ghost, ghosts):
    position = ()
    if direction == 'N':
        position = (ghost[0] - 1, ghost[1])
    elif direction == 'S':
        position = (ghost[0] + 1, ghost[1])
    elif direction == 'W':
        position = (ghost[0], ghost[1] - 1)
    elif direction == 'E':
        position = (ghost[0], ghost[1] + 1)
    return position not in ghosts.values()


def eval_state_action(pacman, ghosts, action, food_store):
    food_distances = []
    pacman_x, pacman_y = pacman
    ghost_positions = [(g[0], g[1]) for g in ghosts.values()]

    # 更新 Pacman 的位置
    if action == 'E':
        pacman_y += 1
    elif action == 'N':
        pacman_x -= 1
    elif action == 'W':
        pacman_y -= 1
    elif action == 'S':
        pacman_x += 1

    # if encounter a ghost, return negative score
    if (pacman_x, pacman_y) in ghost_positions:
        return -float('inf')

    #calculate the score of ghost and food to pacman
    #multiple ghost, should combine min and average distance
    ghost_distance_scores = [
        abs(pacman_x - ghost_x) + abs(pacman_y - ghost_y) for ghost_x, ghost_y in ghost_positions
    ]
    min_ghost_distance = min(ghost_distance_scores)
    avg_ghost_distance = sum(ghost_distance_scores) / len(ghost_positions)
    ghost_score = 1 / (min_ghost_distance + avg_ghost_distance)

    food_distances = [abs(pacman_y - food[1]) + abs(pacman_x - food[0]) for food in food_store]

    #if no food, give food distance a default value
    if not food_distances:
        food_distances.append(1)


    action_score = 0
    if min(food_distances) == 0:
        action_score += 5
    else:
        action_score += 1 / (min(food_distances) + 1)

    total_score = action_score - ghost_score
    return total_score

def choose_best_action(pacman, ghosts, action_space, food_store):
    best_action = action_space[0]
    best_score = -float('inf')
    for action in action_space:
        score = eval_state_action(pacman, ghosts, action, food_store)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action

def better_play_multiple_ghosts(problem):
    #Your p4 code here
    seed = int(problem['seed'])
    random.seed(seed, version=1)
    gamemap = problem["game"]
    # solution = f'seed: {seed}\n0\n'
    # solution += '\n'.join(''.join(row) for row in gamemap) + '\n'

    wall = []
    food_store = []
    pacman = ()
    ghosts = {}
    position = ()
    score = 0
    win = None
    solution = ''
    ghost_order = ['W', 'X', 'Y', 'Z']

    for x, row in enumerate(gamemap):
        for y, char in enumerate(row):
            if char == 'P':
                pacman = (x, y)
            elif char in 'WXYZ':
                ghosts[char] = (x, y)
            elif char == '%':
                wall.append((x, y))
            elif char == '.':
                food_store.append((x, y))

    while True:
        best_direction = None
        best_score = -float('inf')
        directionchoice = ['N', 'S', 'W', 'E']

        #pacman round
        if 1:
            if (pacman[0]-1, pacman[1]) in wall:
                directionchoice.remove('N')
            if (pacman[0]+1, pacman[1]) in wall:
                directionchoice.remove('S')
            if (pacman[0], pacman[1]-1) in wall:
                directionchoice.remove('W')
            if (pacman[0], pacman[1]+1) in wall:
                directionchoice.remove('E')

            # select best direction based on the score calculated by evaluation_function
            best_direction = choose_best_action(pacman, ghosts, directionchoice, food_store)

            x, y = pacman
            if best_direction == 'N':
                position = (x - 1, y)
            elif best_direction == 'S':
                position = (x + 1, y)
            elif best_direction == 'W':
                position = (x, y - 1)
            elif best_direction == 'E':
                position = (x, y + 1)

            gamemap[x][y] = ' '
            pacman = position

            # judge whether there is a gift on that grid
            if gamemap[pacman[0]][pacman[1]] == '.':

                if (pacman[0], pacman[1]) in food_store:
                    food_store.remove((pacman[0], pacman[1]))
            # change the position of pacman on map and revise the score
            gamemap[pacman[0]][pacman[1]] = 'P'

            #solution += "P moving {}\n".format(best_direction)

            for ghost_name, ghost_position in ghosts.items():
                if ghost_position == pacman:
                    gamemap[pacman[0]][pacman[1]] = ghost_name
                    #solution += '\n'.join(''.join(row) for row in gamemap)

                    win = 'Ghost'
                    #print(solution)
                    return solution, win

            if not food_store:
                #solution += '\n'.join(''.join(row) for row in gamemap)
                win = 'Pacman'
                #print(solution)
                return solution, win

            #solution += '\n'.join(''.join(row) for row in gamemap)



        #ghost round
        for ghost_name in ghost_order:
            if ghost_name not in ghosts:
                continue
            ghost = ghosts[ghost_name]

            directionchoice = ['N', 'S', 'W', 'E']

            if (ghost[0] - 1, ghost[1]) in wall:
                directionchoice.remove('N')
            if (ghost[0] + 1, ghost[1]) in wall:
                directionchoice.remove('S')
            if (ghost[0], ghost[1] - 1) in wall:
                directionchoice.remove('W')
            if (ghost[0], ghost[1] + 1) in wall:
                directionchoice.remove('E')

            temp = []
            for direction in directionchoice:
                if free_check(direction, ghost, ghosts):
                    temp.append(direction)
            if not temp:
                # solution += "{} moving \n".format(ghost_name)
                # solution += '\n'.join(''.join(row) for row in gamemap)
                continue

            #print(turn, ':', ghosts)

            direction = random.choice(sorted(temp))

            # print('ghost name', ghost_name)
            # print('direction', direction)


            x, y = ghost
            if direction == 'N':
                position = (x - 1, y)
            elif direction == 'S':
                position = (x + 1, y)
            elif direction == 'W':
                position = (x, y - 1)
            elif direction == 'E':
                position = (x, y + 1)

            #if this ghost pass the free check
            #solution has the output if free check is passed
            ghosts[ghost_name] = position
            gamemap[x][y] = ' '
            ghost = position

            for food in food_store:
                if (food[0], food[1]) not in ghosts.values():
                    gamemap[food[0]][food[1]] = '.'

            gamemap[ghost[0]][ghost[1]] = '{}'.format(ghost_name)
            # solution += "{} moving {}\n".format(ghost_name, direction)
            # solution += '\n'.join(''.join(row) for row in gamemap)

            if ghosts[ghost_name] == pacman:
                win = "Ghost"
                #print(solution)
                return solution, win



if __name__ == "__main__":
    test_case_id = int(sys.argv[1])    
    problem_id = 4
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join('test_cases','p'+str(problem_id)) 
    problem = parse.read_layout_problem(os.path.join(path,file_name_problem))
    num_trials = int(sys.argv[2])
    verbose = bool(int(sys.argv[3]))
    print('test_case_id:',test_case_id)
    print('num_trials:',num_trials)
    print('verbose:',verbose)
    start = time.time()
    win_count = 0
    for i in range(num_trials):
        solution, winner = better_play_multiple_ghosts(copy.deepcopy(problem))
        if winner == 'Pacman':
            win_count += 1
        if verbose:
            print(solution)
    win_p = win_count/num_trials * 100
    end = time.time()
    print('time: ',end - start)
    print('win %',win_p)