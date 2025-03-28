import sys, parse
import time, os, copy
import random

EAT_FOOD_SCORE = 10
PACMAN_EATEN_SCORE = -500
PACMAN_WIN_SCORE = 500
PACMAN_MOVING_SCORE = -1

def Manhattan_distance(po1, po2):
    return abs(po1[0] - po2[0]) + abs(po1[1] - po2[1])

def evaluation_function(pacman, ghost, direction, food_store):
    x, y = pacman
    position = ()
    if direction == 'N':
        position = (x - 1, y)
    elif direction == 'S':
        position = (x + 1, y)
    elif direction == 'W':
        position = (x, y - 1)
    elif direction == 'E':
        position = (x, y + 1)

    ghost_distance = Manhattan_distance(position, ghost)
    #closest food
    food_distance = min(Manhattan_distance(position, f) for f in food_store) if food_store else float('inf')

    #if pacman is closed to ghost, far from food, this value will decrease
    #if pacman is closed to food, far from ghost, this value will increase
    #we encourage this value to be larger
    return 0.4*ghost_distance - 0.6*food_distance

def better_play_single_ghosts(problem):
    #Your p2 code here
    gamemap = problem["game"]
    solution = '\n'.join(''.join(row) for row in gamemap) + '\n'
    wall = []
    food_store = []
    food_sign = ()
    pacman = ()
    ghost = ()
    position = ()
    score = 0
    turn = 1
    last_position = None
    loop_count = 0
    winner = ''

    for x, row in enumerate(gamemap):
        for y, char in enumerate(row):
            if char == 'P':
                pacman = (x, y)
            elif char == 'W':
                ghost = (x, y)
            elif char == '%':
                wall.append((x, y))
            elif char == '.':
                food_store.append((x, y))
    # print(wall)
    while True:
        best_direction = None
        best_score = -float('inf')
        solution += "%d: " % turn
        directionchoice = ['N', 'S', 'W', 'E']

        # pacman round
        if turn % 2 == 1:
            if (pacman[0] - 1, pacman[1]) in wall:
                directionchoice.remove('N')
            if (pacman[0] + 1, pacman[1]) in wall:
                directionchoice.remove('S')
            if (pacman[0], pacman[1] - 1) in wall:
                directionchoice.remove('W')
            if (pacman[0], pacman[1] + 1) in wall:
                directionchoice.remove('E')
            # print("pacman direction choice:", directionchoice)
            # print(len(gamemap)) #row
            # print(len(gamemap[0])) #col


            #select best direction based on the score calculated by evaluation_function
            for direction in directionchoice:
                score = evaluation_function(pacman, ghost, direction, food_store)
                if score > best_score:
                    best_score = score
                    best_direction = direction

            x, y = pacman
            if best_direction == 'N':
                position = (x - 1, y)
            elif best_direction == 'S':
                position = (x + 1, y)
            elif best_direction == 'W':
                position = (x, y - 1)
            elif best_direction == 'E':
                position = (x, y + 1)

            #for q5, avoid infinite loop
            if last_position is not None and pacman == last_position:
                loop_count += 1
            else:
                loop_count = 0
            if loop_count > 10000:
                best_direction = random.choice(directionchoice)
                loop_count = 0


            gamemap[x][y] = ' '
            pacman = position
            score += PACMAN_MOVING_SCORE
            # judge whether there is a food on that grid
            if gamemap[pacman[0]][pacman[1]] == '.':
                score += EAT_FOOD_SCORE
                if (pacman[0], pacman[1]) in food_store:
                    food_store.remove((pacman[0], pacman[1]))

            # change the position of pacman on map and revise the score
            gamemap[pacman[0]][pacman[1]] = 'P'
            if ghost == pacman:
                score += PACMAN_EATEN_SCORE
                gamemap[pacman[0]][pacman[1]] = 'W'
            solution += "P moving {}\n".format(best_direction)

        # ghost round
        if turn % 2 == 0:
            if (ghost[0] - 1, ghost[1]) in wall:
                directionchoice.remove('N')
            if (ghost[0] + 1, ghost[1]) in wall:
                directionchoice.remove('S')
            if (ghost[0], ghost[1] - 1) in wall:
                directionchoice.remove('W')
            if (ghost[0], ghost[1] + 1) in wall:
                directionchoice.remove('E')

            # print("ghost direction choice:", (directionchoice))
            direction = random.choice(sorted(directionchoice))
            x, y = ghost
            if direction == 'N':
                position = (x - 1, y)
            elif direction == 'S':
                position = (x + 1, y)
            elif direction == 'W':
                position = (x, y - 1)
            elif direction == 'E':
                position = (x, y + 1)

            gamemap[x][y] = ' '
            ghost = position

            # if gamemap[ghost[0]][ghost[1]] == 'P':
            if ghost == pacman:
                score += PACMAN_EATEN_SCORE
            if food_sign:
                gamemap[food_sign[0]][food_sign[1]] = '.'
            if (ghost[0], ghost[1]) in food_store:
                food_sign = ghost
            gamemap[ghost[0]][ghost[1]] = 'W'
            solution += "W moving {}\n".format(direction)

        if not food_store:
            score += PACMAN_WIN_SCORE

        solution += '\n'.join(''.join(row) for row in gamemap)
        solution += f"\nscore: {score}\n"
        turn += 1

        if not food_store or ghost == pacman:
            if ghost == pacman:
                winner = 'Ghost'
            else:
                winner = 'Pacman'
            break

        # result = ''
        # for row in gamemap:
        #     result += ''.join(row) + '\n'
        #
        # print(result)
        # input()

    return solution, winner


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])    
    problem_id = 2
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
        solution, winner = better_play_single_ghosts(copy.deepcopy(problem))
        if winner == 'Pacman':
            win_count += 1
        if verbose:
            print(solution)
    win_p = win_count/num_trials * 100
    end = time.time()
    print('time: ',end - start)
    print('win %',win_p)