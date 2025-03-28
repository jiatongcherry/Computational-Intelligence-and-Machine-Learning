import sys, random, grader, parse

EAT_FOOD_SCORE = 10
PACMAN_EATEN_SCORE = -500
PACMAN_WIN_SCORE = 500
PACMAN_MOVING_SCORE = -1

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

def random_play_multiple_ghosts(problem):
    seed = int(problem['seed'])
    random.seed(seed, version=1)
    gamemap = problem["game"]
    solution = f'seed: {seed}\n0\n'
    solution += '\n'.join(''.join(row) for row in gamemap) + '\n'

    wall = []
    food_store = []
    pacman = ()
    ghosts = {}
    position = ()
    score = 0
    turn = 1
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

            direction = random.choice(sorted(directionchoice))
            x, y = pacman
            if direction == 'N':
                position = (x - 1, y)
            elif direction == 'S':
                position = (x + 1, y)
            elif direction == 'W':
                position = (x, y - 1)
            elif direction == 'E':
                position = (x, y + 1)

            gamemap[x][y] = ' '
            pacman = position
            score += PACMAN_MOVING_SCORE
            # judge whether there is a gift on that grid
            if gamemap[pacman[0]][pacman[1]] == '.':
                score += EAT_FOOD_SCORE
                if (pacman[0], pacman[1]) in food_store:
                    food_store.remove((pacman[0], pacman[1]))
            # change the position of pacman on map and revise the score
            gamemap[pacman[0]][pacman[1]] = 'P'

            solution += "%d: " % turn
            turn += 1
            solution += "P moving {}\n".format(direction)

            for ghost_name, ghost_position in ghosts.items():
                if ghost_position == pacman:
                    score += PACMAN_EATEN_SCORE
                    gamemap[pacman[0]][pacman[1]] = ghost_name
                    solution += '\n'.join(''.join(row) for row in gamemap)
                    solution += f"\nscore: {score}\n"
                    solution += f"WIN: Ghost"
                    return solution

            if not food_store:
                score += PACMAN_WIN_SCORE
                solution += '\n'.join(''.join(row) for row in gamemap)
                solution += f"\nscore: {score}\n"
                solution += 'WIN: Pacman'
                return solution

            solution += '\n'.join(''.join(row) for row in gamemap)
            solution += f"\nscore: {score}\n"




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
                solution += "%d: " % turn
                turn += 1
                solution += "{} moving \n".format(ghost_name)
                solution += '\n'.join(''.join(row) for row in gamemap)
                solution += f"\nscore: {score}\n"
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
            solution += "%d: " % turn
            turn += 1
            solution += "{} moving {}\n".format(ghost_name, direction)
            solution += '\n'.join(''.join(row) for row in gamemap)

            if ghosts[ghost_name] == pacman:
                score += PACMAN_EATEN_SCORE
                solution += f"\nscore: {score}\n"
                solution += "WIN: Ghost"
                return solution

            solution += f"\nscore: {score}\n"






if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 3
    grader.grade(problem_id, test_case_id, random_play_multiple_ghosts, parse.read_layout_problem)