import sys, random, grader, parse

EAT_FOOD_SCORE = 10
PACMAN_EATEN_SCORE = -500
PACMAN_WIN_SCORE = 500
PACMAN_MOVING_SCORE = -1

def random_play_single_ghost(problem):
    # #Your p1 code here
    seed = int(problem['seed'])
    random.seed(seed, version = 1)
    gamemap = problem["game"]
    solution = f'seed: {seed}\n0\n'
    solution += '\n'.join(''.join(row) for row in gamemap) + '\n'
    wall = []
    food_store = []
    food_sign = ()
    pacman = ()
    ghost = ()
    position = ()
    score = 0
    turn = 1
    for x, row in enumerate(gamemap):
        for y, char in enumerate(row):
            if char == 'P':
                pacman = (x, y)
            elif char == 'W':
                ghost = (x, y)
            elif char == '%':
                wall.append((x, y))
            elif char == '.':
                food_store.append((x,y))
    #print(wall)
    while True:
        solution += "%d: " % turn
        directionchoice = ['N', 'S', 'W', 'E']
        # pacman round
        if turn % 2 == 1:
            if (pacman[0]-1, pacman[1]) in wall:
                directionchoice.remove('N')
            if (pacman[0]+1, pacman[1]) in wall:
                directionchoice.remove('S')
            if (pacman[0], pacman[1]-1) in wall:
                directionchoice.remove('W')
            if (pacman[0], pacman[1]+1) in wall:
                directionchoice.remove('E')
            #print("pacman direction choice:", directionchoice)
            # print(len(gamemap)) #row
            # print(len(gamemap[0])) #col
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
            if ghost == pacman:
                score += PACMAN_EATEN_SCORE
                gamemap[pacman[0]][pacman[1]] = 'W'
            solution += "P moving {}\n".format(direction)


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
            #print("ghost direction choice:", (directionchoice))
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
            #if gamemap[ghost[0]][ghost[1]] == 'P':
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
                solution += 'WIN: Ghost'
            else:
                solution += 'WIN: Pacman'
            break

    return solution

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 1
    grader.grade(problem_id, test_case_id, random_play_single_ghost, parse.read_layout_problem)