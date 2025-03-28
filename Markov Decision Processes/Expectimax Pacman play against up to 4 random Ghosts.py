import sys, parse
import time, os, copy
import random
from collections import deque

EAT_FOOD_SCORE = 10
PACMAN_EATEN_SCORE = -500
PACMAN_WIN_SCORE = 500
PACMAN_MOVING_SCORE = -1

move_directions = {
    "E": (0, 1),
    "W": (0, -1),
    "S": (1, 0),
    "N": (-1, 0)
}

def is_position_valid(position, wall, width, height):
    return (position not in wall and
            0 <= position[0] < height and
            0 <= position[1] < width)

#judge the possible direction, not crash on wall or other ghost
def find_direction(agent, problem):
    wall = problem['wall']
    map_width = problem['width']
    map_height = problem['height']
    ghost_positions = [problem['W'], problem['X'], problem['Y'], problem['Z']]
    pacman_position = problem['P']
    ghosts = ["W", "X", "Y", "Z"]
    ghost_index = 0
    valid_moves = {}

    if agent in ghosts:
        ghost_index = ghosts.index(agent)

    for direction, move in move_directions.items():
        #print(pacman_position)
        if agent == "P":
            new_position = (pacman_position[0]+move[0], pacman_position[1] + move[1])

            if is_position_valid(new_position, wall, map_width, map_height):
                valid_moves[direction] = (move[0], move[1])
        else:
            new_position = (ghost_positions[ghost_index][0] + move[0], ghost_positions[ghost_index][1] + move[1])
            combined_barriers = wall + ghost_positions[:ghost_index] + ghost_positions[ghost_index + 1:]

            if is_position_valid(new_position, combined_barriers, map_width, map_height):
                valid_moves[direction] = (move[0], move[1])

    return valid_moves

#build a bfs tree, on every state, count the distance to the target(ghost W or X or... or food)
def bfs(map_size, wall, start, goal):
    width, height = map_size
    #print(width, height)

    #hash table  ->  save time
    visited = set()
    visited.add(start)
    queue = deque([(start, 0)])
    distance = float('inf')


    while queue:
        current, distance = queue.popleft()
        if current == goal:
            return distance

        #if one position is not visited, and is valid, then store it into the tree
        for direction in move_directions:
            #print(direction)
            position = (current[0] + move_directions[direction][0], current[1] + move_directions[direction][1])
            if is_position_valid(position, wall, width, height):
                if position not in visited:
                    visited.add(position)
                    queue.append((position, distance + 1))

    return distance

#decide score of current state
def evaluate_function(problem):

    # sigmoid func， Provide smooth feedback
    # allowing the score to gradually change with the distance
    def sigmoid(distance, k=0.1, x0=5):
        return 1 / (1 + 2.71828 * (-k * (distance - x0)))

    walls = set(tuple(w) for w in problem['wall'])
    map = (problem['width'], problem['height'])

    #no food -- pacman win
    #in ghost position -- pacman lose
    if not problem['food_store']:
        return problem['score'] + PACMAN_WIN_SCORE
    if problem['P'] in [problem['W'], problem['X'], problem['Y'], problem['Z']]:
        return -float('inf')


    # ghost distance
    for ghost in ['W', 'X', 'Y', 'Z']:
        if problem[ghost]:
            ghost_distance = bfs(map, walls, problem['P'], problem[ghost])
            penalty = sigmoid(ghost_distance)  # be closer, higher penalty
            problem['score'] -= 1000 * penalty  # be closer, less score

    #food distance
    if problem['food_store']:
        closest_food_distance = min(bfs(map, walls, problem['P'], food) for food in problem['food_store'])
        reward = sigmoid(closest_food_distance)   # be closer, higher reward
        problem['score'] += 10 * reward  # be closer, higher score

    return problem['score']

#decide next move direction according to the score
def expectimax(problem, k, num_agents,whichagent, recent_moves):
    if k == 0 or problem['P'] in [problem['W'], problem['X'], problem['Y'], problem['Z']] or len(problem['food_store']) == 0:
        return evaluate_function(problem), None

    best_moves = []
    expected_score = 0
    agents = ['P', 'W', 'X', 'Y', 'Z']
    agent = agents[whichagent]
    best_score = -float('inf') if agent == 'P' else float('inf')#decide whose turn and the initial score


    # 4 directions, iteration to decide best move
    for direction, position in find_direction(agent, problem).items():
        new_problem = problem.copy()
        new_problem[agent] = (new_problem[agent][0] + position[0], new_problem[agent][1] + position[1])

        if agent == 'P' and position in new_problem['food_store']:
            new_problem['food_store'].remove(position)
            new_problem['score'] += EAT_FOOD_SCORE
        next_agent_index = (whichagent + 1) % num_agents

        result = expectimax(new_problem, k - 1, num_agents, next_agent_index, recent_moves)
        score = result[0]

        # Check for repeated moves
        recent_moves[agent].append((new_problem[agent][0], new_problem[agent][1]))
        if len(recent_moves[agent]) > 10:
            recent_moves[agent].pop(0)  # Keep only the last 10 moves

        if len(set(recent_moves[agent])) == 1:  # All moves are the same
            # Randomly choose a less favorable move
            valid_moves = find_direction(agent, problem)
            if valid_moves:
                best_move = random.choice(list(valid_moves.keys()))
                return best_score, best_move

        if agent == 'P':
            if score >= best_score:
                if score > best_score:
                    best_moves = [direction]
                    best_score = score
                else:
                    best_moves.append(direction)

            # randomness
            if best_moves:
                best_move = random.choice(best_moves)


        else:
            actions = find_direction(agent, problem)

            if len(actions) == 0:
                return evaluate_function(problem), None

            new_problem[agent] = (new_problem[agent][0] + position[0], new_problem[agent][1] + position[1])
            next_agent_index = (whichagent + 1) % num_agents
            score, _ = expectimax(new_problem, k - 1,  num_agents,next_agent_index, recent_moves)
            expected_score += score

            expected_score /= len(actions)


    if agent == 'P':
        return best_score, best_move
    else:
        return expected_score, []
    # print("agent and best move", agent, best_move)
    # input()

#analyse features of problem, including number of ghost, the position of pacman and ghost etc.
#cuz the predefined problem lose some features
def problem_analyse(problem):
    gamemap = problem["game"]
    height = len(gamemap) - 2
    width = len(gamemap[1]) - 2
    pacman = None
    W = None
    X = None
    Y = None
    Z = None
    score = 0
    food_store = []
    wall = []


    for x, row in enumerate(gamemap):
        for y, char in enumerate(row):
            if char == 'P':
                pacman = (x, y)
            elif char == 'W':
                W = (x, y)
            elif char == 'X':
                X = (x, y)
            elif char == 'Y':
                Y = (x, y)
            elif char == 'Z':
                Z = (x, y)
            elif char == '%':
                wall.append((x, y))
            elif char == '.':
                food_store.append((x, y))

    analyse_problem = {
        "width": width,
        "height": height,
        "P": pacman,
        "W": W,
        "X": X,
        "Y": Y,
        "Z": Z,
        "food_store": food_store,
        "wall": wall,
        "score": score
    }

    return analyse_problem

def expecti_max_multiple_ghosts(problem, k):
    #Your p6 code here
    solution = ""
    agent_num = 1
    ghost_order = ['W', 'X', 'Y', 'Z']
    problem = problem_analyse(problem)
    recent_moves = {agent: [] for agent in ['P', 'W', 'X', 'Y', 'Z']}  # Track recent moves
    ghostpos = {"W": problem['W'], "X": problem['X'],
                "Y": problem['Y'], "Z": problem['Z']}
    #print(problem)

    for ghost in ghost_order:
        if problem[ghost]:
            agent_num += 1


    while True:
        # pacman round
        action_list = find_direction("P", problem)

        if action_list:
            move = expectimax(problem, k,  agent_num, 0, recent_moves)[1]

            problem['P'] = (problem['P'][0]+action_list[move][0], problem['P'][1]+action_list[move][1])


        if problem['P'] in problem['food_store']:
            problem['score'] += EAT_FOOD_SCORE
            problem['food_store'].remove(problem['P'])

        if not problem['food_store']:
            winner = 'Pacman'
            problem['score'] += PACMAN_WIN_SCORE
            return solution, winner

        if any(problem['P'] == ghostpos for ghostpos in (problem['W'], problem['X'], problem['Y'], problem['Z'])):
            winner = 'Ghost'
            problem['score'] += PACMAN_EATEN_SCORE
            return solution, winner



        #ghost round
        for ghost in ghost_order:
            if ghostpos[ghost]:
                action_list = find_direction(ghost, problem)
                if action_list:
                    #random ghost
                    move = random.choice(sorted(action_list.keys()))
                    #print(move)

                    ghostpos[ghost] = (ghostpos[ghost][0]+action_list[move][0], ghostpos[ghost][1]+action_list[move][1])
                    problem['W'] = ghostpos['W']
                    problem['X'] = ghostpos['X']
                    problem['Y'] = ghostpos['Y']
                    problem['Z'] = ghostpos['Z']


            if problem['P'] == ghostpos[ghost]:
                problem['score'] += PACMAN_EATEN_SCORE
                winner = 'Ghost'
                return solution, winner


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])    
    problem_id = 6
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join('test_cases','p'+str(problem_id)) 
    problem = parse.read_layout_problem(os.path.join(path,file_name_problem))
    k = int(sys.argv[2])
    num_trials = int(sys.argv[3])
    verbose = bool(int(sys.argv[4]))
    print('test_case_id:',test_case_id)
    print('k:',k)
    print('num_trials:',num_trials)
    print('verbose:',verbose)
    start = time.time()
    win_count = 0
    for i in range(num_trials):
        solution, winner = expecti_max_multiple_ghosts(copy.deepcopy(problem), k)
        if winner == 'Pacman':
            win_count += 1
        if verbose:
            print(solution)
    win_p = win_count/num_trials * 100
    end = time.time()
    print('time: ',end - start)
    print('win %',win_p)