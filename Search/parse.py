import os, sys
def read_graph_search_problem(file_path):
    #Your p1 code here
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_state = lines[0].split(': ')[1].strip()
    goal_line = lines[1].split(': ')[1].strip()
    heuristic = {}
    edges = {}

    goal_states = goal_line.split()
    for line in lines[2:]:
        if line.strip():
            tokens = line.split(' ')
            if len(tokens) == 2:
                state = tokens[0]
                cost = int(tokens[1])
                heuristic[state] = cost

            if len(tokens) == 3:
                source = tokens[0]
                target = tokens[1]
                cost = float(tokens[2])
                if source in edges:
                    edges[source].append((target, cost))
                else:
                    edges[source] = [(target, cost)]

    problem = {
        'start_state': start_state,
        'goal_states': goal_states,
        'edges': edges,
        'heuristic': heuristic
    }
    #print(problem)
    return problem

def read_8queens_search_problem(file_path):
    #Your p6 code here
    # Initialize the chessboard configuration
    with open(file_path, 'r') as file:
        lines = file.readlines()

    chessboard = []
    for line in lines:
        row = line.strip().split()
        chessboard.append(row)
    #print(chessboard)
    return chessboard


if __name__ == "__main__":
    #print(sys.argv)
    if len(sys.argv) == 3:
        problem_id, test_case_id = sys.argv[1], sys.argv[2]
        if int(problem_id) <= 5:
            problem = read_graph_search_problem(os.path.join('test_cases','p'+problem_id, test_case_id+'.prob'))
        else:
            problem = read_8queens_search_problem(os.path.join('test_cases','p'+problem_id, test_case_id+'.prob'))
        print(problem)
    else:
        print('Error: I need exactly 2 arguments!')