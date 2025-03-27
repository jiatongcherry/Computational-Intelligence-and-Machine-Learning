import sys, grader, parse
import collections

#BFS-GSA
def bfs_search(problem):
    #Your p2 code here
    global path
    start_state = problem['start_state']
    goal_state = problem['goal_states']

    frontier = collections.deque([start_state])
    exploredlist = list()  # GSA need exploredset
    while frontier:
        path = frontier.popleft()
        cur = path.split(' ')[-1]
        # edge check
        if cur in goal_state:
            break

        if (cur not in exploredlist):
            exploredlist.append(cur)
            # print("---exploredlist",exploredlist)

            # guarantee edge exists
            if cur in problem['edges']:
                for neighbor in problem['edges'][cur]:
                    frontier.append(path + ' ' + neighbor[0])
        # print("---frontier", frontier)
    # solution = 'Ar D C\nAr C G'
    # Exploration order\nsolution path
    solution = ' '.join(exploredlist)
    solution += '\n'
    solution += path
    return solution

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 2
    grader.grade(problem_id, test_case_id, bfs_search, parse.read_graph_search_problem)