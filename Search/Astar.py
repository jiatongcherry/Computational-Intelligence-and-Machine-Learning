import sys, parse, grader
from heapq import heappop, heappush

def astar_search(problem):
    #Your p5 code here
    global path
    start_state = problem['start_state']
    goal_state = problem['goal_states']

    #initial state:[(start_state, 0)]
    frontier = []
    heappush(frontier, (0, start_state))
    exploredlist = list()  # GSA need exploredset
    while frontier:
        node = heappop(frontier)
        path = node[1]
        cur = path.split(' ')[-1]
        # edge check
        if cur in goal_state:
            break

        if (cur not in exploredlist):
            exploredlist.append(cur)
            #print("---exploredlist",exploredlist)

            # guarantee edge exists
            if cur in problem['edges']:
                for neighbor in problem['edges'][cur]:
                    heuristic_new = problem['heuristic'][neighbor[0]]
                    heuristic_old = problem['heuristic'][cur]
                    #current cost: node[0]   new edge cost: neighbor[1]
                    heappush(frontier, (node[0] + neighbor[1] + heuristic_new - heuristic_old,
                                        path + ' ' + neighbor[0]))
        #print("---frontier", frontier)

    # Exploration order\nsolution path
    solution = ' '.join(exploredlist)
    solution += '\n'
    solution += path
    #solution = 'S D C B\nS C G'
    return solution

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 5
    grader.grade(problem_id, test_case_id, astar_search, parse.read_graph_search_problem)