import sys, parse, grader
from heapq import heappush, heappop

#UCS-GSA
def ucs_search(problem):
    #Your p3 code here
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
                    heappush(frontier, (node[0] + neighbor[1], path + ' ' + neighbor[0]))
        #print("---frontier", frontier)
    # solution = 'S D B C\nS C G'
    # Exploration order\nsolution path
    solution = ' '.join(exploredlist)
    solution += '\n'
    solution += path
    return solution


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 3
    grader.grade(problem_id, test_case_id, ucs_search, parse.read_graph_search_problem)