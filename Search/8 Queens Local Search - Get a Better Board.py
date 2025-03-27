import sys, parse, grader
from p6 import count_attack
from p6 import number_of_attacks

# local search

def better_board(problem):
    #Your p7 code here
    # record the positio of queens
    queen = []
    neighbor = []
    minattack = -1
    for j in range(len(problem)):
        for i in range(len(problem)):
            if (problem[i][j] == 'q'):
                queen.append(i)

    #get the cost matrix
    coststring = number_of_attacks(problem)
    rows = coststring.strip().split('\n')
    costmatrix = [[int(item) for item in row.split()] for row in rows]

    currentattack = costmatrix[queen[0]][0]
    minattack = currentattack

    for i in range(len(costmatrix)):
        for j in range(len(costmatrix)):
            attack = costmatrix[i][j]
            if attack == minattack:
                neighbor.append((i, j))
            if minattack > attack:
                neighbor = [(i, j)]
                minattack = attack

    if(len(neighbor) > 1):
        position = min(neighbor, key=lambda pos: pos[0] + pos[1])
    else:
        position = neighbor[0] #position should be tuple

    problem[queen[position[1]]][position[1]] = '.'
    problem[position[0]][position[1]] = 'q'

    solution = "\n".join(" ".join(row) for row in problem)



#     solution = """. q . . . . . .
# . . . . . . . .
# . . . . . . . .
# . . . q . . . .
# q . . . q . . .
# . . . . . q . q
# . . q . . . q .
# . . . . . . . ."""


    return solution

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 7
    grader.grade(problem_id, test_case_id, better_board, parse.read_8queens_search_problem)