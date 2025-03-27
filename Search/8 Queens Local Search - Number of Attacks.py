import sys, parse, grader

def count_attack(queen):
    length = len(queen)
    position = [(queen[i], i)for i in range(0, length)]
    attack = 0
    for i in range(len(position) - 1):
        x1, y1 = position[i]
        for k in range(i+1, len(position)):
            x2, y2 = position[k]
            if abs(x1 - x2) == abs(y1 - y2) or x1 == x2:
                attack += 1
    return attack

def number_of_attacks(problem):
    #Your p6 code here
    solutionmatrix = [[0 for _ in range(len(problem))] for _ in range(len(problem))]
    queen = []

    #record the positio of queens
    for j in range(len(problem)):
        for i in range(len(problem)):
            if(problem[i][j] == 'q'):
                queen.append(i)

    for j in range(len(problem)): #col
        queenchange = queen[:]
        for i in range(len(problem)): #row
            queenchange[j] = i
            # print("queenchange", queenchange)
            # print("queen", queen)
            solutionmatrix[i][j] = count_attack(queenchange)

    # solution = """18 12 14 13 13 12 14 14
    # # 14 16 13 15 12 14 12 16
    # # 14 12 18 13 15 12 14 14
    # # 15 14 14 17 13 16 13 16
    # # 17 14 17 15 17 14 16 16
    # # 17 17 16 18 15 17 15 17
    # # 18 14 17 15 15 14 17 16
    # # 14 14 13 17 12 14 12 18"""
    solution = '\n'.join([' '.join([(' ' + str(item)) if int(item)<=9 else str(item) for item in row]) for row in solutionmatrix])

    return solution


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 6
    grader.grade(problem_id, test_case_id, number_of_attacks, parse.read_8queens_search_problem)