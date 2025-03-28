import os, sys
def read_layout_problem(file_path):
    #Your p1 code here
    #problem = ''
    with open(file_path, 'r') as file:
        lines = file.readlines()

    seed_line = lines[0].strip()
    seed = int(seed_line.split(':')[1].strip())
    grid = []

    for line in lines[1:]:
        grid.append(list(line.strip()))

    problem = {
        'seed': seed,
        'game': grid
    }
    return problem

if __name__ == "__main__":
    if len(sys.argv) == 3:
        problem_id, test_case_id = sys.argv[1], sys.argv[2]
        problem = read_layout_problem(os.path.join('test_cases','p'+problem_id, test_case_id+'.prob'))
        print(problem)
    else:
        print('Error: I need exactly 2 arguments!')