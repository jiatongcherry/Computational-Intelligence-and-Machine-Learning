def read_grid_mdp_problem_p1(file_path):
    seed = 0
    noise = 0.0
    living_reward = 0.0
    start_position = None
    linefunc = None
    grid = []
    policy = []

    section_handlers = {
        "seed:": lambda value: int(value.strip()),
        "noise:": lambda value: float(value.strip()),
        "livingReward:": lambda value: float(value.strip()),
        "grid:": lambda _: 'grid',
        "policy:": lambda _: 'policy'
    }


    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            for key, handler in section_handlers.items():
                if line.startswith(key):
                    if key in {"grid:", "policy:"}:
                        linefunc = handler(None)
                    else:
                        if key == "seed:":
                            seed = handler(line.split(':')[1])
                        elif key == "noise:":
                            noise = handler(line.split(':')[1])
                        elif key == "livingReward:":
                            living_reward = handler(line.split(':')[1])
                    break
            else:
                if linefunc == 'grid':
                    row = line.split()
                    grid.append(row)
                    if 'S' in row:
                        start_position = (len(grid) - 1, row.index('S'))
                elif linefunc == 'policy':
                    policy.append(line.split())

    problem = {
        'seed': seed,
        'noise': noise,
        'livingReward': living_reward,
        'grid': grid,
        'policy': policy,
        'start_position': start_position
    }
    #print(problem)
    return problem


def read_grid_mdp_problem_p2(file_path):
    def parse_grid_line(line, grid):
        row = []
        for cell in line.split():
            if cell == '_':
                row.append(0)
            elif cell == 'S':
                row.append('S')
            elif cell == '#':
                row.append('#')
            else:
                row.append(int(cell))
        grid.append(row)

    def parse_value(value):
        if value.isdigit():
            return int(value)
        elif '.' in value:
            return float(value)
        return value

    problem = {
        "grid": [],
        "policy": []
    }

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    section_handlers = {
        "grid:": lambda line: parse_grid_line(line, problem["grid"]),
        "policy:": lambda line: problem["policy"].append(line.split())
    }

    current_section = None

    for line in lines:
        if line in section_handlers:
            current_section = line
            continue

        if current_section:
            section_handlers[current_section](line)
        else:
            key, value = line.split(': ')
            problem[key] = parse_value(value)
    #print(problem)
    return problem


def read_grid_mdp_problem_p3(file_path):
    discount = noise = livingReward = iterations = 0
    grid = []

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    for line in lines:
        if line.startswith("discount"):
            discount = float(line.split(":")[1].strip())

        elif line.startswith("noise"):
            noise = float(line.split(":")[1].strip())

        elif line.startswith("livingReward"):
            livingReward = float(line.split(":")[1].strip())

        elif line.startswith("iterations"):
            iterations = int(line.split(":")[1].strip())

        elif line.startswith("grid:"):
            grid.clear()
        else:
            if line:
                processed_row = []
                for cell in line.split():
                    stripped_cell = cell.strip()
                    processed_row.append(stripped_cell if stripped_cell in ('_', '#', 'S') else float(stripped_cell))
                grid.append(processed_row)

    problem = {
        'discount': discount,
        'noise': noise,
        'livingReward': livingReward,
        'iterations': iterations,
        'grid': grid
    }
    #print(problem)
    return problem