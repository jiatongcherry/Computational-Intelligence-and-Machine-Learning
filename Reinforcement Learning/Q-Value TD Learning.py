"""
This code implements a Q-learning algorithm
aimed at allowing an agent to learn an optimal policy for navigating a grid environment. The agent explores the grid, receives rewards, and adjusts its Q-values to improve its strategy. Key features include:

1. Q-Table and Visit Count Initialization:
    Q-values for all valid state-action pairs are initialized to 0,
    with visit counts starting at 1 to avoid division by zero errors.

2. Exploration vs. Exploitation (ε-greedy Strategy):
    The agent selects actions based on visit counts,
    balancing learning new behaviors and leveraging known rewards.

3. Action Noise:
    The intended actions of the agent may change due to a noise probability,
    simulating uncertainties in the real world.

4. Learning and Q-value Updates:
    Q-values are updated using the Bellman equation,
    which takes into account immediate rewards and expected future rewards.

5. Rewards and Terminal States:
    The agent receives rewards based on its position,
    including a reward for reaching the goal, penalties for falling into traps,
    and a small living reward to encourage quicker completion.

6. Decay of Exploration Probability:
    The exploration probability decays after each episode,
    reducing random exploration over time.

7. Optimal Policy Extraction:
    After training, the optimal policy is extracted from the Q-table and printed.

Some points needed to be illustrated
1. Greedy and Exploration Functions
    greedy is applied, selects the action that has the highest Q-value for the current state
    epsilon greedy strategy, allows the agent to explore less frequently visited actions with a probability of epsilon

2. Learning Rate Decay
    with decay, to force exploration

3. Stopping Criteria
    check if the agent consistently reaches the goal over a certain number of episodes



Parameter setting is written in the code below as comment




How to run this code?
I have set trials = 10 in 'if __name__ == "__main__"', and input python p4.py to run this code
output will be like:
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 1
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 2
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 3
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 4
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 5
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 6
[['>', '<', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 6
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 7
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', '<']] matches count: 7
[['>', '>', '>', 'x'], ['^', '#', 'v', 'x'], ['^', '<', '<', 'v']] matches count: 7
Optimal policy found 7/10 times.

[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 1
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 2
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 3
[['>', '>', '>', 'x'], ['^', '#', 'v', 'x'], ['^', '<', '<', 'v']] matches count: 3
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 4
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 5
[['>', '>', '>', 'x'], ['^', '#', 'v', 'x'], ['^', '<', '<', 'v']] matches count: 5
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 6
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 7
[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 8
Optimal policy found 8/10 times.

[['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']]
this is the grid generated in one round as the optimal grid
'^': up, or north direction
'v': down, or south direction
'>': right, or east direction
'<': left, or west direction

matches count:
this is a cumulative count, means that up to this round, how many rounds matches the right optimal answer

Optimal policy found x/10 times:
result, show the ability my code to predict the right answer





My Finding - based on the result shown before

1.Increasing Matches with the Optimal Policy:
    In the initial iterations, the number of matches with the optimal policy was relatively low, ranging from 1 to 3 matches.
    As the iterations progressed, the number of matches with the optimal policy gradually increased, ultimately reaching 8 matches.

2.Balance between Exploration and Exploitation:
    In the earlier iterations, the algorithm explored a variety of different strategies,
    such as [['>', '>', '>', 'x'], ['^', '#', 'v', 'x'], ['^', '<', '<', 'v']] matches count: 3
    to [['>', '>', '>', 'x'], ['^', '#', '<', 'x'], ['^', '<', '<', 'v']] matches count: 4
    ,leading to a lower number of matches with the optimal policy.

    As the iterations continued, the algorithm learned the optimal policy and started to exploit it more, resulting in an increasing number of matches with the optimal policy.

3.Convergence:
    In the final few iterations, the number of matches with the optimal policy stabilized at 8, indicating that the algorithm had converged to the optimal policy.
    This suggests that the algorithm is capable of learning and converging to the best solution over time.

4.Exploration-Exploitation Trade-off:
    In the initial iterations, the algorithm tended to explore more strategies, even if they were not optimal.
    Subsequently, the algorithm reduced its exploration and focused more on exploiting the learned optimal policy.
    This exploration-exploitation trade-off helps the algorithm find the best balance during the learning process.

5.policy iteration
    In each iteration of Q-learning, the agent updates its policy based on the current Q values.
    Specifically, the agent selects the action with the highest Q value as the optimal action for the current state and updates the policy accordingly.
    This process ensures that the agent gradually converges to the optimal strategy during the learning process.


"""


import random

# copy from 2.prob
discount = 1
noise = 0.1
livingReward = -0.01
iterations = 3000
optimal_time_count = 0

grid = [
    ['-', '-', '-', 1],
    ['-', '#', '-', -1],
    ['S', '-', '-', '-']
]
start_position = (2, 0)
actions = {'^': (-1, 0), 'v': (1, 0), '<': (0, -1), '>': (0, 1)}

true_direction_with_noise = {
    '^': ['^', '>', '<'],
    'v': ['v', '>', '<'],
    '>': ['>', '^', 'v'],
    '<': ['<', '^', 'v']
}


def Q_learning(Q_values, visit_counts, policy, learning_rate, exploration_rate):
    for ep in range(iterations):
        current_row, current_col = start_position

        while True:
            current_state = (current_row, current_col)

            # Epsilon-greedy action selection using the policy
            if random.random() >= exploration_rate:
                selected_action = max(Q_values[current_state], key=Q_values[current_state].get)
            else:
                selected_action = random.choice(list(actions.keys()))

            potential_actions = true_direction_with_noise[selected_action]
            action_weights = [1 - 2 * noise, noise, noise]
            actual_action = random.choices(potential_actions, action_weights)[0]
            row_step, col_step = actions[actual_action]
            next_row, next_col = current_row + row_step, current_col + col_step

            # Check for boundaries
            if (next_row < 0 or next_row >= len(grid) or
                    next_col < 0 or next_col >= len(grid[0]) or
                    grid[next_row][next_col] == '#'):
                next_row, next_col = current_row, current_col

            cell_type = grid[next_row][next_col]
            reward_value = 1 if cell_type == 1 else -1 if cell_type == -1 else livingReward

            # Update Q-value
            next_state = (next_row, next_col)
            max_future_reward = max(Q_values[next_state].values()) if next_state in Q_values else 0
            Q_values[current_state][selected_action] += learning_rate * (
                reward_value + discount * max_future_reward - Q_values[current_state][selected_action])
            visit_counts[current_state][selected_action] += 1

            # end condition
            if cell_type in [1, -1]:
                break

            current_row, current_col = next_row, next_col

        # Decay
        """
                iteration = 1000时
                episode = 0.9 匹配率基本在 0-2/10次
                episode = 0.999 匹配率基本在 4-6/10次
                episode = 0.999 匹配率基本在 3-6/10次 但是偶尔出现8/10次

                发现当提高iteration，比如到3000的时候，使用episode = 0.999比较稳定，基本都在5-8/10次
                就是速度稍微略慢


                When iteration = 1000:

                For episode = 0.9, the match rate is generally between 0-2 out of 10 trials.
                For episode = 0.999, the match rate is generally between 4-6 out of 10 trials.
                For episode = 0.999 (repeated), the match rate is typically between 3-6 out of 10 trials, but occasionally it reaches 8 out of 10.

                It was observed that when the iterations are increased, for example to 3000, using episode = 0.999
                becomes more stable, with the match rate consistently between 5-8 out of 10 trials, although the speed is slightly slower.
        """
        exploration_rate *= 0.999


        """
            policy iteration
        """

        # Policy improvement
        for state in Q_values:
            best_action = max(Q_values[state], key=Q_values[state].get)
            policy[state] = best_action  # Update the policy

def optimal(Q_values, environment_grid):
    global optimal_time_count
    generated_grid = []
    reference_grid = get_optimal_policy()

    for row_num, row_data in enumerate(environment_grid):
        current_row = []
        for col_num, cell_data in enumerate(row_data):
            if isinstance(cell_data, (int, float)):
                current_row.append('x')
            elif cell_data == '#':
                current_row.append('#')
            elif cell_data in ('-', 'S'):
                if (row_num, col_num) in Q_values:
                    optimal_action = max(Q_values[(row_num, col_num)], key=Q_values[(row_num, col_num)].get)
                    current_row.append(optimal_action)
                else:
                    current_row.append(' ')
            else:
                current_row.append(' ')
        generated_grid.append(current_row)

    optimal_time_count += 1 if generated_grid == reference_grid else 0
    print(generated_grid, 'matches count:', optimal_time_count)

def get_optimal_policy():
    return [
        ['>', '>', '>', 'x'],
        ['^', '#', '<', 'x'],
        ['^', '<', '<', 'v']
    ]

if __name__ == "__main__":
    trials = 10
    Q = {}
    counts = {}
    policy = {}

    for _ in range(trials):
        Q.clear()
        counts.clear()
        policy.clear()
        for i, row_data in enumerate(grid):
            for j, cell in enumerate(row_data):
                if cell in ('-', 'S'):
                    Q[(i, j)] = {}
                    counts[(i, j)] = {}
                    for action in actions:
                        Q[(i, j)][action] = 0
                        counts[(i, j)][action] = 1
                    policy[(i, j)] = random.choice(list(actions.keys()))  # Initialize random policy

                """
        1. alpha (学习率)
        alpha 控制的是每次更新 Q 值时新旧信息的平衡
        较高的 alpha 值（接近 1）意味着代理对新信息非常敏感，每次更新时会更重视新的奖励和状态，较快调整 Q 值。
        较低的 alpha 值（接近 0）则意味着代理更依赖之前的经验，更新时更倾向于保留旧的 Q 值，这样可以让学习更加稳定
        环境变化较快，快速适应新的信息 alpha 0.7 - 0.9。
        环境相对稳定，平稳学习 alpha 0.1 - 0.3。
        2. epsilon (探索率)
        较高的 epsilon 值（接近 1）探索更多新的动作而不是总是选择已知的最优动作。探索有助于发现新的更优策略，尤其是在早期阶段，但也可能导致效率较低。
        较低的 epsilon 值（接近 0）选择当前已知的最优动作 更快收敛，但也可能陷入次优解，无法发现更好的动作。
        较高的 epsilon 值 0.5 - 0.9
        依赖已学到的最优策略 较低的值 0.01 - 0.1

        Alpha (Learning Rate)
        Alpha controls the balance between old and new information when updating Q-values.
        A higher alpha value (close to 1) means the agent is very sensitive to new information, placing greater emphasis on the new rewards and states during each update, allowing for faster adjustments to the Q-values.
        A lower alpha value (close to 0) indicates that the agent relies more on previous experiences, favoring the retention of old Q-values during updates, which can lead to more stable learning.
        For environments that change rapidly and require quick adaptation to new information, an alpha of 0.7 - 0.9 is recommended.
        For relatively stable environments where learning can be smoother, an alpha of 0.1 - 0.3 is preferable.
        Epsilon (Exploration Rate)
        A higher epsilon value (close to 1) encourages the exploration of new actions rather than always choosing the known optimal action. Exploration helps discover new, better strategies, especially in the early stages, but it may also lead to lower efficiency.
        A lower epsilon value (close to 0) results in selecting the currently known optimal actions, leading to faster convergence but potentially causing the agent to get stuck in suboptimal solutions and miss better actions.
        A higher epsilon value can range from 0.5 - 0.9, while relying on learned optimal strategies would correspond to a lower value of 0.01 - 0.1.
                """
        #alpha 0.5 - 5 6 8 6 7 6
        #alpha 0.6 - 9 8 7 9 6 7
        #可以根据trial改变epsilon
        #增加epsilon可以一定程度提高匹配度 甚至达到10

        Q_learning(Q, counts, policy, 0.6, 0.3)
        optimal(Q, grid)

    print(f"Optimal policy found {optimal_time_count}/{trials} times.")
