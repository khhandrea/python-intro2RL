import numpy as np

from iterative_PE_grid import Grid_DP

if __name__ == '__main__':
    state_values = np.full(16, 0.0)
    state_num = len(state_values)

    policies = np.full(state_num, {'n': 0.25, 'e': 0.25, 'w': 0.25, 's': 0.25})
    policies[0] = {'n': 0., 'e': 0., 'w': 0., 's': 0.}
    policies[15] = {'n': 0., 'e': 0., 'w': 0., 's': 0.}

    gamma = 0.5

    transitions = np.empty((state_num, state_num), dtype=dict)
    for state_from in range(state_num):
        for state_to in range(state_num):
            transitions[state_from][state_to] = {'n': 0., 'e': 0., 'w': 0., 's': 0.}

    for state in range(1, state_num - 1):
        if state % 4 != 0: # Left available
            transitions[state][state - 1]['w'] = 1.0
        else:
            transitions[state][state]['w'] = 1.0

        if state % 4 != 3: # Right available
            transitions[state][state + 1]['e'] = 1.0
        else:
            transitions[state][state]['e'] = 1.0

        if state / 4 >= 1: # Up available
            transitions[state][state - 4]['n'] = 1.0
        else:
            transitions[state][state]['n'] = 1.0

        if state / 4 < 3: # Down available
            transitions[state][state + 4]['s'] = 1.0
        else:
            transitions[state][state]['s'] = 1.0

    rewards = np.full(state_num, {'n': -1.0, 'e': -1.0, 'w': -1.0, 's': -1.0})

    grid_DP = Grid_DP(
        state_values=state_values,
        state_num=state_num,
        policies=policies,
        gamma=gamma,
        transitions=transitions,
        rewards=rewards
        )

    iteration = 15
    epoch = 1

    print(f'iteration {0}')
    grid_DP.print_value()
    for i in range(1, iteration + 1):
        grid_DP.policy_evaluate()
        if i % epoch == 0:
            print(f'iteration {i}')
            grid_DP.print_value()