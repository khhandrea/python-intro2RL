import numpy as np

from iterative_PE_grid import Grid_DP

if __name__ == '__main__':
    state_values = np.full(16, 0.0)
    state_num = len(state_values)

    policies = np.empty(state_num, dtype=dict)
    policies[0] = {'n': 0., 'e': 0., 'w': 0., 's': 0.}
    policies[1] = {'n': 0., 'e': 0., 'w': 1., 's': 0.}
    policies[2] = {'n': 0., 'e': 0., 'w': 1., 's': 0.}
    policies[3] = {'n': 0., 'e': 0., 'w': .5, 's': .5}
    policies[4] = {'n': 1., 'e': 0., 'w': 0., 's': 0.}
    policies[5] = {'n': .5, 'e': 0., 'w': .5, 's': 0.}
    policies[6] = {'n': 0., 'e': 0., 'w': .5, 's': .5}
    policies[7] = {'n': 0., 'e': 0., 'w': 0., 's': 1.}
    policies[8] = {'n': 1., 'e': 0., 'w': 0., 's': 0.}
    policies[9] = {'n': .5, 'e': .5, 'w': 0., 's': 0.}
    policies[10] = {'n': 0., 'e': .5, 'w': 0., 's': .5}
    policies[11] = {'n': 0., 'e': 0., 'w': 0., 's': 1.}
    policies[12] = {'n': .5, 'e': .5, 'w': 0., 's': 0.}
    policies[13] = {'n': 0., 'e': 1., 'w': 0., 's': 0.}
    policies[14] = {'n': 0., 'e': 1., 'w': 0., 's': 0.}
    policies[15] = {'n': 0., 'e': 0., 'w': 0., 's': 0.}

    gamma = 0

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

    iteration = 5
    epoch = 1

    print(f'iteration {0}')
    grid_DP.print_value()
    for i in range(1, iteration + 1):
        grid_DP.policy_evaluate()
        if i % epoch == 0:
            print(f'iteration {i}')
            grid_DP.print_value()