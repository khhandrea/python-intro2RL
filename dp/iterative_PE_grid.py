from dataclasses import dataclass, field
import numpy as np

@dataclass
class Grid_DP:
    state_num: int
    state_values: np.ndarray
    policies: np.ndarray
    transitions: np.ndarray
    rewards: np.ndarray
    gamma: float = 0.
    policy_transitions: np.ndarray = field(init=False)
    policy_rewards: np.ndarray = field(init=False)

    def __post_init__(self):
        self.policy_transitions = np.zeros((self.state_num, self.state_num))
        self.policy_rewards = np.zeros(self.state_num)

        for state_from in range(self.state_num):
            for state_to in range(self.state_num):
                for action in self.policies[state_from]:
                    self.policy_transitions[state_from][state_to] += \
                        self.policies[state_from][action] * self.transitions[state_from][state_to][action]
                    
        for state in range(self.state_num):
            for action in self.policies[state]:
                self.policy_rewards[state] += self.policies[state][action] * self.rewards[state][action]

    def policy_evaluate(self) -> None:
        old_values = self.state_values.copy()
        for state in range(self.state_num):
            self.state_values[state] = self.policy_rewards[state] + self.gamma * np.dot(self.policy_transitions[state], old_values)

    def policy_improve(self) -> np.ndarray:
        print('improve!')
        new_policies = self.polices.copy()
        for state in range(self.state_num):
            pass

        return new_policies

    def print_value(self) -> None:
        for i in range(16):
            print(f'{self.state_values[i]: .3f}', end=' ')
            if i % 4 == 3:  
                print()