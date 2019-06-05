import random
from ABWPL import ABWPL
import numpy as np


class DeterministicEnv:
    def __init__(self):
        self.number_of_actions = 5

    def step(self, action):
        return 0, [0.6, 0.4, -0.4, -0.4, 0][action], True

    def reset(self):
        return 0


def get_pi(wpl, s):
    wpl.check_state(s)
    return wpl.Q[s], wpl.PI[s]


def get_action(wpl, s):
    if random.random() < exploration_rate:
        return int(random.random() * number_of_actions)
    wpl.check_state(s)
    action_prob = random.random() * sum(wpl.PI[s])
    for i in range(number_of_actions):
        if action_prob < wpl.PI[s][i]:
            return i
        action_prob -= wpl.PI[s][i]
    return -1


abwpl_enabled = True

exploration_rate = 0.1  # this should be annealed from 1 to min_exploration_rate in complex environments
min_exploration_rate = 0.0001
learning_rate = 0.1
eta = learning_rate / 100
future_rewards_importance = 0.9
number_of_actions = 5

wpl_0 = ABWPL(learning_rate, future_rewards_importance, min_exploration_rate,
              eta, number_of_actions, abwpl_enabled)

env = DeterministicEnv()
state = env.reset()

max_iterations = 10000

for iteration in range(max_iterations):
    action = get_action(wpl_0, state)

    prev_state = state
    state, reward, terminal = env.step(action)

    wpl_0.update(prev_state, state, action, reward, terminal)

    if iteration % 100 == 0:
        print("ABWPL:", abwpl_enabled, ", Policy:",[x for x in wpl_0.PI[0]])


