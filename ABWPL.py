from Projection import projection
import numpy as np


class StateStatistic:
    def __init__(self):
        self.adjusted_boundary = 0
        self.prev_best_action = 0
        self.steps_since_update = 0
        self.avg_update_steps = 1


class ABWPL:
    def __init__(self, learning_rate, future_rewards_importance, exploration_rate, eta, number_of_actions, enable_wpl3):
        self.learning_rate = learning_rate
        self.future_rewards_gamma = future_rewards_importance
        self.exploration_rate = exploration_rate
        self.eta = eta
        self.number_of_actions = number_of_actions
        self.Q = {}
        self.PI = {}

        if enable_wpl3:
            self.statistics = {}
        else:
            self.statistics = None

    def check_state(self, nstate):
        if nstate not in self.Q:
            self.Q[nstate] = np.zeros(self.number_of_actions)
            self.PI[nstate] = np.full(self.number_of_actions, 1 / self.number_of_actions)
            if self.statistics is not None:
                self.statistics[nstate] = StateStatistic()

    def update(self, prevstate, nstate, action, reward, terminal):
        # create states if they dont exist already
        self.check_state(prevstate)
        self.check_state(nstate)

        # Update the Q-table of agent A and agent B
        if terminal:
            self.Q[prevstate][action] = (1 - self.learning_rate) * self.Q[prevstate][action] + \
                                        self.learning_rate * reward
        else:
            self.Q[prevstate][action] = (1 - self.learning_rate) * self.Q[prevstate][action] + \
                                        self.learning_rate * (reward + self.future_rewards_gamma * max(self.Q[nstate]))

        if self.statistics is not None:
            best_action = np.argmax(self.Q[prevstate])
            self.statistics[prevstate].steps_since_update += 1
            # if best action has not changed
            if best_action == self.statistics[prevstate].prev_best_action:
                # and we've stepped over our avg update steps
                if self.statistics[prevstate].steps_since_update > self.statistics[prevstate].avg_update_steps:
                    # we start adjusting our boundary
                    self.statistics[prevstate].adjusted_boundary += 0.5 / self.statistics[prevstate].avg_update_steps
                    # but without going over the limit
                    if self.statistics[prevstate].adjusted_boundary >= 0.5:
                        self.statistics[prevstate].adjusted_boundary = 0.5 - self.eta
            # otherwise if best action is changed
            else:
                # we reset our adjusted boundary
                self.statistics[prevstate].adjusted_boundary = 0
                # and update our avg_update_steps IF they're over "avg update steps / 2" (to ignore noise)
                if self.statistics[prevstate].steps_since_update > self.statistics[prevstate].avg_update_steps / 2:
                    # moving average of last 2 windows
                    self.statistics[prevstate].avg_update_steps = (self.statistics[prevstate].avg_update_steps +
                                                                    self.statistics[prevstate].steps_since_update) / 2
                # steps since last change
                self.statistics[prevstate].steps_since_update = 0

            self.statistics[prevstate].prev_best_action = best_action

        for action in range(self.number_of_actions):
            difference = 0

            # compute difference between this reward and average reward
            for action2 in range(self.number_of_actions):
                difference += self.Q[prevstate][action] - self.Q[prevstate][action2]
            difference /= self.number_of_actions - 1

            # scale to sort of normalize the effect of a policy
            if difference > 0:
                # when we are favoring the best action, we take a lower delta_policy, so that we move slowly
                delta_policy = 1 - self.PI[prevstate][action]
            else:
                # when we are favoring the worst action, we take a higher delta_policy, so that we move quickly
                delta_policy = self.PI[prevstate][action]

            if self.statistics is not None:
                delta_policy = delta_policy * (1 - (self.statistics[prevstate].adjusted_boundary * 2)) + \
                    self.statistics[prevstate].adjusted_boundary

            rate = self.eta * difference * delta_policy
            self.PI[prevstate][action] += rate

        # project policy back into valid policy space
        self.PI[prevstate] = projection(self.PI[prevstate], self.exploration_rate)

