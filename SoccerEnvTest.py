import random
from ABWPL import ABWPL
import numpy as np


def get_action_mod(action):
    if action == 0:
        return [-1, 0]
    if action == 1:
        return [1, 0]
    if action == 2:
        return [0, -1]
    if action == 3:
        return [0, 1]
    return [0, 0]


def get_movement_action_to_target(target):
    if target[0] < 0:
        return 0
    elif target[0] > 0:
        return 1
    elif target[1] < 0:
        return 2
    elif target[1] > 0:
        return 3

    return np.random.randint(0, 4)


class SoccerEmptyGoalEnv:
    def __init__(self):
        self.at0 = [0, 0]
        self.at1 = [0, 0]
        self.at2 = [0, 0]
        self.def1 = [0, 0]
        self.def2 = [0, 0]
        self.ball = [0, 0]
        self.number_of_actions = 6

    def step(self, actions):
        at_with_ball_passed = False
        new_ball_pos = self.ball
        for index, action in enumerate(actions):
            if index == 0:
                me = self.at0
                left_comrade = self.at1
                right_comrade = self.at2
            elif index == 1:
                me = self.at1
                left_comrade = self.at0
                right_comrade = self.at2
            elif index == 2:
                me = self.at2
                left_comrade = self.at0
                right_comrade = self.at1
            elif index == 3:
                me = self.def1
            elif index == 4:
                me = self.def2

            # move
            if action <= 3:
                movement = get_action_mod(action)
                carry_ball = self.ball == me

                my_new_pos = [me[0] + movement[0], me[1] + movement[1]]
                # dont move on top of other players
                if my_new_pos != left_comrade and my_new_pos != right_comrade:
                    me[0] = my_new_pos[0]
                    me[1] = my_new_pos[1]

                    # dont get off field
                    if me[0] < 0: me[0] = 0
                    if me[1] < 0: me[1] = 0
                    if me[0] > 10: me[0] = 10
                    if me[1] > 10: me[1] = 10

                # carry ball with me
                if carry_ball:
                    new_ball_pos = list(me)
            # pass ball if we have it
            elif self.ball == me and action <= 5:
                at_with_ball_passed = True
                if action == 4:
                    new_ball_pos = list(left_comrade)
                elif action == 5:
                    new_ball_pos = list(right_comrade)

        self.ball = new_ball_pos
        return self.get_state(at_with_ball_passed)

    def get_state(self, at_with_ball_passed=False):
        min_ball_dist_to_def = 100

        states = []
        for index in range(5):
            if index == 0:
                me = self.at0
                left_comrade = self.at1
                right_comrade = self.at2
            elif index == 1:
                me = self.at1
                left_comrade = self.at0
                right_comrade = self.at2
            elif index == 2:
                me = self.at2
                left_comrade = self.at0
                right_comrade = self.at1
            elif index == 3:
                me = self.def1
                comrade = self.def2
                if self.ball == self.at0:
                    left_enemy = self.at1
                    right_enemy = self.at2
                elif self.ball == self.at1:
                    left_enemy = self.at0
                    right_enemy = self.at2
                else:
                    left_enemy = self.at0
                    right_enemy = self.at1
            elif index == 4:
                me = self.def2
                comrade = self.def1
                if self.ball == self.at0:
                    left_enemy = self.at1
                    right_enemy = self.at2
                elif self.ball == self.at1:
                    left_enemy = self.at0
                    right_enemy = self.at2
                else:
                    left_enemy = self.at0
                    right_enemy = self.at1

            if index < 3:
                # passes only
                if only_passes:
                    states.append(str(min(np.abs(self.def1[0] - me[0]) + np.abs(self.def1[1] - me[1]),
                                          np.abs(self.def2[0] - me[0]) + np.abs(self.def2[1] - me[1]))) + ";" +
                                  str(min(
                                      np.abs(self.def1[0] - left_comrade[0]) + np.abs(self.def1[1] - left_comrade[1]),
                                      np.abs(self.def2[0] - left_comrade[0]) + np.abs(
                                          self.def2[1] - left_comrade[1]))) + ";" +
                                  str(min(
                                      np.abs(self.def1[0] - right_comrade[0]) + np.abs(self.def1[1] - right_comrade[1]),
                                      np.abs(self.def2[0] - right_comrade[0]) + np.abs(
                                          self.def2[1] - right_comrade[1]))) + ";" +
                                  str(self.ball == me))
                else:
                    states.append(str(left_comrade[0] - me[0]) + "," + str(left_comrade[1] - me[1]) + ";" +
                                  str(right_comrade[0] - me[0]) + "," + str(right_comrade[1] - me[1]) + ";" +
                                  str(min(np.abs(self.def1[0] - me[0]) + np.abs(self.def1[1] - me[1]),
                                          np.abs(self.def2[0] - me[0]) + np.abs(self.def2[1] - me[1]))) + ";" +
                                  str(min(
                                      np.abs(self.def1[0] - left_comrade[0]) + np.abs(self.def1[1] - left_comrade[1]),
                                      np.abs(self.def2[0] - left_comrade[0]) + np.abs(
                                          self.def2[1] - left_comrade[1]))) + ";" +
                                  str(min(
                                      np.abs(self.def1[0] - right_comrade[0]) + np.abs(self.def1[1] - right_comrade[1]),
                                      np.abs(self.def2[0] - right_comrade[0]) + np.abs(
                                          self.def2[1] - right_comrade[1]))) + ";" +
                                  str(self.ball == me))
            else:
                min_ball_dist_to_def = min(min_ball_dist_to_def,
                                           np.abs(self.ball[0] - me[0]) + np.abs(self.ball[1] - me[1]))
                states.append(str(self.ball[0] - me[0]) + "," + str(self.ball[1] - me[1]) + ";" +
                              str(left_enemy[0] - me[0]) + "," + str(left_enemy[1] - me[1]) + ";" +
                              str(right_enemy[0] - me[0]) + "," + str(right_enemy[1] - me[1]) + ";" +
                              str(comrade[0] - me[0]) + "," + str(comrade[1] - me[1]))

        terminal = min_ball_dist_to_def < 3
        if only_passes:
            at_reward = -1 if terminal else (0 if at_with_ball_passed else 1)
        else:
            at_reward = -1 if terminal else 0.1
        return states, at_reward, 1000 if terminal else 0, terminal

    def reset(self):
        self.at0 = [0, 0]
        self.at1 = [10, 0]
        self.at2 = [5, 10]
        self.def1 = [5, 0]
        self.def2 = [5, 10]
        self.ball = [0, 0]

        return self.get_state()


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


only_passes = False
abwpl_enabled = True

exploration_rate = 0.1
init_exploration_rate = 0.1
min_exploration_rate = 0.0001
max_exploration_steps = 50000

learning_rate = 0.1
eta = learning_rate / 100
future_rewards_importance = 0.9
number_of_actions = 3 if only_passes else 6
number_of_actions2 = 4

wpl_0 = ABWPL(learning_rate, future_rewards_importance, min_exploration_rate,
              eta, number_of_actions, abwpl_enabled)

env = SoccerEmptyGoalEnv()
state, _, _, _ = env.reset()
episode_steps = 0
episode_reward = 0

avg_steps = []
all_steps = []
log_avg_steps = []
max_iterations = 160000

for iteration in range(max_iterations):
    episode_steps += 1
    exploration_rate = max(min_exploration_rate,
                           init_exploration_rate * (1-(iteration / max_exploration_steps)))

    action_at0 = get_action(wpl_0, state[0])
    action_at1 = get_action(wpl_0, state[1])
    action_at2 = get_action(wpl_0, state[2])

    # hardcode enemy actions
    targets_def1 = [[int(coord) for coord in at.split(",")] for at in state[3].split(";")]
    targets_def2 = [[int(coord) for coord in at.split(",")] for at in state[4].split(";")]
    distances_def1 = [np.sum([np.abs(coord) for coord in at]) for at in targets_def1]
    distances_def2 = [np.sum([np.abs(coord) for coord in at]) for at in targets_def2]
    if distances_def1[0] < distances_def2[0]:
        # def 1 walks to ball, def 2 to closest player with no ball
        action_def1 = get_movement_action_to_target(targets_def1[0])
        if distances_def2[1] < distances_def2[2]:
            action_def2 = get_movement_action_to_target(targets_def2[1])
        else:
            action_def2 = get_movement_action_to_target(targets_def2[2])
    else:
        # def 2 walks to ball, def 1 to closest player with no ball
        action_def2 = get_movement_action_to_target(targets_def2[0])
        if distances_def2[1] < distances_def2[2]:
            action_def1 = get_movement_action_to_target(targets_def1[1])
        else:
            action_def1 = get_movement_action_to_target(targets_def1[2])

    prev_state = state
    state, reward_at, reward_def, terminal = env.step([action_at0 + 4 if only_passes else action_at0,
                                                       action_at1 + 4 if only_passes else action_at1,
                                                       action_at2 + 4 if only_passes else action_at2,
                                                       action_def1, action_def2])
    episode_reward += reward_at

    if "True" in prev_state[0] or not only_passes:
        wpl_0.update(prev_state[0], state[0], action_at0, reward_at, terminal)
    if "True" in prev_state[1] or not only_passes:
        wpl_0.update(prev_state[1], state[1], action_at1, reward_at, terminal)
    if "True" in prev_state[2] or not only_passes:
        wpl_0.update(prev_state[2], state[2], action_at2, reward_at, terminal)

    if terminal or episode_steps==5000:
        state, _, _, _ = env.reset()
        print_state = False
        avg_steps.append(episode_steps)
        all_steps.append(episode_steps)
        episode_steps = 0

    if iteration % 10000 == 9999:
        tmp_avg_steps = np.mean(avg_steps)
        print("ABWPL:", abwpl_enabled, ", ", tmp_avg_steps, "steps, ", 100 * iteration / max_iterations, "%")
        log_avg_steps.append(tmp_avg_steps)
        avg_steps = []

print(log_avg_steps)
print(all_steps)
