import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import random


# 悬崖漫步环境
class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.x = 0
        self.y = nrow - 1
        self.change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.P = self.build_P()

    def build_P(self):
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]
        for y in range(self.nrow):
            for x in range(self.ncol):
                s = y * self.ncol + x
                if y == self.nrow - 1 and x > 0:
                    for a in range(4):
                        P[s][a] = [(1.0, s, 0, True)]
                    continue
                for a in range(4):
                    dx, dy = self.change[a]
                    new_x = min(self.ncol - 1, max(0, x + dx))
                    new_y = min(self.nrow - 1, max(0, y + dy))
                    next_s = new_y * self.ncol + new_x
                    done = False
                    reward = -1
                    if new_y == self.nrow - 1 and new_x > 0 and new_x != self.ncol - 1:
                        reward = -100
                        done = True
                    if new_y == self.nrow - 1 and new_x == self.ncol - 1:
                        done = True
                    P[s][a] = [(1.0, next_s, reward, done)]
        return P

    def step(self, action):
        self.x = min(self.ncol - 1, max(0, self.x + self.change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + self.change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


# 蒙特卡洛 MC
class MonteCarlo:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([ncol * nrow, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_action)
        else:
            return np.argmax(self.Q_table[state])

    def best_action(self, state):
        q_max = np.max(self.Q_table[state])
        res = [0] * self.n_action
        for i in range(self.n_action):
            if self.Q_table[state][i] == q_max:
                res[i] = 1
        return res

    # 回合结束后，根据轨迹更新Q表
    def update(self, trajectory):
        G = 0
        # 反向遍历轨迹计算累计回报
        for s, a, r in reversed(trajectory):
            G = self.gamma * G + r
            td_error = G - self.Q_table[s][a]
            self.Q_table[s][a] += self.alpha * td_error


# Q-Learning TD
# 蒙特卡洛 MC
class MonteCarlo:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([ncol * nrow, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_action)
        else:
            return np.argmax(self.Q_table[state])

    def best_action(self, state):
        q_max = np.max(self.Q_table[state])
        res = [0] * self.n_action
        for i in range(self.n_action):
            if self.Q_table[state][i] == q_max:
                res[i] = 1
        return res

    # 回合结束后，根据轨迹更新Q表
    def update(self, trajectory):
        G = 0
        # 反向遍历轨迹计算累计回报
        for s, a, r in reversed(trajectory):
            G = self.gamma * G + r
            td_error = G - self.Q_table[s][a]
            self.Q_table[s][a] += self.alpha * td_error


# DQN 深度TD

# 策略打印工具
def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            state = i * env.ncol + j
            if state in disaster:
                print('****', end=' ')
            elif state in end:
                print('EEEE', end=' ')
            else:
                if hasattr(agent, "pi"):
                    act = agent.pi[state]
                else:
                    act = agent.best_action(state)
                str_buf = ""
                for val in act:
                    str_buf += action_meaning[len(str_buf)] if val > 0 else 'o'
                print(str_buf, end=' ')
        print()


if __name__ == "__main__":
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    epsilon = 0.1
    alpha = 0.1
    episode_num = 1000

    # 3.蒙特卡洛MC
    agent = MonteCarlo(12, 4, epsilon, alpha, gamma)
    reward_list = []
    for ep in tqdm(range(episode_num)):
        state = env.reset()
        done = False
        total_r = 0
        trajectory = []
        while not done:
            action = agent.take_action(state)
            next_s, r, done = env.step(action)
            trajectory.append((state, action, r))
            state = next_s
            total_r += r
        agent.update(trajectory)
        reward_list.append(total_r)

    plt.plot(reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.show()
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
