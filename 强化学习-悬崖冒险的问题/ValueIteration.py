import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.ncol = ncol
        self.nrow = nrow
        self.x = 0
        self.y = nrow - 1
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右
        # 坐标系原点(0,0), 定义在左上⻆
        self.change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.P = self.build_P()

    # 外部调⽤这个函数来改变当前位置
    def step(self, action):
        # 执⾏action后的机器⼈坐标点(self.x, self.y)
        self.x = min(self.ncol - 1, max(0, self.x + self.change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + self.change[action][1]))
        # 以⼀维列表的序列index作为环境的state位置
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        # 下⼀个位置在悬崖或者⽬标
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    # 回归初始状态,坐标轴原点在左上⻆
    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

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


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0.0 else 'o'
                print(pi_str, end=' ')
        print()


#

# 价值迭代 DP
class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.state_num = env.ncol * env.nrow
        self.v = [0.0] * self.state_num
        self.pi = [[] for _ in range(self.state_num)]

    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0
            new_v = [0.0] * self.state_num
            for s in range(self.state_num):
                q_list = []
                for a in range(4):
                    q = 0
                    for p, s_n, r, d in self.env.P[s][a]:
                        q += p * (r + self.gamma * self.v[s_n] * (1 - d))
                    q_list.append(q)
                new_v[s] = max(q_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            cnt += 1
            if max_diff < self.theta:
                break
        print(f"价值迭代总轮数：{cnt}")
        self.get_policy()

    def get_policy(self):
        for s in range(self.state_num):
            q_list = []
            for a in range(4):
                q = 0
                for p, s_n, r, d in self.env.P[s][a]:
                    q += p * (r + self.gamma * self.v[s_n] * (1 - d))
                q_list.append(q)
            max_q = max(q_list)
            num = q_list.count(max_q)
            self.pi[s] = [1 / num if q == max_q else 0 for q in q_list]


if __name__ == "__main__":
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    epsilon = 0.1
    alpha = 0.1
    episode_num = 1000

    # 选择算法运行，取消注释即可切换
    # 1.策略迭代
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
