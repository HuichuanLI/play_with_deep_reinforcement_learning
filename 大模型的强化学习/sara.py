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





# SARSA TD
class SARSA:
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

    def update(self, s0, a0, r, s1, a1):
        td = r + self.gamma * self.Q_table[s1][a1] - self.Q_table[s0][a0]
        self.Q_table[s0][a0] += self.alpha * td


# Q-Learning TD
class QLearning:
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

    def update(self, s0, a0, r, s1):
        td = r + self.gamma * np.max(self.Q_table[s1]) - self.Q_table[s0][a0]
        self.Q_table[s0][a0] += self.alpha * td


# DQN 深度TD
class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.count = 0

        self.q_net = DQNNet(state_dim, action_dim).to(device)
        self.target_q_net = DQNNet(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
            q_value = self.q_net(state_tensor)
            action = q_value.argmax().item()
        return action

    def update(self, replay_buffer, batch_size=64):
        state_list, action_list, reward_list, next_state_list, done_list = replay_buffer.sample(batch_size)
        state = torch.tensor(state_list, dtype=torch.float).to(self.device)
        action = torch.tensor(action_list).view(-1, 1).to(self.device)
        reward = torch.tensor(reward_list, dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(next_state_list, dtype=torch.float).to(self.device)
        done = torch.tensor(done_list, dtype=torch.float).view(-1, 1).to(self.device)

        q_val = self.q_net(state).gather(1, action)
        next_q_val = self.target_q_net(next_state).max(dim=1, keepdim=True)[0]
        target_q = reward + self.gamma * next_q_val * (1 - done)

        loss = nn.MSELoss()(q_val, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


# 经验回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, exp):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(exp)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)


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
                elif isinstance(agent, DQN):
                    act_idx = agent.take_action(state)
                    act = [1 if idx == act_idx else 0 for idx in range(4)]
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

    # 选择算法运行，取消注释即可切换
    # 1.策略迭代
    # agent = PolicyIteration(env, theta, gamma)
    # agent.policy_iteration()
    # print_agent(agent, env, action_meaning, list(range(37,47)), [47])

    # 2.价值迭代
    # agent = ValueIteration(env, theta, gamma)
    # agent.value_iteration()
    # print_agent(agent, env, action_meaning, list(range(37,47)), [47])

    # 3.蒙特卡洛MC
    # agent = MonteCarlo(12, 4, epsilon, alpha, gamma)
    # reward_list = []
    # for ep in tqdm(range(episode_num)):
    #     state = env.reset()
    #     done = False
    #     total_r = 0
    #     trajectory = []
    #     while not done:
    #         action = agent.take_action(state)
    #         next_s, r, done = env.step(action)
    #         trajectory.append((state, action, r))
    #         state = next_s
    #         total_r += r
    #     agent.update(trajectory)
    #     reward_list.append(total_r)

    # # 4.SARSA
    agent = SARSA(12, 4, epsilon, alpha, gamma)
    reward_list = []
    for ep in tqdm(range(episode_num)):
        state = env.reset()
        done = False
        total_r = 0
        action = agent.take_action(state)
        while not done:
            next_s, r, done = env.step(action)
            next_a = agent.take_action(next_s)
            agent.update(state, action, r, next_s, next_a)
            state, action = next_s, next_a
            total_r += r
        reward_list.append(total_r)

    # # 5.Q-Learning
    # agent = QLearning(12,4,epsilon,alpha,gamma)
    # reward_list = []
    # for ep in tqdm(range(episode_num)):
    #     state = env.reset()
    #     done = False
    #     total_r = 0
    #     while not done:
    #         action = agent.take_action(state)
    #         next_s, r, done = env.step(action)
    #         agent.update(state, action, r, next_s)
    #         state = next_s
    #         total_r += r
    #     reward_list.append(total_r)

    # 6.DQN
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # state_dim = 1
    # action_dim = 4
    # lr = 1e-3
    # target_update_step = 10
    # buffer_capacity = 5000
    # batch_size = 64
    # agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update_step, device)
    # replay_buffer = ReplayBuffer(buffer_capacity)
    # reward_list = []
    # for ep in tqdm(range(episode_num)):
    #     state = env.reset()
    #     done = False
    #     total_r = 0
    #     while not done:
    #         action = agent.take_action(state)
    #         next_s, r, done = env.step(action)
    #         replay_buffer.add((state, action, r, next_s, done))
    #         if len(replay_buffer) > batch_size:
    #             agent.update(replay_buffer, batch_size)
    #         state = next_s
    #         total_r += r
    #     reward_list.append(total_r)

    plt.plot(reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.show()
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
