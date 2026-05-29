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


# DQN 网络
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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = target_update
        self.device = device
        self.count = 0

        self.q_net = DQNNet(state_dim, action_dim).to(device)
        self.target_q_net = DQNNet(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_value = self.q_net(state_tensor)
            return q_value.argmax().item()

    def get_best_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_value = self.q_net(state_tensor)
        return q_value.argmax().item()

    def update(self, replay_buffer, batch_size=64):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # ✅ 这里绝对不能加 unsqueeze(-1)！
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        action = torch.tensor(action, dtype=torch.int64).view(-1, 1).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).view(-1, 1).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).view(-1, 1).to(self.device)

        q_val = self.q_net(state).gather(1, action)
        next_q_val = self.target_q_net(next_state).max(1, keepdim=True)[0]
        target_q = reward + self.gamma * next_q_val * (1 - done)

        loss = nn.MSELoss()(q_val, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


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
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


# 打印策略
def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            state = i * env.ncol + j
            one_hot = np.zeros(env.nrow * env.ncol, dtype=np.float32)
            one_hot[state] = 1.0

            if state in disaster:
                print('****', end=' ')
            elif state in end:
                print('EEEE', end=' ')
            else:
                act_idx = agent.get_best_action(one_hot)
                str_buf = ''.join([action_meaning[k] if k == act_idx else 'o' for k in range(4)])
                print(str_buf, end=' ')
        print()


if __name__ == "__main__":
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    gamma = 0.9
    episode_num = 800

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STATE_DIM = env.nrow * env.ncol  # 48
    ACTION_DIM = 4
    LR = 1e-3
    TARGET_UPDATE = 10
    BUFFER_CAPACITY = 10000
    BATCH_SIZE = 64

    agent = DQN(STATE_DIM, ACTION_DIM, LR, gamma, 0.1, TARGET_UPDATE, device)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    reward_list = []

    for ep in tqdm(range(episode_num)):
        s = env.reset()
        done = False
        total_r = 0

        while not done:
            one_hot_s = np.zeros(STATE_DIM, dtype=np.float32)
            one_hot_s[s] = 1.0

            a = agent.take_action(one_hot_s)
            s_next, r, done = env.step(a)

            one_hot_snext = np.zeros(STATE_DIM, dtype=np.float32)
            one_hot_snext[s_next] = 1.0

            replay_buffer.add((one_hot_s, a, r, one_hot_snext, done))
            s = s_next
            total_r += r

        if len(replay_buffer) > BATCH_SIZE:
            for _ in range(5):
                agent.update(replay_buffer, BATCH_SIZE)

        agent.decay_epsilon()
        reward_list.append(total_r)

    plt.plot(reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Cliff Walking Reward Curve")
    plt.show()

    print("\n最终策略：")
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
