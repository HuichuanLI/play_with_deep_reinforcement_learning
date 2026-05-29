import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


# --------------------- 悬崖漫步环境 ---------------------
class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.x = 0
        self.y = nrow - 1
        self.change = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # 上下左右

    def step(self, action):
        self.x = np.clip(self.x + self.change[action][0], 0, self.ncol - 1)
        self.y = np.clip(self.y + self.change[action][1], 0, self.nrow - 1)
        state = self.y * self.ncol + self.x
        reward = -1
        done = False

        # 悬崖
        if self.y == self.nrow - 1 and 0 < self.x < self.ncol - 1:
            reward = -100
            done = True
        # 终点
        if self.y == self.nrow - 1 and self.x == self.ncol - 1:
            done = True
        return state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


# --------------------- 策略网络（修复输入维度） ---------------------
class PolicyNet(nn.Module):
    def __init__(self, state_num, action_dim, hidden_dim=64):
        super().__init__()
        self.state_num = state_num  # 总状态数 48
        self.fc1 = nn.Linear(state_num, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x 是 one-hot 向量 [batch, state_num]
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


# --------------------- 原始策略梯度 REINFORCE ---------------------
class PolicyGradient:
    def __init__(self, state_num, action_dim, lr, gamma, device):
        self.gamma = gamma
        self.device = device
        self.policy_net = PolicyNet(state_num, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def take_action(self, state):
        # 状态 one-hot 编码（关键修复）
        state_one_hot = torch.zeros(self.policy_net.state_num).to(self.device)
        state_one_hot[state] = 1.0

        with torch.no_grad():
            probs = self.policy_net(state_one_hot)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def calc_discount_reward(self, rewards):
        discounted = []
        sum_r = 0
        for r in reversed(rewards):
            sum_r = r + self.gamma * sum_r
            discounted.append(sum_r)
        return torch.tensor(discounted[::-1], dtype=torch.float32).to(self.device)

    def update(self, states, actions, rewards):
        # 转换 one-hot
        state_tensor = torch.zeros(len(states), self.policy_net.state_num, device=self.device)
        for i, s in enumerate(states):
            state_tensor[i, s] = 1.0

        action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        G = self.calc_discount_reward(rewards)

        # 前向 + loss
        probs = self.policy_net(state_tensor)
        log_prob = torch.log(probs.gather(1, action_tensor.unsqueeze(1))).squeeze()
        loss = -(log_prob * G).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --------------------- 训练 ---------------------
def train():
    env = CliffWalkingEnv(ncol=12, nrow=4)
    state_num = 12 * 4  # 48
    action_dim = 4
    lr = 1e-3
    gamma = 0.98
    episodes = 800
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PolicyGradient(state_num, action_dim, lr, gamma, device)
    return_list = []

    for i in tqdm(range(episodes)):
        state = env.reset()
        done = False
        states, actions, rewards = [], [], []
        total_return = 0

        while not done:
            action = agent.take_action(state)
            next_state, r, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(r)
            total_return += r
            state = next_state

        agent.update(states, actions, rewards)
        return_list.append(total_return)

    # 画图
    plt.plot(return_list)
    plt.title("Policy Gradient (REINFORCE) - CliffWalking")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.show()


if __name__ == "__main__":
    train()
