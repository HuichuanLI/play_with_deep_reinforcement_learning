import matplotlib

matplotlib.use('MacOSX')  # Mac 专用弹出图片
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


# ------------------- 策略网络 & 价值网络 -------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        return self.fc2(out)


# ------------------- GAE 优势函数（必备） -------------------
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


# ------------------- PPO 算法（优化版） -------------------
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        # 网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # PPO 参数
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        # ✅ 修复维度：保证输入为 (1,4)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        # 数据转换
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 计算 TD 目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_error = td_target - self.critic(states)
        # 计算优势函数
        advantage = compute_advantage(self.gamma, self.lmbda, td_error).to(self.device)

        # 旧策略概率（不更新）
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 多轮训练 PPO
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO Clip 核心
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic 损失
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            # 反向传播
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# ------------------- 训练函数 -------------------
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    with tqdm(total=num_episodes, desc="训练进度") as pbar:
        for i in range(num_episodes):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            done = False

            while not done:
                action = agent.take_action(state)
                res = env.step(action)
                next_state = res[0]
                reward = res[1]
                done = res[2]

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                state = next_state
                episode_return += reward

            return_list.append(episode_return)
            agent.update(transition_dict)
            pbar.set_postfix({"当前奖励": episode_return})
            pbar.update(1)
    return return_list


# 滑动平均
def moving_average(a, window_size=9):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# ------------------- 主训练 -------------------
if __name__ == '__main__':
    # 超参数
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 300
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

    print("开始训练 PPO...")
    return_list = train_on_policy_agent(env, agent, num_episodes)

    # 画图
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))

    plt.subplot(122)
    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Returns')
    plt.title('Smoothed PPO on {}'.format(env_name))

    plt.tight_layout()
    plt.show()
