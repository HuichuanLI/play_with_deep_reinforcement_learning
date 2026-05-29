import copy
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm  # 修复：缺失进度条库导入


# 策略网络 (Actor)
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out


# GRPO (Group Relative Policy Optimization)
class GRPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr,
                 epochs, eps, gamma, beta, device):
        # 当前策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 参考策略网络 (冻结参数)
        self.ref_model = copy.deepcopy(self.actor)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False  # 严格冻结参考网络

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        self.beta = beta
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        # 数据准备
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        returns = torch.tensor(transition_dict['returns'], dtype=torch.float).view(-1, 1).to(self.device)

        # GRPO 核心：Group 标准化优势函数
        mean_returns = returns.mean()
        std_returns = returns.std() + 1e-8
        advantages = (returns - mean_returns) / std_returns

        # 旧策略概率
        with torch.no_grad():
            old_probs = self.actor(states)
            old_log_probs = torch.log(old_probs.gather(1, actions) + 1e-10)

        # 多轮训练
        for _ in range(self.epochs):
            # 当前策略
            probs = self.actor(states)
            log_probs = torch.log(probs.gather(1, actions) + 1e-10)

            # PPO 截断损失
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            grpo_loss = -torch.min(surr1, surr2).mean()

            # KL 散度惩罚
            ref_probs = self.ref_model(states)
            all_log_probs = torch.log(probs + 1e-10)
            all_ref_log_probs = torch.log(ref_probs + 1e-10)
            kl_div = torch.sum(probs * (all_log_probs - all_ref_log_probs), dim=1).mean()

            # 总损失
            loss = grpo_loss + self.beta * kl_div

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# 计算蒙特卡洛回报
def compute_returns(rewards, gamma):
    returns = []
    discounted_sum = 0
    for r in reversed(rewards):
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    return returns


# 滑动平均函数
def moving_average(a, window_size=9):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# GRPO 专用训练函数
def train_grpo_agent(env, agent, num_episodes, batch_size=5):
    return_list = []
    num_iterations = int(num_episodes / batch_size)

    with tqdm(total=num_iterations, desc='GRPO Training') as pbar:
        for i in range(num_iterations):
            batch_transition_dict = {
                'states': [], 'actions': [], 'returns': [], 'dones': []
            }
            group_rewards = []

            # 采集一个 Group 数据
            for _ in range(batch_size):
                state = env.reset()
                done = False
                episode_rewards = []
                episode_states = []
                episode_actions = []

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated = env.step(action)
                    done = terminated or truncated

                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    state = next_state

                # 记录回报
                total_reward = sum(episode_rewards)
                return_list.append(total_reward)
                group_rewards.append(total_reward)

                # 计算折扣回报
                returns = compute_returns(episode_rewards, agent.gamma)

                # 存入 batch
                batch_transition_dict['states'].extend(episode_states)
                batch_transition_dict['actions'].extend(episode_actions)
                batch_transition_dict['returns'].extend(returns)

            # 更新策略
            agent.update(batch_transition_dict)

            # 更新进度条
            pbar.set_postfix({
                'episode': f'{batch_size * (i + 1)}',
                'avg_return': f'{np.mean(group_rewards):.2f}'
            })
            pbar.update(1)

    return return_list


# ===================== 主训练程序 =====================
if __name__ == '__main__':
    # 超参数
    actor_lr = 1e-3
    num_episodes = 600
    batch_size = 10
    hidden_dim = 128
    epochs = 10
    eps = 0.2
    gamma = 0.98
    beta = 0.04
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 环境
    env_name = 'CartPole-v1'
    env = gym.make(env_name)  # 训练时先关闭 render，更快
    # env = gym.make(env_name, render_mode='human')  # 想看画面再打开

    # 随机种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建智能体
    agent = GRPO(state_dim, hidden_dim, action_dim, actor_lr,
                 epochs, eps, gamma, beta, device)

    # 训练
    return_list = train_grpo_agent(env, agent, num_episodes, batch_size)

    # 画图
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'GRPO on {env_name} (Batch Size={batch_size})')
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns (Moving Avg)')
    plt.title(f'GRPO on {env_name}')
    plt.show()
