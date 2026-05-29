import copy
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


# 策略网络
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out


# 辅助函数: 计算折扣回报
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# GSPO 算法
class GSPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr,
                 epochs, eps, gamma, beta, device):
        # 当前策略
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 参考策略（冻结）
        self.ref_model = copy.deepcopy(self.actor)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

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
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        returns = torch.tensor(transition_dict['returns'], dtype=torch.float).view(-1, 1).to(self.device)
        timesteps = torch.tensor(transition_dict['timesteps'], dtype=torch.long).to(self.device)

        # GSPO 核心：按时间步 t 标准化优势函数
        advantages = torch.zeros_like(returns)
        max_t = timesteps.max().item()
        for t in range(max_t + 1):
            idxs = (timesteps == t).nonzero(as_tuple=True)[0]
            if len(idxs) > 1:
                t_ret = returns[idxs]
                mean = t_ret.mean()
                std = t_ret.std() + 1e-8
                advantages[idxs] = (t_ret - mean) / std
            elif len(idxs) == 1:
                advantages[idxs] = 0.0

        # 旧策略概率
        with torch.no_grad():
            old_probs = self.actor(states)
            old_log_probs = torch.log(old_probs.gather(1, actions) + 1e-10)

        # 多轮更新
        for _ in range(self.epochs):
            probs = self.actor(states)
            log_probs = torch.log(probs.gather(1, actions) + 1e-10)
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO Clip 损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            gspo_loss = -torch.min(surr1, surr2).mean()

            # KL 散度惩罚
            ref_probs = self.ref_model(states)
            log_p = torch.log(probs + 1e-10)
            log_q = torch.log(ref_probs + 1e-10)
            kl = torch.sum(probs * (log_p - log_q), dim=1).mean()

            # 总损失
            loss = gspo_loss + self.beta * kl
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# GSPO 训练函数（同状态采样）
def train_gspo_agent(env, agent, num_episodes, batch_size=5):
    return_list = []
    num_iter = num_episodes // batch_size

    with tqdm(total=num_iter, desc='GSPO Training') as pbar:
        for i in range(num_iter):
            batch = {
                'states': [], 'actions': [], 'returns': [],
                'timesteps': [], 'dones': []
            }
            group_rewards = []
            group_seed = random.randint(0, 100000)

            # 采集一组（相同初始状态）
            for _ in range(batch_size):
                state = env.reset(seed=group_seed)
                done = False
                ep_rewards, ep_states, ep_actions, ep_timesteps = [], [], [], []
                step = 0

                while not done:
                    action = agent.take_action(state)
                    next_s, r, term, trunc = env.step(action)
                    done = term or trunc

                    ep_states.append(state)
                    ep_actions.append(action)
                    ep_rewards.append(r)
                    ep_timesteps.append(step)

                    state = next_s
                    step += 1

                total = sum(ep_rewards)
                return_list.append(total)
                group_rewards.append(total)

                # 计算回报
                returns = compute_returns(ep_rewards, agent.gamma)

                # 存入 batch
                batch['states'].extend(ep_states)
                batch['actions'].extend(ep_actions)
                batch['returns'].extend(returns)
                batch['timesteps'].extend(ep_timesteps)

            # 更新策略
            agent.update(batch)

            pbar.set_postfix(episode=f'{batch_size * (i + 1)}', avg_return=f'{np.mean(group_rewards):.2f}')
            pbar.update(1)

    return return_list


# 移动平均
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    return (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size


# ==================== 主训练 ====================
if __name__ == "__main__":
    # 超参数
    actor_lr = 1e-3
    num_episodes = 600
    batch_size = 10
    hidden_dim = 128
    epochs = 10
    eps = 0.2
    gamma = 0.98
    beta = 0.04

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print("Using device:", device)

    # 环境
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    # 随机种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 智能体
    agent = GSPO(state_dim, hidden_dim, action_dim, actor_lr,
                 epochs, eps, gamma, beta, device)

    # 训练
    return_list = train_gspo_agent(env, agent, num_episodes, batch_size)

    # 画图
    episodes = list(range(len(return_list)))
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'GSPO on {env_name}')
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(mv_return)), mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Moving Avg Returns')
    plt.title('GSPO Moving Average')
    plt.show()
