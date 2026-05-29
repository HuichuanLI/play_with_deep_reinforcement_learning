import copy
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


# 1: 策略网络 (Actor)
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out


# 2: SAPO 算法 (Soft Adaptive Policy Optimization)
class SAPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr,
                 epochs, gamma, beta, tau_pos, tau_neg, device):
        # 当前策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 参考策略网络 (冻结参数)
        self.ref_model = copy.deepcopy(self.actor)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.epochs = epochs
        self.beta = beta
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
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

        # Group Relative Advantage
        mean_returns = returns.mean()
        std_returns = returns.std() + 1e-8
        advantages = (returns - mean_returns) / std_returns

        # 旧策略概率
        with torch.no_grad():
            old_probs = self.actor(states)
            old_log_probs = torch.log(old_probs.gather(1, actions) + 1e-10)

        # 多轮训练
        for _ in range(self.epochs):
            probs = self.actor(states)
            log_probs = torch.log(probs.gather(1, actions) + 1e-10)
            ratio = torch.exp(log_probs - old_log_probs)

            # ==================== SAPO 核心 ====================
            # 非对称温度
            tau = torch.where(
                advantages > 0,
                torch.tensor(self.tau_pos, device=self.device),
                torch.tensor(self.tau_neg, device=self.device)
            )
            # 软门控函数
            soft_gate = torch.sigmoid(tau * (ratio - 1)) * (4.0 / tau)
            sapo_loss = -(soft_gate * advantages).mean()

            # KL 散度惩罚
            ref_probs = self.ref_model(states)
            log_p = torch.log(probs + 1e-10)
            log_q = torch.log(ref_probs + 1e-10)
            kl_div = torch.sum(probs * (log_p - log_q), dim=1).mean()

            # 总损失
            loss = sapo_loss + self.beta * kl_div
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# 3: 辅助函数: 计算折扣回报
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# 4: SAPO 训练函数
def train_sapo_agent(env, agent, num_episodes, batch_size=5):
    return_list = []
    num_iterations = num_episodes // batch_size

    with tqdm(total=num_iterations, desc='SAPO Training') as pbar:
        for i in range(num_iterations):
            batch_transition_dict = {
                'states': [], 'actions': [], 'returns': [], 'dones': []
            }
            group_rewards = []

            # 采集一组数据
            for _ in range(batch_size):
                state, info = env.reset()
                done = False
                episode_rewards = []
                episode_states = []
                episode_actions = []

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    state = next_state

                # 记录回报
                total_reward = sum(episode_rewards)
                return_list.append(total_reward)
                group_rewards.append(total_reward)

                # 计算回报
                returns = compute_returns(episode_rewards, agent.gamma)

                # 存入 batch
                batch_transition_dict['states'].extend(episode_states)
                batch_transition_dict['actions'].extend(episode_actions)
                batch_transition_dict['returns'].extend(returns)

            # 更新策略
            agent.update(batch_transition_dict)

            pbar.set_postfix({
                'episode': f'{batch_size * (i + 1)}',
                'avg_return': f'{np.mean(group_rewards):.2f}'
            })
            pbar.update(1)

    return return_list


# 移动平均
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    return (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size


# ==================== 主训练程序 ====================
if __name__ == "__main__":
    # 超参数
    actor_lr = 1e-3
    num_episodes = 800
    batch_size = 10
    hidden_dim = 128
    epochs = 10
    gamma = 0.98
    beta = 0.01
    tau_pos = 1.0
    tau_neg = 10.0

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

    # 实例化 Agent
    agent = SAPO(
        state_dim, hidden_dim, action_dim, actor_lr,
        epochs, gamma, beta, tau_pos, tau_neg, device
    )

    # 训练
    return_list = train_sapo_agent(env, agent, num_episodes, batch_size)

    # 画图
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, return_list, alpha=0.5, label='Raw Returns')

    window_size = 20
    if len(return_list) > window_size:
        mv_return = moving_average(return_list, window_size)
        plt.plot(range(window_size - 1, len(return_list)), mv_return, color='red', label='Moving Avg')

    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'SAPO on {env_name} (tau_pos={tau_pos}, tau_neg={tau_neg})')
    plt.legend()
    plt.grid(True)
    plt.show()
