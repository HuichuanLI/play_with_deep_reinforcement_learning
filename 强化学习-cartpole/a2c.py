import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条工具


# ------------------- 1. 定义网络结构 -------------------
# 策略网络 Actor：输出动作概率分布
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)  # 输出概率和为1
        return out


# 价值网络 Critic：输出状态价值
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出单个价值数值

    def forward(self, x):
        out = F.relu(self.fc1(x))
        return self.fc2(out)


# ------------------- 2. 定义Actor-Critic算法主体 -------------------
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        # 初始化两个网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):
        """根据当前状态选择动作（带探索）"""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)  # 获取动作概率
        action_dist = torch.distributions.Categorical(probs)  # 概率分布
        action = action_dist.sample()  # 采样动作
        return action.item()

    def update(self, transition_dict):
        """更新Actor和Critic网络（核心训练逻辑）"""
        # 数据转换为tensor
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 1. 计算TD目标和TD误差
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_value = self.critic(states)
        td_error = td_target - td_value

        # 2. 计算Actor损失
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_error.detach())  # detach切断梯度

        # 3. 计算Critic损失
        critic_loss = torch.mean(F.mse_loss(td_value, td_target.detach()))

        # 4. 反向传播更新参数
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


# ------------------- 3. 训练工具函数（替代rl_utils，避免依赖缺失） -------------------
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    # 保存轨迹
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)  # 每局结束更新一次
                # 更新进度条
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def moving_average(a, window_size):
    """滑动平均，平滑奖励曲线"""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# ------------------- 4. 主训练代码 -------------------
if __name__ == '__main__':
    # 超参数
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 环境初始化
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    np.random.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建智能体
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

    # 开始训练
    return_list = train_on_policy_agent(env, agent, num_episodes)

    # 绘图
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))

    plt.subplot(1, 2, 2)
    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Returns')
    plt.title('Smoothed Actor-Critic on {}'.format(env_name))

    plt.tight_layout()
    plt.show()
