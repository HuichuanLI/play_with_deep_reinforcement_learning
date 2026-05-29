import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils
from torch import nn
import random


# 输⼊是某个状态, 输出则是状态下的动作概率分布.
# 我们采⽤在动作上的softmax()函数来实现⼀个可学习的多式分布.
# 策略梯度的定
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out


# 策略梯度算法实现
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        # 实化策略梯度的对象
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 优化对象Adam
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)
        # 因
        self.gamma = gamma
        self.device = device

    # 动作采样函数
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # state⾸先经过策略的计算, 得到softmax分布的输出
        probs = self.policy_net(state)
        # 根据动作概率分布随机采样
        action_distribution = torch.distributions.Categorical(probs)
        action = action_distribution.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        G = 0
        self.optimizer.zero_grad()
        # 此处的回报值计算, 采⽤蒙特卡洛算法 (⽤条序列做估计)
        # 蒙特卡洛算法能在序列后进⾏更新, 这就要求具有有限的数!!!
        # 蒙特卡洛算法对策略梯度的估计是⽆偏的, 但是⽅差⾮常⼤!!!
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            # 策略在当前state下, 计算action, 然后计算log (公式中 ▽theta)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            # 策略梯度公式计算向计算, 并加
            G = self.gamma * G + reward
            # 每⼀个时的函数
            loss = -log_prob * G
            # 每⼀个时都要向计算梯度, 是在for环中将梯度值进⾏了动加
            loss.backward()
        # 个for环后, 将所有时加的梯度⽤于参数更新
        self.optimizer.step()


learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode='human')
random.seed(0)
np.random.seed(0)
env.reset(seed=1)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
# print("state_dim: ", state_dim)
action_dim = env.action_space.n
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = env.reset()
            # print("state", state)
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                # print("next_state", next_state)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward

            # 加episode回报值, 是为了后⾯图
            return_list.append(episode_return)
            # 策略梯度算法的计算都在update()中
            agent.update(transition_dict)
            # 进度条显示
            if (i_episode + 1) % 10 == 0:
                avg_return = np.mean(return_list[-10:])
                pbar.set_postfix(
                    run=i + 1,
                    episode=i_episode + 1,
                    avg_return=f"{avg_return:.2f}"
                )

            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()
