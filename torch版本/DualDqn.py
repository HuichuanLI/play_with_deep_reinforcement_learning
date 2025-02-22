# DQN
# DoubleDQN DuelingDQN
import gym  # pip install gym==0.23,
import random

import matplotlib.pyplot as plt
import numpy as np
import collections
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列

    # r + gamma * v(s_t+1) - v(s_t)
    # r - v(s_t)
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # [(state,xxx),(state,xxx),()]
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    # Q(s,a), Q(s)-> [3, 1.7] \pi
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VAnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # V
        self.V = nn.Linear(hidden_dim, 1)
        # A
        self.A = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        x = F.relu(self.fc1(x))
        A = self.A(x)
        V = self.V(x)

        Q = V + A - A.mean(1).view(-1, 1)

        return Q


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device,
                 dqn_type='DuelingDQN'):
        self.action_dim = action_dim
        if dqn_type == 'DuelingDQN':
            self.q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.dqn_type = dqn_type

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            # tensor-> numpy
            action = self.q_net(state).argmax().item()

        return action

    def update(self, transition_dict):
        # [[0.1,0.3,xxx],[0.2,0.x,xxx]]
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # [1, 0, 1],shpae(3), [[1],[0],[1]],shape(N,1)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # [[3, 1.7], [5, 1.4], xxx]
        q_values = self.q_net(states).gather(1, actions)
        # r + gamma max Q
        if self.dqn_type == 'DoubleDQN':
            max_a = self.q_net(next_states).argmax(1).view(-1, 1)
            max_next_q = self.target_q_net(next_states).gather(1, max_a).view(-1, 1)
        else:
            max_next_q = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        q_target = rewards + self.gamma * max_next_q * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_target))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


if __name__ == '__main__':
    lr = 2e-3  # 5e-5
    num_episodes = 500
    hidden_dim = 64
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    env = gym.make('CartPole-v1')
    print(env.observation_space)
    print(env.action_space)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    return_list = []
    for i in range(6):
        with tqdm(total=int(num_episodes / 10), desc='interation %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episode_list = list(range(len(return_list)))
    plt.plot(episode_list, return_list)
    plt.xlabel('episodes')
    plt.ylabel('returns')
    plt.title('dqn')
    plt.show()

    state = env.reset()
    for _ in range(1000):
        env.render()
        action = agent.take_action(state)
        state, reward, done, truncated = env.step(action)
        if done or truncated:
            state = env.reset()
    env.close()
# env = gym.make('CartPole-v1')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
# env.close()
