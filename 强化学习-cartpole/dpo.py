import matplotlib

matplotlib.use('MacOSX')
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt


# ------------------- 策略网络 -------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out


# ------------------- DPO 算法 -------------------
class DPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, beta, epochs, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.ref_model = copy.deepcopy(self.actor)
        self.ref_model.eval()
        self.ref_model.to(device)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.beta = beta
        self.epochs = epochs
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions_w = torch.tensor(transition_dict['actions_w']).view(-1, 1).to(self.device)
        actions_l = torch.tensor(transition_dict['actions_l']).view(-1, 1).to(self.device)

        for _ in range(self.epochs):
            probs = self.actor(states)
            log_probs = torch.log(probs + 1e-10)
            policy_log_w = log_probs.gather(1, actions_w)
            policy_log_l = log_probs.gather(1, actions_l)

            with torch.no_grad():
                ref_probs = self.ref_model(states)
                ref_log_probs = torch.log(ref_probs + 1e-10)
                ref_log_w = ref_log_probs.gather(1, actions_w)
                ref_log_l = ref_log_probs.gather(1, actions_l)

            logits_w = policy_log_w - ref_log_w
            logits_l = policy_log_l - ref_log_l
            diff = logits_w - logits_l
            loss = -F.logsigmoid(self.beta * diff).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# ------------------- DPO 专用训练函数 -------------------
def train_dpo_online_agent(env, agent, num_episodes):
    return_list = []
    action_dim = env.action_space.n

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions_w': [],
                    'actions_l': [],
                    'next_states': [],
                    'dones': []
                }

                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

                done = False

                while not done:
                    action = agent.take_action(state)

                    action_l = action
                    while action_l == action:
                        action_l = random.randint(0, action_dim - 1)

                    res = env.step(action)
                    next_state = res[0]
                    reward = res[1]
                    done = res[2]

                    transition_dict['states'].append(state)
                    transition_dict['actions_w'].append(action)
                    transition_dict['actions_l'].append(action_l)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                agent.update(transition_dict)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                        'return': f'{np.mean(return_list[-10:]):.2f}'
                    })
                pbar.update(1)
    return return_list


# ------------------- 滑动平均 -------------------
def moving_average(a, window_size=9):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# ------------------- 主训练 -------------------
if __name__ == '__main__':
    actor_lr = 1e-3
    num_episodes = 500
    hidden_dim = 128
    epochs = 10
    beta = 2.0
    device = torch.device("cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DPO(state_dim, hidden_dim, action_dim, actor_lr, beta, epochs, device)
    return_list = train_dpo_online_agent(env, agent, num_episodes)

    # 画图
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DPO (Online) on {}'.format(env_name))

    plt.subplot(122)
    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Returns')
    plt.title('DPO Moving Average')

    plt.tight_layout()
    plt.show()
