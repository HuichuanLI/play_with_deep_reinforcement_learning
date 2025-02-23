import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Set random seed for reproducibility
np.random.seed(2)
torch.manual_seed(2)

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater than this threshold
MAX_EP_STEPS = 1000  # maximum time steps in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(nn.Module):
    def __init__(self, n_features, n_actions, lr=LR_A):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)
        self.fc2 = nn.Linear(20, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        probs = self.forward(state).detach().numpy()
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def learn(self, state, action, td_error):
        self.optimizer.zero_grad()
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.LongTensor([action])
        probs = self.forward(state)

        # Calculate the loss
        loss = -torch.log(probs[0, action]) * td_error
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Critic(nn.Module):
    def __init__(self, n_features, lr=LR_C):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)
        self.fc2 = nn.Linear(20, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def learn(self, state, reward, next_state):
        self.optimizer.zero_grad()
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        v_next = self.forward(next_state).detach()
        target = reward + GAMMA * v_next
        v_eval = self.forward(state)

        td_error = target - v_eval
        loss = td_error.pow(2).mean()  # Mean squared error
        loss.backward()
        self.optimizer.step()

        return td_error.item()


# Initialize the actor and critic
actor = Actor(N_F, N_A, lr=LR_A)
critic = Critic(N_F, lr=LR_C)

# Training loop
for i_episode in range(MAX_EPISODE):
    state = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        action = actor.choose_action(state)
        # print(env.step(action))
        next_state, reward, done, truncated, _ = env.step(action)

        if done or truncated:
            reward = -20

        track_r.append(reward)

        td_error = critic.learn(state, reward, next_state)  # Learn from critic
        actor.learn(state, action, td_error)  # Learn from actor

        state = next_state
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
