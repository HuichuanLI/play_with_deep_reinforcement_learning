# -*- coding:utf-8 -*-
# @Time : 2021/8/16 11:54 下午
# @Author : huichuan LI
# @File : main.py
# @Software: PyCharm
from maze import Maze
from ActorCrtic import Actor
from ActorCrtic import Critic
import numpy as np



def update():
    step = 0
    circle_reward = 0

    for episode in range(100):
        # initial observation
        observation = env.reset()
        observation = observation[np.newaxis, :]
        total_reward = 0
        total_step = 0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation

            action = actor.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            observation_ = observation_[np.newaxis, :]

            # RL learn from this transition
            advantage = critic.learn(observation, observation_, reward)
            actor.learn(observation, [action], advantage)

            observation = observation_
            total_reward += reward
            total_step += 1
            # break while loop when end of this episode
            if done:
                print('episode: {}, total reward: {}, total step: {}'.format(episode, total_reward, total_step))
                circle_reward += total_reward
                if episode % 100 == 0:
                    print('average reward in 100 episode: {}'.format(circle_reward / 100))
                    circle_reward = 0
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    actor = Actor(env.n_features, env.n_actions)
    critic = Critic(env.n_features, env.n_actions)

    env.after(100, update)
    env.mainloop()
