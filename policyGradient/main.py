# -*- coding:utf-8 -*-
# @Time : 2021/8/16 11:54 下午
# @Author : huichuan LI
# @File : main.py
# @Software: PyCharm
from maze import Maze
from policygradient import PolicyGradient
import numpy as np


def update():
    step = 0

    for episode in range(200):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation

            action = RL.choose_action(observation[np.newaxis, :])

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.store_memorys(observation, action, reward)

            # swap observation

            # break while loop when end of this episode
            if done:
                rewards, total_reward, total_step = RL.learn()
                print('episode: {}, total reward: {}, total step: {}'.format(episode, total_reward, total_step))

                break
            observation = observation_

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = PolicyGradient(env.n_actions, env.n_features
                        )

    env.after(100, update)
    env.mainloop()
