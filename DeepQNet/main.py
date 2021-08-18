# -*- coding:utf-8 -*-
# @Time : 2021/8/16 11:54 下午
# @Author : huichuan LI
# @File : main.py
# @Software: PyCharm
from maze import Maze
from DeepQN import DeepQNetwork


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            print(observation)

            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.store_transition(observation, action, reward, observation_)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    env.after(100, update)
    env.mainloop()
