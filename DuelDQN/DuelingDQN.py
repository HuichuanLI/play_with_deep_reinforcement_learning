import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

np.random.seed(1)
tf.random.set_seed(1)


class DuelingDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_action = n_actions
        self.n_feature = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learning_counter = 0
        self.memory_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        ## initializer
        self.w_initializer = tf.initializers.RandomNormal(mean=0, stddev=0.3)
        self.b_initializer = tf.initializers.constant(0.2)

        ## loss & optimizer & metric
        self.loss = tf.losses.mean_squared_error
        self.optimizer = tf.optimizers.RMSprop(self.lr)
        self.metrics = ['acc']
        ## build net
        self.eval_net = self.build_eval_net()
        self.target_net = self.build_target_net()

    def build_eval_net(self):
        n_l1 = 20
        inputs = tf.keras.Input(shape=(self.n_feature,))
        x = tf.keras.layers.Dense(n_l1, activation='relu', kernel_initializer=self.w_initializer,
                                  bias_initializer=self.b_initializer, name='evaluate_Dense1_1')(inputs)
        A = Dense(self.n_action, activation='softmax', kernel_initializer=self.w_initializer,
                  bias_initializer=self.b_initializer, name='evaluate_Dense1_2')(x)

        x = tf.keras.layers.Dense(n_l1, activation='relu', kernel_initializer=self.w_initializer,
                                  bias_initializer=self.b_initializer, name='evaluate_Dense2_1')(inputs)
        V = Dense(1, activation='softmax', kernel_initializer=self.w_initializer,
                  bias_initializer=self.b_initializer, name='evaluate_Dense2_2')(x)
        out = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        model = tf.keras.Model(inputs=inputs, outputs=out)

        return model

    def build_target_net(self):
        n_l1 = 20
        inputs = tf.keras.Input(shape=(self.n_feature,))
        x = tf.keras.layers.Dense(n_l1, activation='relu', kernel_initializer=self.w_initializer,
                                  bias_initializer=self.b_initializer, name='target_Dense1_1')(inputs)
        A = Dense(self.n_action, activation='softmax', kernel_initializer=self.w_initializer,
                  bias_initializer=self.b_initializer, name='target_Dense1_2')(x)

        x = tf.keras.layers.Dense(n_l1, activation='relu', kernel_initializer=self.w_initializer,
                                  bias_initializer=self.b_initializer, name='target_Dense2_1')(inputs)
        V = Dense(1, activation='softmax', kernel_initializer=self.w_initializer,
                  bias_initializer=self.b_initializer, name='target_Dense2_2')(x)
        out = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        model = tf.keras.Model(inputs=inputs, outputs=out)

        return model

    def replace_parameters(self):
        w = self.eval_net.get_weights()
        self.target_net.set_weights(w)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, (a, r), s_))
        index = self.memory_counter % self.memory_size
        self.memory[index:] = transition
        self.memory_counter += 1

    def get_q_target(self, batch_memory):
        ## extarct q_eval (shape -> q_target)
        row_index = np.arange(0, self.batch_size)
        column_index = batch_memory[:, self.n_feature].astype(np.int)
        index = list(zip(row_index, column_index))

        ## y_true
        # 真实值
        # DQN直接通过targetnet进行学习，目前通过eval_net进行选择
        q_next = self.target_net.predict(batch_memory[:, -self.n_feature:])
        q_eval4next = self.eval_net.predict(batch_memory[:, -self.n_feature:])
        max_act4next = np.argmax(q_eval4next, axis=1)  # the action that brings the highest value is evaluated by q_eval

        selected_q_next = q_next[row_index, max_act4next]  # Double DQN, select q_next depending on above actions

        rewards = batch_memory[:, self.n_feature + 1]
        q_target = rewards + self.gamma * selected_q_next

        return q_target, index

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.rand() > self.epsilon:
            action = np.random.randint(0, self.n_action)
        else:
            q_eval = self.eval_net.predict(observation)
            action = np.argmax(q_eval)
        return action

    @tf.function
    def train_model(self, batch_memory, q_target, index):
        with tf.GradientTape() as tape:
            # q预测值 通过上个状态查表
            q_eval = self.eval_net(batch_memory[:, :self.n_feature])
            q_eval = tf.gather_nd(q_eval, index)
            loss = self.loss(q_target, q_eval)

        ## optimize
        # 每次更新预测eval
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))

    def learn(self):
        if self.learning_counter % self.replace_target_iter == 0:
            self.replace_parameters()

        ## sample batch
        if self.memory_counter < self.memory_size:
            index = np.random.choice(self.memory_counter, size=self.batch_size)
        else:
            index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[index, :]
        ## training model
        q_target, index = self.get_q_target(batch_memory)
        batch_memory, q_target, index = tf.convert_to_tensor(batch_memory), \
                                        tf.convert_to_tensor(q_target), \
                                        tf.convert_to_tensor(index)
        self.train_model(batch_memory, q_target, index)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learning_counter += 1
