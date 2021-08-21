# -*- coding:utf-8 -*-
# @Time : 2021/8/21 7:01 下午
# @Author : huichuan LI
# @File : A3C.py
# @Software: PyCharm
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tensorflow as tf
import tensorflow_probability as tfp


class ACModel(Model):
    def __init__(self, n_feature, n_action, learning_rate=0.001, gamma=0.99):
        super().__init__()
        self.n_feature = n_feature
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.gamma = gamma

        ## losser & optimizer
        self.loss = None
        self.optimizer = None

        ## initializer
        self.w_init = tf.keras.initializers.RandomNormal(0., .1)
