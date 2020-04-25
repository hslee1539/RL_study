import tensorflow as tf
import numpy as np
from collections import deque
import random

# VS code 인텔리센스 에러용
np.random = np.random


def rrange(loop : int, a = 0, b = 1):
    for _ in range(loop):
        yield random.randint(a,b)

def deargmax(indexs : np.ndarray, max : int, dtype = float):
    for index in indexs:
        for i in range(max):
            yield dtype(i == index)


class DQN:
    def __init__(self, num_inputs : int, num_outputs : int):
        self.__model = DQN.createModel(num_inputs, num_outputs)
        self.__previous_model = DQN.createModel(num_inputs, num_outputs)
        self.__buffer_size = 2000
        self.__buffer_len = 0
        self.__buffer_index = 0
        self.__buffer = (
            np.zeros([self.__buffer_size, num_inputs]), # state
            np.zeros([self.__buffer_size], int), # action
            np.zeros([self.__buffer_size, num_inputs]), # action_state
            np.zeros([self.__buffer_size]), # action_reward
            np.zeros([self.__buffer_size]) # action_done
        )

        self.__epsilon = 1.0
        self.__epsilon_decay = 0.991
        self.__epsilon_min = 0.001

        self.__num_inputs = num_inputs
        self.__num_outputs = num_outputs

    def summary(self):
        return "epsilon : [" + str(self.__epsilon) + "] buffer index [" + str(self.__buffer_index) + "]"

    
    def set_batch(self, batch_size = 32):
        self.batch_size = batch_size
        self.states = np.zeros([batch_size, self.__num_inputs])
        self.actions = np.zeros([batch_size], int)
        self.actions_state = np.zeros([batch_size, self.__num_inputs])
        self.actions_reward = np.zeros([batch_size])
        self.actions_done = np.zeros([batch_size])

    def act(self, state : np.ndarray):
        self.__epsilon *= self.__epsilon_decay
        self.__epsilon = max(self.__epsilon, self.__epsilon_min)
        if np.random.random() < self.__epsilon:
            return np.random.randint(self.__num_outputs)
        else:
            return np.argmax(self.__model.predict(state))
    
    def act2(self, state : np.ndarray):
        """state-> [-1,1] 연속적인 값 전용"""
        self.__epsilon *= self.__epsilon_decay
        self.__epsilon = max(self.__epsilon, self.__epsilon_min)
        if np.random.random() < self.__epsilon:
            return (np.random.random(self.__num_outputs) * 2 - 1).reshape((state.shape[0], -1))
        else:
            return self.__model.predict(state)
    
    def remember(self, state : np.ndarray, action : int, action_state : np.ndarray, action_reward : float, action_done : bool):
        self.__buffer[0][self.__buffer_index] = state
        self.__buffer[1][self.__buffer_index] = action
        self.__buffer[2][self.__buffer_index] = action_state
        self.__buffer[3][self.__buffer_index] = action_reward
        self.__buffer[4][self.__buffer_index] = action_done

        self.__buffer_index = (self.__buffer_index + 1) % self.__buffer_size
        if self.__buffer_len < self.__buffer_size:
            self.__buffer_len += 1
    
    def update(self, dis = 0.85):
        if self.__buffer_len < self.batch_size:
            return
        
        indexs = np.random.randint(0, self.__buffer_len, self.batch_size)
        self.states[:] = self.__buffer[0][indexs]
        self.actions[:] = self.__buffer[1][indexs]
        self.actions_state[:] = self.__buffer[2][indexs]
        self.actions_reward[:] = self.__buffer[3][indexs]
        self.actions_done[:] = self.__buffer[4][indexs]
        Q= self.__model.predict(self.states)
        Q[range(self.batch_size),self.actions] = self.actions_reward +\
            (1 - self.actions_done) * (dis * self.__previous_model.predict(self.actions_state).max(-1))
    
        return self.__model.fit(self.states, Q, epochs=1, verbose=0)

        
    
    def updatePreviousModel(self):
        # 만약 약하게 업데이트 하고 싶으면 previous_model과 model간에 비율을 주고 업데이트 해도 좋음.
        mw = self.__model.get_weights()
        pmw = self.__previous_model.get_weights()
        for index in range(len(pmw)):
            pmw[index] = 0.1 * mw[index] + 0.9 * pmw[index]
        self.__previous_model.set_weights(pmw)
        #self.__previous_model.set_weights(self.__model.get_weights())

    @staticmethod
    def createModel(num_inputs : int, num_outputs : int):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(num_inputs, activation=tf.keras.layers.ReLU(), kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_inputs * 2, activation=tf.keras.layers.ReLU(), kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_outputs * 2, activation=tf.keras.layers.ReLU(), kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_inputs, activation=tf.keras.layers.ReLU(), kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(num_outputs, activation="tanh")
        ])

        model.compile(
            optimizer = tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy']
        )

        model.build([1, num_inputs])

        return model
    
    
    
    