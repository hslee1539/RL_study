from RL_study.memory import Memory
from RL_study.sequential import Sequential
import tensorflow as tf
import numpy as np
# 인텔리센스 에러용
np.random = np.random

from typing import Tuple

def Actor(state_size, action_size, action_scale):
    actor_in, actor_out = Sequential(
        tf.keras.layers.Input(state_size),
        tf.keras.layers.Dense(199, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.GaussianNoise(1.0), # 이 레이어로 탐험 효과도 주고 정규화 효과도 얻음
        tf.keras.layers.Dense(71),
        tf.keras.layers.GaussianNoise(1.0), # 이 레이어로 탐험 효과도 주고 정규화 효과도 얻음
        tf.keras.layers.Dense(action_size, action_size="tanh", kernel_initializer='uniform'),
        tf.keras.layers.Lambda(lambda x: x * action_scale)
    )
    return tf.keras.Model(inputs=actor_in, outputs=actor_out)

def Critic(state_size, action_size):
    state_input, state_output = Sequential(
        tf.keras.layers.Input(state_size),
        tf.keras.layers.Dense(71, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(33, activation="relu", kernel_initializer='he_uniform')
    )

    action_input, action_output = Sequential(
        tf.keras.layers.Input(action_size),
        tf.keras.layers.Dense(71, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(33, activation="relu", kernel_initializer='he_uniform')
    )

    _, output = Sequential(
        tf.keras.layers.concatenate([state_output, action_output]),
        tf.keras.layers.Dense(199, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(33, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(1, activation="linear")
    )
    return tf.keras.Model(inputs=[state_input, action_input], outputs = output)


def create_actor_tensor(state_size : int, action_size : int, action_scale : float) -> Tuple[tf.Tensor]:
    return Sequential(
        tf.keras.layers.Input(state_size),
        tf.keras.layers.Dense(199, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.GaussianNoise(1.0), # 이 레이어로 탐험 효과도 주고 정규화 효과도 얻음
        tf.keras.layers.Dense(71),
        tf.keras.layers.GaussianNoise(1.0), # 이 레이어로 탐험 효과도 주고 정규화 효과도 얻음
        tf.keras.layers.Dense(action_size, action_size="tanh", kernel_initializer='uniform'),
        tf.keras.layers.Lambda(lambda x: x * action_scale)
    )

def create_critic_tensor_v0(state_size : int, action_size : int, actor_out : tf.Tensor) ->Tuple[tf.Tensor]:
    """
    critic 모델을 만듭니다.
    
    ## 문제점

    actor_out tensor를 그대로 연결하여, on policy 처럼 됨.

    효과는 아직 확인하지 않았지만, 오리지날 DDPG 알고리즘이 아니기 떄문에 버림.

    ## 장점

    actor_out을 그대로 연결하여 actor과 critic이 한 모델로 합쳐저 gradient가 자연스롭게 넘거가짐.
    """
    state_input, state_output = Sequential(
        tf.keras.layers.Input(state_size),
        tf.keras.layers.Dense(71, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(33, activation="relu", kernel_initializer='he_uniform')
    )

    _, actor_out_out = Sequential(
        actor_out,
        tf.keras.layers.Dense(71, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(33, activation="relu", kernel_initializer='he_uniform')
    )

    _, output = Sequential(
        tf.keras.layers.concatenate([state_output, actor_out_out]),
        tf.keras.layers.Dense(199, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(33, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(1, activation="linear")
    )
    return state_input, output

def create_critic_tensor_v1(state_size : int, action_size : int) ->Tuple[tf.Tensor]:
    state_input, state_output = Sequential(
        tf.keras.layers.Input(state_size),
        tf.keras.layers.Dense(71, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(33, activation="relu", kernel_initializer='he_uniform')
    )

    action_input, action_out = Sequential(
        tf.keras.layers.Input(action_size),
        tf.keras.layers.Dense(71, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(33, activation="relu", kernel_initializer='he_uniform')
    )

    _, output = Sequential(
        tf.keras.layers.concatenate([state_output, action_out]),
        tf.keras.layers.Dense(199, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(33, activation="relu", kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(1, activation="linear")
    )
    return state_input, action_input, output

class DDPG:
    def __init__(self, state_size : int, action_size : int, action_scale : float):

        self.action_scale = action_scale
        self.action_size = action_size

        self.memory = Memory(state_size, action_size)


        ############    learning param    ##############
        self.epsilon = 0.9
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.0001
        self.memory = Memory(state_size, action_size, 2000)

        ############    model    ##############
        self.actor = Actor(state_size, action_size, action_scale)
        self.critic = Critic(state_size, action_size)
        self.target_actor = Actor(state_size, action_size, action_scale)
        self.target_critic = Critic(state_size, action_size)


        ###########      loss, optimizer       #############
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

        # TODO : update variable 하자 (TAU값 1로 줘서 똑같이 복사 되도록)

    def act(self, states : tf.Variable, stddev : float):
        actions = self.actor(states)
        
        actions += tf.random.normal(shape=actions.shape, stddev=stddev)
        return tf.clip_by_value(actions, -self.action_scale, self.action_scale)

    def train(self, dis = 0.95):
        states, actions, actions_reward, actions_state, actions_done = self.memory.get()

        # input에 대한 gradient를 기록하기 위해
        with tf.GradientTape() as tape:
            # tf.constant가 안되면 tf.Variable로 바꾸자
            actions_state_tensor = tf.constant(actions_state)
            predicted_action = self.target_actor(actions_state_tensor)
            
            predicted_rewards = self.target_critic([
                actions_state_tensor, 
                predicted_action
            ])
            predicted_rewards = actions_reward + (1 - actions_done) * dis * predicted_rewards
            
        tape.gradient()


        pass
    





        


    def train(self, batch_size = 32, gamma = 0.9):
        if self.memory.size < batch_size:
            return None
        states, actions, actions_reward, actions_state, actions_done = self.memory.get(batch_size)
        predicted_reward = self.target_actor_critic.predict_on_batch([actions_state, actions_state])
        actions_reward += gamma * predicted_reward * (1 - actions_done)
        return self.actor_critic.fit([states, states], actions_reward, verbose=0)

    def update(self, tau = 0.01):
        # for문으로 돌리는 것 보다 c로 작성된 ndarray로 배열 만들고 계산하는게 더 빠를 수 있음
        # 코드가 간결해 지는것은 덤
        actor_critic_w = np.array(self.actor_critic.get_weights())
        target_w = np.array(self.target_actor_critic.get_weights())
        target_w = actor_critic_w * tau + target_w * (1 - tau)
        self.target_actor_critic.set_weights(target_w)
    
    def act(self, states : np.ndarray):
        self.epsilon = max(self.epsilon_decay * self.epsilon, self.epsilon_min)
        if np.random.random() < self.epsilon:
            return np.random.random(self.action_size * states.shape[0]).reshape((states.shape[0], self.action_size))
        else:
            return self.actor.predict_on_batch(states)
        
            




