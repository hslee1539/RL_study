import numpy as np
from typing import Tuple

class Memory:
    def __init__(self, state_size : int, action_size : int, size = 2000):
        self.states = np.zeros([size, state_size])
        self.actions = np.zeros([size, action_size])
        self.action_rewards = np.zeros([size])
        self.action_states = np.zeros([size, state_size])
        self.action_done = np.zeros([size])
        self.pos = 0
        self.size = 0
        self.max_size = size
    
    def store(self, state, action, action_reward, action_state, action_done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.action_rewards[self.pos] = action_reward
        self.action_states[self.pos] = action_state
        self.action_done[self.pos] = action_done

        self.pos = (self.pos + 1) % self.max_size
        #self.size += 1
        #if self.size > self.max_size:
        #    self.size = self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def get(self, batch_size = 32) -> Tuple[np.ndarray]:
        indexs = np.random.randint(0, self.size, batch_size)
        return (
            self.states[indexs],
            self.actions[indexs],
            self.action_rewards[indexs],
            self.action_states[indexs],
            self.action_done[indexs]
        )

     