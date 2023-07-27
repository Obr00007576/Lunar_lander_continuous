import numpy as np
import random
batch_size = 1024
buffer_size = 10**4

class ReplayBuffer:
    def __init__(self) -> None:
        self.observation_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.new_obervation_buffer = []

    def add(self, observation, action, reward, new_obervation):
        self.observation_buffer.append(observation)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.new_obervation_buffer.append(new_obervation)
        if len(self.observation_buffer)>buffer_size:
            del self.observation_buffer[0]
            del self.action_buffer[0]
            del self.reward_buffer[0]
            del self.new_obervation_buffer[0]

    def get_batch(self):
        if len(self.observation_buffer)<batch_size:
            return np.array(self.observation_buffer),self.action_buffer,self.reward_buffer,np.array(self.new_obervation_buffer)
        else:
            index_list = random.sample([i for i in range(len(self.observation_buffer))], k=batch_size)
            observation_sample = [self.observation_buffer[i] for i in index_list]
            action_sample = [self.action_buffer[i] for i in index_list]
            reward_sample = [self.reward_buffer[i] for i in index_list]
            new_observation_sample = [self.new_obervation_buffer[i] for i in index_list]
            return np.array(observation_sample), action_sample, reward_sample, np.array(new_observation_sample)