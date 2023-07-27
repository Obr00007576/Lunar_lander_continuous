import gym
import copy
import torch
import random
import os
import numpy as np
import json
from ReplayBuffer import ReplayBuffer
from DQNLL import device, DQNLL, action_space_len, action_type


def main():

    configure_path = './checkpoint.pth'
    env = gym.make("LunarLander-v2", render_mode='human', continuous = True)
    observation, info = env.reset() 
    model = DQNLL()
    if os.path.isfile(configure_path):
        checkpoint = torch.load(configure_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    reward_sum = 0
    n, c = 0, 0

    while n < 10000:
        action = model.get_action_value([observation])[0]
        main_engine = action[0].item()/(action_space_len/action_type-1)
        lateral_engine = action[1].item()/(action_space_len/action_type-1)
        lateral_engine = lateral_engine*2 - 1
        new_observation, reward, terminated, truncated, info = env.step([main_engine, lateral_engine])

        reward_sum += reward
    
        observation = new_observation
        if terminated or truncated:
            n+=1
            print(f'epoch {n}: {reward_sum} {model.scheduler.get_last_lr()}')
            reward_sum=0
            observation, info = env.reset()
        c+=1
    env.close()

if __name__=='__main__':
    main()