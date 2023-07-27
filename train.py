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
    epsilon = 0.05
    epsilon_decay = 0.99
    epsilon_min=0.0001

    configure_path = './checkpoint.pth'
    env = gym.make("LunarLander-v2", render_mode=None, continuous = True)
    observation, info = env.reset(seed=42) 
    model = DQNLL()
    if os.path.isfile(configure_path):
        checkpoint = torch.load(configure_path)
        epsilon = checkpoint['epsilon']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.optimizer.param_groups[0]['lr']=0.00001
    model_hat = copy.deepcopy(model)
    reward_sum = 0
    replays = ReplayBuffer()
    n, c = 0, 0

    while n < 10000:
        action = None
        if random.random()>epsilon:
            action = model.get_action_value([observation])[0]
        else:
            action = torch.tensor([random.randint(0, action_space_len/action_type-1), random.randint(0, action_space_len/action_type-1)]).to(device)
        main_engine = action[0].item()/(action_space_len/action_type-1)
        lateral_engine = action[1].item()/(action_space_len/action_type-1)
        lateral_engine = lateral_engine*2 - 1
        new_observation, reward, terminated, truncated, info = env.step([main_engine, lateral_engine])

        reward_sum += reward
        replays.add(observation, action, reward, new_observation)
        obs, acts, rews, nobs = replays.get_batch()
        model.train_batch( 
            torch.FloatTensor(obs).to(device), 
            torch.FloatTensor(rews).to(device), 
            model_hat(torch.FloatTensor(nobs).to(device)), 
            torch.stack(acts).to(device)
            )
    
        observation = new_observation
        if terminated or truncated:
            if reward_sum>0:
                epsilon*=epsilon_decay
                if epsilon < epsilon_min:
                    epsilon = epsilon_min
            model_hat = copy.deepcopy(model)
            if n%100==0:
                torch.save({
                'epsilon': epsilon, 
                'model_state_dict': model.state_dict(),
                'optim_state_dict': model.optimizer.state_dict(),
                'scheduler_state_dict': model.scheduler.state_dict()
                }
                , configure_path)
            n+=1
            if n%1500==0:
                model.scheduler_step()
            print(f'epoch {n}: {reward_sum} {epsilon} {model.scheduler.get_last_lr()}')
            reward_sum=0
            observation, info = env.reset()
        c+=1
    env.close()

if __name__=='__main__':
    main()