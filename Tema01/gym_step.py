# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:32:18 2019

@author: Santiago
"""

import gym


enviroment = gym.make("Qbert-v0")

MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

for episode in range(MAX_NUM_EPISODES):
    obs = enviroment.reset()
    
    for step in range(MAX_STEPS_PER_EPISODE):
        enviroment.render()
        action = enviroment.action_space.sample()##decision aleatoria
        next_state, reward, done, info = enviroment.step(action)
        obs = next_state
        
        if done is True:
            print('\n Episodio #{} terminado en {} steps'.format(episode, step+1))
            break
        
        
        
enviroment.close()