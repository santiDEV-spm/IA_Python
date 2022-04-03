# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:04:40 2019

@author: Santiago
"""

import gym

enviroment = gym.make('MountainCar-v0')
MAX_NUM_EPISODES = 1000

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = enviroment.reset()
    total_reward = 0.0 #variable para guardar la recompensa obtenida en cada episodio
    step = 0
    
    while not done:
        enviroment.render()
        action = enviroment.action_space.sample()#accion aleatoria que despues remplazaremos por la accion del agente 
        next_state, reward, done, info = enviroment.step(action)
        total_reward += reward
        step+= 1
        obs = next_state
    
    print('\n EPISODIO NUMERO {} finalizado con {} itereaciones. Recompensa final {}'.format(episode, step+1, total_reward))

enviroment.close()
        