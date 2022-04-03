# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:48:32 2019

@author: Santiago
"""


import torch
import numpy as np
from libs.Perceptron import SLP
from utils.decay_schedule import LinearDecaySchedule
import random
import gym


MAX_NUM_EPISODES = 100000
STEPS_PER_EPISODE = 300


class SwallowQLearner(object):
    def __init__(self, environment, learning_rate = 0.005, gamma = 0.98):
        self.obs_shape = environment.observation_space.shape
        
        self.action_shape = environment.action_space.n
        self.Q = SLP(self.obs_shape, self.action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = learning_rate)
        
        self.gamma = gamma
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min, 
                                                 max_steps = 0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE)
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
         
    def get_action(self, obs):
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy())   
        return action
        
        
    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()
        
    
        
        
        
    
if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    agent = SwallowQLearner(environment)
    first_episode = True
    episode_rewards = list()
    for episode in range(MAX_NUM_EPISODES):
        obs = environment.reset()
        total_reward = 0.0
        for step in range(STEPS_PER_EPISODE):
            environment.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = environment.step(action)
            agent.learn(obs, action, reward, next_obs)
            
            obs = next_obs
            total_reward += reward
            
            if done is True:
                if first_episode:
                    max_reward = total_reward
                    first_episode = False
                episode_rewards.append(total_reward)
                if total_reward > max_reward:
                    max_reward = total_reward
                print("\nEpisodio#{} finalizado con {} iteraciones. Recompensa = {}, Recompensa media = {}, Mejor recompensa = {}".
                      format(episode, step+1, total_reward, np.mean(episode_rewards), max_reward))
                break
    environment.close()