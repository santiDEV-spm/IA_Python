# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:05:59 2019

@author: Santiago
"""

import gym


enviroment = gym.make("SpaceInvaders-v0")
enviroment.reset()
for _ in range(2000):
    enviroment.render()
    enviroment.step(enviroment.action_space.sample())
    ## next_state -> Object
    ## reward -> float
    ## done -> boolean
    ## info -> diccionary 

enviroment.close()