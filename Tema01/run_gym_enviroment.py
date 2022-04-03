# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:06:39 2019

@author: Santiago
"""

import gym 
import sys

def run_gym_enviroment(argv):
    ##el primer parametro es el nombre del entorno
    enviroment = gym.make(argv[1])
    enviroment.reset()
    
    for _ in range(int(argv[2])):
        enviroment.render()
        enviroment.step(enviroment.action_space.sample())
    enviroment.close()
    
if __name__ == '__main__':
    run_gym_enviroment(sys.argv)