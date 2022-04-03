# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 22:09:04 2019

@author: Santiago
"""



"""
 box -> R^n
gym.spaces.Box(low=-10, high=10, shape=(2,))

 discrete -> numeros entre 0 y n-1
gym.spaces.Discrete(5)

Dict
gym.spaces.Dict({
        'position':gym.spaces.Discrete(3),
        'velocity': gym.spaces.Discrete(2)
        })
"""


import gym
from gym.spaces import *
import sys


def print_spaces(space):
    print(space)
    if isinstance(space, Box):#COMPRUEBA SI EL ESPACIO COMO PARAMETRO ES DEL TIPO BOX
        print('\nCOTA INFERIOR ', space.low)
        print('\nCOTA SUPERIOR ', space.high)

if __name__ == '__main__':
    enviroment = gym.make(sys.argv[1]) ## el usuario llama al script cxon el entorno
    print(' ---------- ESPACIO DE ESTADOS ----------')
    print_spaces(enviroment.observation_space)
    print('\n ---------- ESPACIO DE ACCIONES ----------')
    print_spaces(enviroment.action_space)
    try:
        print("Descripcion de las acciones: ", enviroment.unwrapped.get_action_meanings())
        
    except AttributeError:
        pass