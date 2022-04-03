# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:39:19 2019

@author: Santiago
"""

from gym import envs
import os

enviroments_names = [env.id for env in envs.registry.all()]
file = open('enviroments_names.txt', 'w')

for name in sorted(enviroments_names):
    file.write(name+'\n')
file.close()
print('numero de enviroments: ' + str(len(enviroments_names)))