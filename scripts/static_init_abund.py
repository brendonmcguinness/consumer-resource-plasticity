#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:35:21 2024

@author: brendonmcguinness
"""

from Community import Community
from utils import bary2cart, simplex_vertices
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from random import choice
import seaborn as sns
import pandas as pd


S=2
R=3
c = Community(S,R)

c.setInitialConditions()
c.d[0] = 0
c.d[1] = 1e-6
N = 20
M = 20
n0_1 = np.linspace(1e2,1e9,N)
n0_1 = np.logspace(1,12,N)

n0_2 = 1e4
dist_p_s = np.zeros(N) 
c.setInitialConditions()
c.changeTimeScale(int(1e5),int(1e6))
#for j in range(M):
for i in range(N):
    
    c.n0 = np.array([n0_1[i],n0_2])
    c.z0 = np.concatenate((c.n0, c.c0, c.a0.flatten(), c.E0), axis=None)
    c.runModel()
    dist_p_s[i] = np.linalg.norm(bary2cart((c.a[-1,1,:]/c.a[-1,1,:].sum()),corners=simplex_vertices(R-1))[0]-bary2cart((c.a[-1,0,:]/c.a[-1,0,:].sum()),corners=simplex_vertices(R-1))[0])

# Compute the mean and standard error of dist_p_s
#mean_dist_p_s = dist_p_s.mean(axis=0)
#sem_dist_p_s = dist_p_s.std(axis=0) / np.sqrt(M)

plt.figure()
#plt.errorbar(n0_1/n0_2, mean_dist_p_s, yerr=sem_dist_p_s, fmt='-o', capsize=3)
plt.scatter(n0_1/n0_2,dist_p_s)
plt.xscale('log')
plt.ylabel('distance of eq. plastic to static sp.',fontsize=12)
plt.xlabel('initial ratio of static:plastic sp.',fontsize=12)
#plt.savefig('SI3_static2plastic.pdf')
plt.show()