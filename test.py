#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:39:43 2024

@author: brendonmcguinness
"""

#test file to run a community and try some of the functions

from Community import Community
import matplotlib.pyplot as plt
import numpy as np


#here you can initialize a Community with S species and R resources
c = Community(10, 3)
#sets the initial conditions (n0=1e6,c0=1e-3,a0=5e-7*Dir(Unif(1,5))) and the supply vector
#setting inou=False or True will be out vs in of convex hull. inou=None will just be random
c.setInitialConditions(inou=None)
#this method allows you to choose timescale (t_end,numt) t_end is 
c.changeTimeScale(50000, 50000)
c.setD(1e-6) #sets plasticity rate
c.runModel() #runs Model
neq,ceq,aeq = c.getSteadyState() # returns steady state

#here we can plot the abundance dynamics
c.plotTimeSeries()
c.setInitialConditionsSimple()
c.runModelSimple()
c.plotTimeSeries(title='simple')
#here we can plot the simplex (either at equilibrium or initial), (with or without centroid), and save or not
#the plotSimplex method only works when R=3 (N_R=R in code) thus simplex is a triangle
c.plotSimplex(eq=False,centroid=False,save=False)
c.plotSimplex(eq=True,centroid=False,save=False)


## now let's try another community with a slow plastity rate and initially out of the convex hull
c = Community(10, 3)
c.changeTimeScale(100000, 100000)
c.setInitialConditions(inou=False)
c.setD(1e-7)
#if you look into the Community object constructor we can also change any parameter manually (can do this for any parameter)
c.d = np.ones(10)*1e-7 # does the same as above, sets plasticity rate to 1e-7 for all 10 species
c.runModel()
c.plotTimeSeries()
c.plotSimplex(eq=False,centroid=False,save=False)
c.plotSimplex(eq=True,centroid=False,save=False)

#let's say we wanted to plot the resource dynamics we could do it this way
plt.figure()
plt.semilogy(c.t,c.c)
plt.ylabel('resource concentration (g/mL)')
plt.xlabel('time')
plt.show()