#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:39:43 2024

@author: brendonmcguinness
"""

#test file to run a community and try some of the functions

from Community import Community

#here you can initialize a Community with S species and R resources
c = Community(10, 3)
#sets the initial conditions (n0=1e6,c0=1e-3,a0=5e-7*Dir(Unif(1,5))) and the supply vector
#setting inou=False or True will be out vs in of convex hull. inou=None will just be random
c.setInitialConditions(inou=False)
c.changeTimeScale(50000, 50000)
c.setD(5e-7) #sets plasticity rate
c.runModel() #runs Model
c.getSteadyState() # returns steady state

#here we can plot the abundance dynamics
c.plotTimeSeries()
#here we can plot the simplex (either at equilibrium or initial), (with or without centroid), and save or not
#the plotSimplex method only works when R=3 (N_R=R in code) thus simplex is a triangle
c.plotSimplex(eq=False,centroid=False,save=False)
c.plotSimplex(eq=True,centroid=False,save=False)