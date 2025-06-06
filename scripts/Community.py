#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 18:46:53 2023

@author: brendonmcguinness
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.lines as mlines
from scipy.spatial import ConvexHull
import matplotlib

from utils import model
#from utils import model_simple
from utils import pickPointInitDist
from utils import full_point_in_hull
from utils import simplex_vertices
from utils import bary2cart
from utils import weighted_centroid
from utils import chooseAbundWeights
from utils import avg_eq_time
from utils import shannon_diversity
from utils import supply_to_weighted_centroid
from utils import centeroidnp
from utils import pick_inout_hull
from utils import distanceN0
from utils import distanceA0
from utils import compute_jacobian
from utils import pick_inout_hull_a0
from utils import compute_jacobian_centDiff

import warnings

class Community(object):
    """
    Represents a biological community with a set of properties and behaviors.
    
    Attributes:
        S (int): Total number of species in the community.
        R (int): Total number of resources available in the community.
        ... [other instance variables]
    
    Methods:
        resetInitialConditions(): Resets initial conditions to None.
        setInitialConditions(): Sets the initial conditions for the community model.
        setD(dnew): Sets the acclimation speed for the community.
        runModel(): Runs the model for the community dynamics.
        getSteadyState(): Retrieves the steady state of the community model.
        getSA(): Returns the supply rate and initial uptake matrix.
        getRanks(): Returns the rank distribution of species based on density.
        onePlastic(): Sets all species to non-plastic except for the first.
        changeTimeScale(tend, numt): Changes the time scale of the community model.
        plotTimeSeries(title=None): Plots a time series of the community dynamics.
    """
    def __init__(self,S,R):
        """
        Initializes a Community instance.
        
        Parameters:
            S (int): Total number of species in the community.
            R (int): Total number of resources available in the community.
        """
        self.S = S
        self.R = R
        
        self.v = np.random.uniform(10e7, 10e7, R)
        self.dlta = np.random.uniform(5e-3, 5e-3, S) 
        self.Q = np.ones(S)*1e-5 #10e-5
        #self.Q = np.random.uniform(1e-6,1e-5,S)
        self.eps = np.random.uniform(1e-6,2e-5,S)
        self.E0 = np.random.uniform(self.Q*self.dlta, self.Q*self.dlta, S)
        self.K = np.random.uniform(1e-4, 1e-4, R) #10e-6
        self.u = np.zeros(R)
        #self.s = np.random.uniform(10e-5,10e-2,R)
        
        #self.s = np.array([0.06889729, 0.01852063, 0.02463057])
        self.d = 50e-7*np.ones(S) 
        self.num_t = 500000
        self.t_end = 50000
        self.t = np.linspace(0,self.t_end, self.num_t)
        self.gamma = 1.0
        self.k = 0.75
        self.n0 = None
        self.c0 = None
        self.a0 = None
        self.dlta0 = None
        self.dltat = None
        self.E = None
        self.z0 = None
        self.n = None
        self.c = None
        self.a = None
        self.ww = None
        self.ev = None
        
    def resetInitialConditions(self):
        """Resets initial conditions for n, c, a, and z to None."""
        self.n0 = None
        self.c0 = None
        self.a0 = None
        self.dlta0 = None
        self.z0 = None
        self.n = None
        self.c = None
        self.a = None
        return None
    
    
    # def setInitialConditionsOG(self,inou=None):
    #     """Sets random initial conditions for the community based on species and resources."""
    #     self.E0 = np.random.uniform(self.Q*self.dlta, self.Q*self.dlta, self.S)
    #     if inou==None:
    #         self.a0 = np.zeros((self.S, self.R), dtype=float)
    #         for i in range(0, self.S):
    #             #change back now uniform random
    #             dirc = np.random.randint(1,5,size=self.R) #sub dirc
    #             self.a0[i, :] = np.random.dirichlet(dirc*np.ones(self.R), size=1) * self.E0[i]
    #             #self.a0[i, :] = np.random.dirichlet(np.ones(self.R), size=1) * self.E0[i]
    #         #now s is in center change back
    #         self.s = np.random.uniform(10e-5,10e-2,self.R) #5,2
    #         self.s = (self.s / self.s.sum())*10e-2 #S total will always be the same -> nmaybe change later
    #         #self.s = np.random.uniform(10e-3,10e-3,self.R) #5,2
    #     elif inou==True:
    #         self.s,self.a0 = pick_inout_hull(self.S,self.R,self.E0,a=10e-5,b=10e-2,inout=True,di=np.random.randint(1,5,size=self.R))
    #         self.s = (self.s / self.s.sum())*10e-2

    #     else:
    #         self.s, self.a0 = pick_inout_hull(self.S,self.R,self.E0,a=10e-5,b=10e-2,inout=False,di=np.random.randint(1,5,size=self.R))
    #         self.s = (self.s / self.s.sum())*10e-2


    #     self.n0 = np.random.uniform(1e6, 1e6, self.S) #10e-6
    #     self.c0 = np.random.uniform(1e-3, 1e-3, self.R) #10e-3
    #     self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten(), self.E0), axis=None)
    #     return None
    
    def setInitialConditions(self,inou=None):
        """Sets random initial conditions for the community based on species and resources."""
        self.E0 = np.random.uniform(self.Q*self.dlta, self.Q*self.dlta, self.S)
        if inou==None:
            self.a0 = np.zeros((self.S, self.R), dtype=float)
            for i in range(0, self.S):
                #change back now uniform random
                dirc = np.random.randint(1,5,size=self.R) #sub dirc
                #self.a0[i, :] = np.random.dirichlet(dirc*np.ones(self.R), size=1) * self.E0[i]
                self.a0[i, :] = np.random.dirichlet(np.ones(self.R), size=1) * self.E0[i]
            #now s is in center change back
            self.s = np.random.uniform(10e-5,10e-2,self.R) #5,2
            self.s = (self.s / self.s.sum())*10e-2 #S total will always be the same -> nmaybe change later
            #self.s = np.random.uniform(10e-3,10e-3,self.R) #5,2
        elif inou==True:
            self.s,self.a0 = pick_inout_hull(self.S,self.R,self.E0,a=10e-5,b=10e-2,inout=True,di=np.random.randint(1,5,size=self.R))
            self.s = (self.s / self.s.sum())*10e-2

        else:
            self.s, self.a0 = pick_inout_hull(self.S,self.R,self.E0,a=10e-5,b=10e-2,inout=False,di=np.random.randint(1,5,size=self.R))
            self.s = (self.s / self.s.sum())*10e-2


        self.n0 = np.random.uniform(1e6, 1e6, self.S) #10e-6
        self.c0 = np.random.uniform(1e-3, 1e-3, self.R) #10e-3
        self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten()), axis=None)
        return None
    
    def setInitialConditionsManual(self,a0=np.array([None]),n0=np.array([None]),c0=np.array([None]),sameS=True):
        """
        

        Parameters
        ----------
        a0 : TYPE
            DESCRIPTION.
        n0 : TYPE, optional
            DESCRIPTION. The default is None.
        c0 : TYPE, optional
            DESCRIPTION. The default is None.
        sameS : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        
        if sameS==False:
            self.s = np.random.uniform(10e-5,10e-2,self.R)
            
        if n0.any()==None:
            self.n0 = np.random.uniform(1e6, 1e6, self.S) #10e-6
        else:
            self.n0 = n0
           
        if c0.any()==None:
            self.c0 = np.random.uniform(1e-3, 1e-3, self.R) #10e-3
        else:
            self.c0 = c0
        if a0.any()==None:
            self.a0 = np.zeros((self.S, self.R), dtype=float)
            for i in range(0, self.S):
                dirc = np.random.randint(1,5,size=self.R) #sub dirc
                self.a0[i, :] = np.random.dirichlet(dirc*np.ones(self.R), size=1) * self.E0[i]
        else:
            self.a0 = a0

        self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten()), axis=None)
        return None
 
    def setInitialConditionsSameS(self,inou=None):
        """Sets random initial conditions for the community based on species and resources."""
        if inou==None:

            self.a0 = np.zeros((self.S, self.R), dtype=float)
            for i in range(0, self.S):
                dirc = np.random.randint(1,5,size=self.R) #sub dirc
                self.a0[i, :] = np.random.dirichlet(dirc*np.ones(self.R), size=1) * self.E0[i]
            self.s = np.ones(self.R) / self.R  #5,2
        elif inou==True:
            self.s = np.ones(self.R) / self.R  #5,2
            self.a0 = pick_inout_hull_a0(self.s,self.S,self.R,self.E0,inout=inou)
        else:
            self.s = np.ones(self.R) / self.R  #5,2
            self.a0 = pick_inout_hull_a0(self.s,self.S,self.R,self.E0,inout=inou)
            
        self.n0 = np.random.uniform(1e6, 1e6, self.S) #10e-6
        self.c0 = np.random.uniform(1e-3, 1e-3, self.R) #10e-3
        self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten()), axis=None)
        return None
    
    def setInitialAlphaRandom(self):
        self.a0 = np.zeros((self.S, self.R), dtype=float)
        for i in range(0, self.S):
            dirc = np.random.randint(1,5,size=self.R) #sub dirc
            self.a0[i, :] = np.random.dirichlet(dirc*np.ones(self.R), size=1) * self.E0[i]
        return self.a0
    
    
    def setInitialConditionsDist(self, dist=0.0):
        s = np.ones(self.R) / self.R  # Uniform distribution over R dimensions
    
        for count in range(300):  # Maximum of 100 attempts
            a0 = self.setInitialAlphaRandom()
            p = pickPointInitDist(s, dist)
    
            if full_point_in_hull(p, a0):
                if not full_point_in_hull(s, a0) and dist > 0.08:
                    self.a0 = a0
                    self.s = s
                    n = chooseAbundWeights(a0, p)
                    self.n0 = n * 1e6
                    self.c0 = np.random.uniform(1e-3, 1e-3, self.R)  # Uniform random values for c0
                if full_point_in_hull(s, a0) and dist < 0.08: #if dist is small we are keeping supply dist at center
                    self.a0 = a0
                    self.s = s
                    n = chooseAbundWeights(a0, p)
                    self.n0 = n * 1e6
                    self.c0 = np.random.uniform(1e-3, 1e-3, self.R) 
                # Print the norm of the error vector
                """
                print(np.linalg.norm(
                    np.dot(
                        (bary2cart(self.a0, corners=simplex_vertices(self.R-1))[0]).T,
                        self.n0 / self.n0.sum()
                    ) - bary2cart(self.s, corners=simplex_vertices(self.R-1))[0]
                ))
                """
                # Concatenate all initial conditions into a single array
                self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten(), self.E0), axis=None)
                return np.linalg.norm(
                    np.dot(
                        (bary2cart(self.a0, corners=simplex_vertices(self.R-1))[0]).T,
                        self.n0 / self.n0.sum()
                    ) - bary2cart(self.s, corners=simplex_vertices(self.R-1))[0]
                )
            
        # If no suitable point is found after the maximum attempts, handle the failure case
        return -1
    
    def setInitialAlpha(self,a0):
        self.a0 = a0
        return None
    
    # def runModelSimple(self,ss=False):
    #     """Executes the ODE model for the community dynamics in simplest form."""
    #     max_attempts = 10
    #     attempt = 0
    #     while attempt < max_attempts:
                
    #         try:
    #             with warnings.catch_warnings():
    #                 warnings.filterwarnings('error')
                        
    #                     # Use odeint here
    #                 z = odeint(model_simple,self.z0,self.t,args=(self.S,self.R,self.v,self.d,self.dlta,self.s,self.u,self.K,self.Q))
    #                 self.n = z[:,0:self.S]
    #                 self.c = z[:,self.S:self.S+self.R]
    #                 at = z[:,self.S+self.R:self.S+self.R+self.S*self.R]
    #                 self.a = np.reshape(at,(self.num_t,self.S,self.R))  
                    
    #                 if ss==False:
    #                     return None
    #                 else:
    #                     neq = self.n[-1,:]
    #                     ceq = self.c[-1,:]
    #                     aeq = self.a[-1,:,:]
    #                     return neq,ceq,aeq
                     
                  
    #                 break
    #         except Warning as w:
    #             print(f"Caught a warning: {w}")
    #             self.resetInitialConditions()
    #             #check debug
    #             self.setInitialConditions()
    #             #self.setInitialConditionsDelta()
    #             self.runModel()
    #             attempt += 1
                
    #         except Exception as e:
    #             print(f"Caught an error: {e}")
    #             attempt +=1
    #     if attempt == max_attempts:
    #         print("Max retry attempts reached.")
            
    #     if ss==True:
    #         return self.n0,self.c0,self.a0
        
    def perturbInitialDensity(self,CV=0.03):
        """
        

        Parameters
        ----------
        CV : FLOAT, coefficient of variation
            DESCRIPTION. The default is 0.03.

        Returns
        -------
        None.

        """
        self.n0 = self.n0 + np.random.normal(0,CV*np.mean(self.n0),self.S)
        self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten(), self.E0), axis=None)

        if (self.n0).any() < 0:
            self.perturbInitialDensity()
        else:
            return None
        
    def perturbInitialDensityA0(self,CV=0.03):
        """
        

        Parameters
        ----------
        CV : FLOAT, coefficient of variation
            DESCRIPTION. The default is 0.03.

        Returns
        -------
        None.
        """
        
        self.a0 = self.a0 + np.random.normal(0,CV*np.mean(self.a0),(self.S,self.R))
        self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten(), self.E0), axis=None)

        if (self.a0).any() < 0:
            self.perturbInitialDensityA0()
        else:
            return None

         
    def getLyapunovExp(self,N=3,CV=0.01):
        """
        

        Parameters
        ----------
        N : INT, number of samples of random initial perturbatinos
            DESCRIPTION. The default is 3.

        Returns
        -------
        FLOAT
            Lyapunov exponent for small perturbations in initial densities 

        """
        #for dens
        dzt = np.zeros(N)
        dz0 = np.zeros(N)
        #save initial conds
        z0 = self.z0
        n00 = self.n0 /self.n0.sum()
        
        if self.n[-1,:].any() > 0:
            neq,ceq,aeq = self.getSteadyState()
        else:
            neq,ceq,aeq = self.runModel(ss=True)
            
        for i in range(N):
            self.perturbInitialDensity(CV=CV)
            dz0[i] = distanceN0(n00,(self.n0/self.n0.sum()))
            neql = self.runModel(ss=True)[0]
            dzt[i] = distanceN0(neq,neql)
        #give back OG initial conds
        self.z0 = z0
        return (1/(self.t_end)) * np.log(dzt/dz0)
            
    def getLyapunovExpA0(self,N=3,CV=0.01,t=0.0):
        """
        

        Parameters
        ----------
        N : INT, number of samples of random initial perturbatinos
            DESCRIPTION. The default is 3.

        Returns
        -------
        FLOAT
            Lyapunov exponent for small perturbations in initial traits 

        """
        #for traits
        dzt = np.zeros(N)
        dz0 = np.zeros(N)
        

        #save initial conds
        z0 = self.z0
        a00 = self.a0 / self.E0[:,None]
        if self.n is not None:
            neq,ceq,aeq = self.getSteadyState()
        else:
            neq,ceq,aeq = self.runModel(ss=True)
            
   
        for i in range(N):
            self.perturbInitialDensityA0(CV=CV)
            dz0[i] = distanceA0(a00,(self.a0/self.E0[:,None]),self.E0)
            aeql = self.runModel(ss=True)[2]
            dzt[i] = distanceA0(aeq,aeql,self.E0)
        #give back OG initial conds            
        self.z0 = z0        
        return (1/((self.t_end))) * np.log(dzt/dz0)
    
    def perturbDensityA(self,a,c0,n0,noise):
        """
        

        Parameters
        ----------
        CV : FLOAT, coefficient of variation
            DESCRIPTION. The default is 0.03.

        Returns
        -------
        None.
        """
        self.c0 = c0
        self.n0 = n0
        self.a0 = a + noise
        self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten(), self.E0), axis=None)

        if (self.a0).any() < 0:
            print('a goes negative')
            self.a0 = a
            self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten(), self.E0), axis=None)
            self.perturbDensityA()
                    
        neq = self.runModel(ss=True)[0]
        return neq,self.a0
            
           
    def seedNewEnvironment(self,dilution,noise=None):
        """ seed new environment ()
        

        Parameters
        ----------
        dilution : dilution factor for population
        noise : array of size S denotes noise added to bottleneck population that is seeding new environment

        Returns
        -------
        None.

        """
        neq,ceq,aeq = self.getSteadyState()
        total_biomass = neq.sum() * dilution
        if noise.any() == None:
            noise = np.random.normal(loc=0,scale=total_biomass/1000,size=self.S)
        self.n0 = (dilution*neq)+noise
        self.n0[self.n0<0] = 0
        self.c0 = np.random.uniform(1e-3, 1e-3, self.R) # or could be ceq if same environent
        self.a0 = aeq
        self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten(), self.E0), axis=None)
        return None
        
        
    def setInitialConditionsDelta(self,inou=None):
        """Sets random initial conditions for the community based on species and resources."""
        if inou==None:
            self.a0 = np.zeros((self.S, self.R), dtype=float)
            for i in range(0, self.S):
                dirc = np.random.randint(1,5,size=self.R)
                self.a0[i, :] = np.random.dirichlet(dirc, size=1) * self.E0[i]
            self.s = np.random.uniform(10e-5,10e-2,self.R)
        elif inou==True:
            self.s,self.a0 = pick_inout_hull(self.S,self.R,self.E0,a=10e-5,b=10e-2,inout=True,di=np.random.randint(1,5,size=self.R))
        else:
            self.s, self.a0 = pick_inout_hull(self.S,self.R,self.E0,a=10e-5,b=10e-2,inout=False,di=np.random.randint(1,5,size=self.R))

        self.n0 = np.random.uniform(1e5, 1e6, self.S) #10e-6
        self.c0 = np.random.uniform(1e-3, 1e-3, self.R) #10e-3
        self.dlta0 = np.random.uniform(self.E0 / self.Q, self.E0 / self.Q, self.S)
        self.z0 = np.concatenate((self.n0, self.c0, self.a0.flatten(), self.E0, self.dlta0), axis=None)
        return None
    
    def changeStot(self,Stot=10e-2):
        self.s = self.s/self.s.sum() * Stot
        return None

    
    def setD(self,dnew):
        """
        Updates the acclimation velocity for each species in the community.
        
        Parameters:
            dnew (float): The new acclimation velocity value to set.
        """
        self.d = dnew*np.ones(self.S)
        return None
    
    
    def runModel(self,ss=False):
        """Executes the ODE model for the community dynamics."""
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
                
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                        
                        # Use odeint here
                    z = odeint(model,self.z0,self.t,args=(self.S,self.R,self.v,self.d,self.dlta,self.s,self.K,self.Q))
            
                    self.n = z[:,0:self.S]
                    self.c = z[:,self.S:self.S+self.R]
                    at = z[:,self.S+self.R:self.S+self.R+self.S*self.R]
                    self.a = np.reshape(at,(self.num_t,self.S,self.R))  
                    
                    if ss==False:
                        return None
                    else:
                        neq = self.n[-1,:]
                        ceq = self.c[-1,:]
                        aeq = self.a[-1,:,:]
                        return neq,ceq,aeq
                     
                  
                    break
            except Warning as w:
                print(f"Caught a warning: {w}")
                self.resetInitialConditions()
                #check debug
                self.setInitialConditions()
                #self.setInitialConditionsDelta()
                self.runModel()
                attempt += 1
                
            except Exception as e:
                print(f"Caught an error: {e}")
                attempt +=1
        if attempt == max_attempts:
            print("Max retry attempts reached.")
            
        if ss==True:
            return self.n0,self.c0,self.a0
        
    
    def getSteadyState(self):
        """
        Returns:
            tuple: Steady state values for n (species densities), c (resources), and a (attack matrix).
        """
        neq = self.n[-1,:]
        ceq = self.c[-1,:]
        aeq = self.a[-1,:,:]
        return neq,ceq,aeq
    
        
    def getSA(self):
        """
        Returns:
            tuple: Resource values and initial uptake strategy matrix.
        """
        return self.s,self.a0
    
    def getRanks(self):
        """
        Ranks species based on their densities.
        
        Returns:
            np.ndarray: Array of densities sorted by rank.
        """
        neq = self.n[-1,:]/self.n[-1,:].sum()
        ranks = np.argsort(neq)[::-1]
        rd = neq[ranks]
        rd[rd < 0] = 1e-30
        return rd
    
    def onePlastic(self,dnew):
        """Sets all species to non-plastic, except for the first species which remains plastic."""
        self.d = np.zeros(self.S)
        self.d[0] = dnew
        return None


    def changeTimeScale(self,tend,numt):
        """
        Adjusts the time scale for the model simulation.
        
        Parameters:
            tend (float): End time for the simulation.
            numt (int): Number of time points in the simulation.
        """
        self.num_t = numt
        self.t_end = tend
        self.t = np.linspace(0,tend, numt)
        return None
        
    def plotTimeSeries(self,title=None):
        """
        Plots the time series of species densities over time.
        
        Parameters:
            title (str, optional): Title for the plot. Default is None.
        """
        nrel = self.n / self.n[-1,:].sum()
        plt.figure()
        plt.semilogy(self.t,nrel)

        #for i in range(self.S):
        #    plt.semilogy(self.t,self.n/self.n[-1,:].sum()) #color=colours[i])
        plt.set_cmap('tab10')
        plt.ylabel('density')
        plt.xlabel('time')
        #plt.xlim([-int(endt/25),self.t[idx]*self.dlta[0]])
        plt.title(title)
        
        return None        

    def plotTraitTimeSeries(self,title=None):
        """
        Plots the time series of species densities over time.
        
        Parameters:
            title (str, optional): Title for the plot. Default is None.
        """
        plt.figure()
        for i in range(self.S):
            plt.plot(self.t*self.dlta[0],self.a[:,i,:]/(self.a[-1,i,:]).sum()) #color=colours[i])
        plt.set_cmap('tab10')
        plt.ylabel('traits')
        plt.xlabel('time (1/$\delta$)')
        plt.ylim(0,1.3)
        plt.xlim([-1,self.t[-1]*self.dlta[0]/5])
        plt.title(title)
        
        return None   
    
    def plotSimplex(self,eq=True,centroid=False,save=False):
        #figure out how to fix
        if eq==True:
            n,c_eq,a_eq = self.getSteadyState()
        else:
            a_eq = self.getSA()[1]
            n = self.n0
        #make simplex
        #ind_del = 4
        #get barycentric coordinates
        #ac,corners = bary2cart(a_eq)
        ac,corners = bary2cart(a_eq,corners=simplex_vertices(self.R-1))
        
        hull_total = ConvexHull(corners)   

        plt.figure(dpi=600)
        #plt.plot(ac[:,0], ac[:,1], 'o')
        cmaps = np.arange(1,self.S+1)
        sizes = 900*n/n.sum()
        sizes[sizes<15] = 15
        #print(sizes)
        if self.S < 11:
            plt.scatter(ac[:,0], ac[:,1],s=sizes, c=cmaps, cmap='tab10')
        else:
            plt.scatter(ac[:,0], ac[:,1],s=sizes, c=cmaps, cmap='tab20')

        if self.S - self.R > 1:
            hull = ConvexHull(ac)
             
            for simplex in hull.simplices:
                plt.plot(ac[simplex, 0], ac[simplex, 1], 'tab:blue',linestyle='solid',alpha=0.01) #linestyle='dashed'
            plt.fill(ac[hull.vertices,0], ac[hull.vertices,1], 'tab:blue', alpha=0.2)    
            

        sc = bary2cart(self.s,corners=simplex_vertices(self.R-1))[0]
        plt.scatter(sc[0],sc[1],s=300,marker='*',color='k',zorder=3)
        for simplex in hull_total.simplices:
            plt.plot(corners[simplex, 0], corners[simplex, 1], 'k-')
            
        if centroid==True:
            cent = weighted_centroid(ac,n)
            #print(cent)
            #cent2 = weighted_centroid(a_eq,n)
            #print('cent2',cent2)
            plt.plot((cent[0],sc[0]),(cent[1],sc[1]),color='r',linestyle='dashed',alpha=0.7,zorder=1)
            plt.scatter(cent[0],cent[1],s=200,marker='d',color='gold',zorder=2)
            #plt.scatter(cent2[0],cent2[1],s=200,marker='d',color='c')
            black_star = mlines.Line2D([], [], color='k', marker='*', linestyle='None',
                          markersize=10, label='supply vector')
            gold_diamond = mlines.Line2D([], [], color='gold', marker='d', linestyle='None',
                          markersize=10, label='initial centroid')
            traits = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None',
                          markersize=10, label='initial traits')
            xx = mlines.Line2D([], [], color='tab:red', linestyle='dashed',
                          markersize=10, label='${||X||}_2$')
            plt.legend(handles=[traits, black_star, gold_diamond,xx],fontsize=12)


        plt.xlim(0,1)
        plt.axis('off')
        plt.text(0.4, 0.78, 'i=1', fontsize=16, fontstyle='italic')
        plt.text(-0.06, -0.06, 'i=2', fontsize=16, fontstyle='italic')
        plt.text(0.86, -0.06, 'i=3', fontsize=16, fontstyle='italic')
        
        if save==True:
            plt.savefig('shuff_pre.pdf')

        if matplotlib.get_backend().lower() != "agg":
            plt.show()
        else:
            plt.close()       
            
    def plotSimplexShuffle(self,eq=True,centroid=False,save=False):
        
        if eq==True:
            n,c_eq,a_eq = self.getSteadyState()
        else:
            a_eq = self.getSA()[1]
            n = self.n0
        #make simplex
        #ind_del = 4
        #get barycentric coordinates
        ac,corners = bary2cart(a_eq,corners=simplex_vertices(self.R-1))
        
        hull = ConvexHull(ac)
        hull_total = ConvexHull(corners)
        plt.figure(dpi=600)
        #plt.plot(ac[:,0], ac[:,1], 'o')
        cmaps = np.arange(1,self.S+1)
        sizes = 900*n/n.sum()
        sizes[sizes<15] = 15
        idx = np.arange(self.S)
        
        np.random.shuffle(idx)
        sizes = sizes[idx]
        print(sizes)
        plt.scatter(ac[:,0], ac[:,1],s=sizes, c=cmaps, cmap='tab10')
        for simplex in hull.simplices:
            plt.plot(ac[simplex, 0], ac[simplex, 1], 'tab:blue',linestyle='solid',alpha=0.01) #linestyle='dashed'
        plt.fill(ac[hull.vertices,0], ac[hull.vertices,1], 'tab:blue', alpha=0.2)    
        

        sc = bary2cart(self.s,corners=simplex_vertices(self.R-1))[0]
        plt.scatter(sc[0],sc[1],s=400,marker='*',color='k')
        for simplex in hull_total.simplices:
            plt.plot(corners[simplex, 0], corners[simplex, 1], 'k-')
            
        if centroid==True:
            cent = weighted_centroid(ac,n[idx])
            #print(cent)
            cent2 = centeroidnp(ac)
            #print('cent2',cent2)
            plt.scatter(cent[0],cent[1],s=225,marker='d',color='gold')
            #plt.scatter(cent2[0],cent2[1],s=200,marker='d',color='c')
            plt.plot((cent[0],sc[0]),(cent[1],sc[1]),color='r',linestyle='dashed',alpha=0.7,zorder=1)
            plt.scatter(cent[0],cent[1],s=200,marker='d',color='gold',zorder=2)
            #plt.scatter(cent2[0],cent2[1],s=200,marker='d',color='c')
            black_star = mlines.Line2D([], [], color='k', marker='*', linestyle='None',
                          markersize=10, label='supply vector')
            gold_diamond = mlines.Line2D([], [], color='gold', marker='d', linestyle='None',
                          markersize=10, label='initial centroid')
            traits = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None',
                          markersize=10, label='traits')
            plt.legend(handles=[traits, black_star, gold_diamond],fontsize=12)

        plt.xlim(0,1)
        plt.axis('off')
        plt.text(0.4, 0.78, 'i=1', fontsize=16, fontstyle='italic')
        plt.text(-0.06, -0.06, 'i=2', fontsize=16, fontstyle='italic')
        plt.text(0.86, -0.06, 'i=3', fontsize=16, fontstyle='italic')
        if save==True:
            plt.savefig('shuffle_fig.pdf')
        
