#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:21:41 2023

@author: brendonmcguinness
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:55:21 2023

@author: brendonmcguinness
"""
#changing plasticity rate d

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from random import choice
import seaborn as sns
import pandas as pd

from Community import Community
from lag_v_budg_fn import full_point_in_hull
from lag_v_budg_fn import supply_to_centroid
from lag_v_budg_fn import shannon_diversity
from lag_v_budg_fn import get_fd_and_centroid
from lag_v_budg_fn import supply_to_weighted_centroid
from lag_v_budg_fn import bary2cart
from lag_v_budg_fn import simplex_vertices
from lag_v_budg_fn import orderParameterCV
from lag_v_budg_fn import pred_rad_from_dist_noscale
from lag_v_budg_fn import pred_rad_from_comp_noscale
from lag_v_budg_fn import avg_eq_time
from lag_v_budg_fn import distance
from lag_v_budg_fn import pred_rad_multiple
from lag_v_budg_fn import pred_abund_from_abund
from lag_v_budg_fn import orderParameter
import statsmodels.api as sm
import time



#here we sweep through a range of d values and change initial abundances to produce figure 4
#10 species 3 resources
#we capture the interaction of initial variation and plasticity rate to measure the 'strength' niche based processes due to longer transient time-scales
N = 100
S = 30
R = 3
#d_list = [1e-7] #[1e-7,5e-7,1e-6,5e-6,1e-5] #1e-7,5e-7,taking two slowest out
d_list = [1e-7,1e-6,1e-5]
numt = 5000000#10000000
tend = 1000000 #100000
sd = np.zeros((N,len(d_list)))
initial_sd = np.zeros(N)
eq_time_c = np.zeros((N,len(d_list)))
eq_time_a = np.zeros((N,len(d_list)))
init_shift_trait = np.zeros((N,len(d_list)))
dist_com_s_init = np.zeros((N,len(d_list)))
dist_com_s_final = np.zeros((N,len(d_list)))
dist_plast = np.zeros((N,len(d_list)))
score = np.zeros((N,len(d_list)))
score0 = np.zeros((N,len(d_list)))
score_og = np.zeros((N,len(d_list)))
score0_og = np.zeros((N,len(d_list)))
scorec0 = np.zeros((N,len(d_list)))
scored0 = np.zeros((N,len(d_list)))
leading_eigenvalue = np.zeros((N,len(d_list)))
le_a = np.zeros((N,len(d_list)))
le_ar = np.zeros((N,len(d_list)))
le_t = np.zeros((N,len(d_list)))

scored = np.zeros((N,len(d_list)))
scorec = np.zeros((N,len(d_list)))
in_out = np.zeros((N,len(d_list)))
fd = np.zeros((N,len(d_list)))
fd_init = np.zeros((N,len(d_list)))
a_fitted = np.zeros((N,len(d_list)))
b_fitted = np.zeros((N,len(d_list)))
peak = np.zeros((N,len(d_list)))

richness = np.zeros((N,len(d_list)))
dd = np.zeros((N,len(d_list)))


ranks = np.zeros((N,len(d_list),S))
abund_pred = np.zeros((N,len(d_list)))
ordT = np.zeros((N,len(d_list),numt))
ordTCV = np.zeros((N,len(d_list),numt))
struct_generated = np.zeros((N,len(d_list)))



for j in range(N):
    
    if j%10==0:
        print('N='+str(j))
    
    
    c = Community(S,R)
    c.changeTimeScale(tend,numt)
    #set to false for 5a
    c.setInitialConditions(inou=None)
    initial_sd[j] = shannon_diversity(c.n0) / shannon_diversity(1/S*np.ones(S))


    
    for i,d in enumerate(d_list):
        

        c.setD(d)

        c.runModel()
        neq,ceq,aeq = c.getSteadyState()
        s,a0 = c.getSA()
        dd[j,i] = d
        if full_point_in_hull(s, a0):
            in_out[j,i] = 1
        else:
            in_out[j,i] = 0
        dist_com_s_init[j,i] = supply_to_weighted_centroid(s,a0,c.n0,c.E0)
        dist_com_s_final[j,i] = supply_to_weighted_centroid(s,aeq,neq,c.E0)
        dist_plast[j] = distance(s,a0[0,:],c.E0)
        
        if ~np.isnan(neq).any() and (neq>0).all():
            #get equilibrium abundances into relative distribution    
            neq = (neq / neq.sum()) 
            #get functinal diversity at final and initial times
            fd[j] = get_fd_and_centroid(aeq)[0]
            fd_init[j] = get_fd_and_centroid(a0)[0]
            #get normalized shannon diversity (evenness)
            sd[j,i] = shannon_diversity(neq) / shannon_diversity(1/S*np.ones(S))
            #structure generated, variability from traits (SSI)
            struct_generated[j,i] = 1 - (sd[j,i] / initial_sd[j])

            
            #get ranked abundances
            ranks[j,i,:] = c.getRanks()
            #get order paramter (fitness variance)
            ordT[j,i,:] = orderParameter(c.n)


        else:
            #j-=1
            print('Failed run on j=',str(j),'and d=',str(d))

df_d1 = pd.DataFrame({'d':dd.flatten(),'hull':in_out.flatten(),'dist_init':dist_com_s_init.flatten(),'struct':struct_generated.flatten(),'pred0':score0.flatten(),'predc0':scorec0.flatten(),'predd0':scored0.flatten(),'predd':scored.flatten(),'predc':scorec.flatten(),'aeq':eq_time_a.flatten(),'ceq':eq_time_c.flatten()})
#df_d230F.to_csv('N100_S30R3_d1e-7_1e-5F_final.csv')




sterr = ranks.std(axis=0) / np.sqrt(N)

plt.figure()
for r in range(len(d_list)):
    plt.semilogy(np.arange(1,S+1),ranks.mean(axis=0)[r,:],label='d='+str(d_list[r]))
    plt.fill_between(np.arange(1,S+1), ranks.mean(axis=0)[r,:] - sterr[r,:], ranks.mean(axis=0)[r,:] + sterr[r,:], alpha=0.2)
plt.xticks(np.arange(5,S+1,5))
plt.xlabel('ranks')
plt.ylabel('abundance')
plt.ylim([1e-4,1])
plt.legend()
plt.show()

sterr_o = ordT.std(axis=0) / np.sqrt(N)
col = plt.cm.viridis(np.linspace(0.1,0.9,3))
plt.figure()
for r in range(len(d_list)):
    plt.semilogx(c.t,ordT[:,r,:].mean(axis=0),label='$d=$'+str(d_list[r]),color=col[r])
    plt.fill_between(c.t, ordT[:,r,:].mean(axis=0) - sterr_o[r,:], ordT[:,r,:].mean(axis=0) + sterr_o[r,:], alpha=0.2,color=col[r])

plt.xlabel('$t$',fontsize=15)
plt.ylabel('fitness variance',fontsize=15)
#plt.ylim([1e-4,1])
plt.legend()
#plt.yticks([0,0.5e9,1e9,1.5e9,2e9,2.5e9],fontsize=13)
plt.legend(fontsize=13)
plt.gca().tick_params(axis='x',labelsize=13)
plt.gca().tick_params(axis='y',labelsize=13)
plt.tight_layout()
#plt.savefig('5b_fitvar_010824.pdf')
plt.show()

plt.figure()
col = plt.cm.viridis(np.linspace(0.1,0.9,3))
ax = sns.scatterplot(x="dist_init", y="struct",hue='d', data=df_d1)
ax.set_xlabel("$||X||$", fontsize = 14)
ax.set_ylabel("$SSI$", fontsize = 14)
plt.show()

