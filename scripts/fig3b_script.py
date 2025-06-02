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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from Community import Community
from utils import full_point_in_hull
from utils import shannon_diversity
from utils import get_fd_and_centroid
from utils import supply_to_weighted_centroid
from utils import distance
from utils import pred_rad_multiple
from utils import pred_abund_from_abund


N = 100
S = 10
R = 3
#d_list = [1e-7] #[1e-7,5e-7,1e-6,5e-6,1e-5] #1e-7,5e-7,taking two slowest out
d_list = [1e-6]
numt = 1000000#10000000
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
lypN = np.zeros((N,len(d_list)))
lypA = np.zeros((N,len(d_list)))

ranks = np.zeros((N,len(d_list),S))
abund_pred = np.zeros((N,len(d_list)))
ordT = np.zeros((N,len(d_list),numt))
ordTCV = np.zeros((N,len(d_list),numt))
struct_generated = np.zeros((N,len(d_list)))
when = np.zeros((N,len(d_list)))

total_eig_ratio = np.zeros((N,len(d_list)))
a_eig_ratio = np.zeros((N,len(d_list)))
ar_eig_ratio = np.zeros((N,len(d_list)))
tr_eig_ratio = np.zeros((N,len(d_list)))
eigst = np.zeros((N,len(d_list),S+R+S*R),dtype='complex_')

communities = []

for j in range(N):
    
    if j%10==0:
        print('N='+str(j))
    
    
    c = Community(S,R)
    c.changeTimeScale(tend,numt)

    c.setInitialConditions(inou=None)
    n00 = np.random.uniform(1e6,1e6,S)
    #for when adding variation
    #c.setInitialConditionsManual(sameS=False,n0=n00)
    #c.setInitialConditionsSameS(inou=False)
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
            richness[j,i] = np.sum(neq > 1e-3)

            #get functinal diversity at final and initial times
            fd[j] = get_fd_and_centroid(aeq)[0]
            fd_init[j] = get_fd_and_centroid(a0)[0]
            #get normalized shannon diversity (evenness)
            sd[j,i] = shannon_diversity(neq) / shannon_diversity(1/S*np.ones(S))
            #structure generated, variability from traits (SSI)
            struct_generated[j,i] = 1 - (sd[j,i] / initial_sd[j])

            
            #get ranked abundances
            ranks[j,i,:] = c.getRanks()
            #get scores predicting how much traits -> abundances
            score0[j,i],scorec0[j,i],scored0[j,i], _ = pred_rad_multiple(a0,np.log(neq),s,c.E0)
            score[j,i],scorec[j,i],scored[j,i], _ = pred_rad_multiple(aeq,np.log(neq),s,c.E0)
            #null model abundance prediction
        else:
            #j-=1
            print('Failed run on j=',str(j),'and d=',str(d))

df_d1 = pd.DataFrame({'d':dd.flatten(),'rich':richness.flatten(),'hull':in_out.flatten(),'pred':score.flatten(),'pred0':score0.flatten(),'predc0':scorec0.flatten(),'predd0':scored0.flatten(),'predd':scored.flatten(),'predc':scorec.flatten()})
#df_eigtest = df_eigtest[df_eigtest.richness != 0]
#df_d230F.to_csv('N100_S30R3_d1e-7_1e-5F_final.csv')


ineq = np.concatenate((np.zeros(N*len(d_list)),np.ones(N*len(d_list))))
dl = np.concatenate((dd.flatten(),dd.flatten()))
predall = np.concatenate((score0.flatten(),score.flatten()))
predallc = np.concatenate((scorec0.flatten(),scorec.flatten()))
predalld = np.concatenate((scored0.flatten(),scored.flatten()))

df_d3 = pd.DataFrame({'d':dl,'pred':(predall),'predc':(predallc),'predd':(predalld),'init':ineq})
plt.figure()
ax = sns.boxenplot(x="init", y="pred", data=df_d3[df_d3.d == d_list[0]])
ax.set_xlabel(" ", fontsize = 10)
ax.set_ylabel("$R^2$", fontsize = 10)
plt.yscale('linear')
ax.set_xticklabels(['Initial traits','Equilibrium traits'])

