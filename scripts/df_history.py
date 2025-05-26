#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 19:38:34 2023

@author: brendonmcguinness
"""

#run many communities to find history
#RAD vs specialization
#from lag_v_budg_fn import model
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from random import choice
import seaborn as sns
import pandas as pd

from Community import Community
from lag_v_budg_fn import full_point_in_hull
from lag_v_budg_fn import supply_to_centroid
from lag_v_budg_fn import supply_to_weighted_centroid
from lag_v_budg_fn import shannon_diversity
from lag_v_budg_fn import get_fd_and_centroid
from lag_v_budg_fn import pred_rad_from_dist_noscale
from lag_v_budg_fn import pred_rad_from_comp_noscale
from lag_v_budg_fn import pred_rad_multiple
from lag_v_budg_fn import pred_rad_from_traits_noscale
from lag_v_budg_fn import avg_eq_time
from lag_v_budg_fn import get_comp_std


N = 50
S = 10
R = 3
num_samples=100


fd = np.zeros((N,num_samples))
fd_init = np.zeros(N)
fd2 = np.zeros((N,num_samples))
fd2_init = np.zeros(N)
var_n0 = np.zeros(N)

cent = np.zeros((N,R-1))
insur = np.zeros(N)


in_out = np.zeros(N)
dist_com_s = np.zeros((N,num_samples))

c_eqs = np.zeros((N,R))


scores = np.zeros((N,num_samples))
scored = np.zeros((N,num_samples))
scorec = np.zeros((N,num_samples))
scores_old = np.zeros((N,num_samples))
scored_old = np.zeros((N,num_samples))
scorec_old = np.zeros((N,num_samples))
eq_time_c = np.zeros(N)
eq_time_a = np.zeros(N)

leading_eig = np.zeros((N,num_samples))

lyp_samples = 3
lypN = np.zeros(N)
lypA = np.zeros(N)
sd = np.zeros(N)
sd_idx = np.zeros((N,num_samples))
idx_store = np.zeros((N,num_samples))


c_list = []
d_store = np.zeros(N)

for j in range(N):
    
    c = Community(S,R)
    #c.setInitialConditions()
    n00 = np.random.uniform(1e6,1e6,S)
    var_n0[j] = np.var(n00)
    c.setInitialConditionsManual(sameS=False,n0=n00)
    #d_store[j] = np.random.choice(d_list)
    c.setD(5e-6)
    c.runModel()
    neq,ceq,aeq = c.getSteadyState()
    s,a0 = c.getSA()

    neq = (neq / neq.sum()) 
    #eq_time_c[j] = avg_eq_time(c.c,c.t,rel_tol=0.003)
    #eq_time_a[j] = avg_eq_time(c.a,c.t,rel_tol=0.003)
    sd[j] = shannon_diversity(neq) / shannon_diversity(1/S*np.ones(S))
    #lypN[j] = (c.getLyapunovExp(N=lyp_samples)).mean()
    #lypA[j] = (c.getLyapunovExpA0(N=lyp_samples)).mean()

        
        #here we classify what the two treatments are in/out acc/nacc
    if full_point_in_hull(s, a0):
        in_out[j] = 1
    else:
        in_out[j] = 0
    stop = int(50/c.dlta[0])
    idx = np.arange(0,stop,int(stop/num_samples))
    for i,k in enumerate(idx):
        #scores_old[j,i] = pred_rad_from_traits_noscale(c.a[k,:,:],np.log(neq),s,c.E0)[0]
        #scored_old[j,i] = pred_rad_from_dist_noscale(c.a[k,:,:], np.log(neq), s, c.E0)
        #scorec_old[j,i] = pred_rad_from_comp_noscale(c.a[k,:,:],np.log(neq),s,c.E0)        
        scores[j,i],scorec[j,i],scored[j,i] = pred_rad_multiple(c.a[k,:,:],np.log(neq),s,c.E0)  
    #in_out[j] = 1
        #jacob = c.getJacobianAtT(k)
        #leading_eig[j,i] = np.linalg.eig(jacob)[0].real.max()

    #s,a0 = pick_inout_hull(S,R,E0,inout=True)
        dist_com_s[j,i] = supply_to_weighted_centroid(s,c.a[k,:,:],c.n[k,:],c.E0)
        fd[j,i] = get_fd_and_centroid(c.a[k,:,:])[0]
        fd2[j,i] = get_comp_std(c.a[k,:,:],c.E0,c.s)
        sd_idx[j,i] = shannon_diversity(c.n[k,:]) / shannon_diversity(1/S*np.ones(S))
    idx_store[j,:] = idx
    
    
#plt.figure()
#plt.plot()

    
#get average and standard error of ranks
#avg_rd = rank_abund.mean(axis=0)
#sterr_rd = rank_abund.std(axis=0) / (N**0.5)

#df_hist = pd.DataFrame({'pred':scores.flatten(),'cent':dist_com_s.flatten(),'fd':fd.flatten(),'fd2':fd2.flatten(),'shannon':sd.flatten(),'idx':idx_store.flatten()*c.dlta[0]})

plt.style.use('default')

plt.plot(idx*c.dlta[0],scores[0,:])
plt.ylim(0,1)
plt.ylabel("$R^2$")
plt.xlabel("time (generations) $1/\delta$")

sters = scores.std(axis=0) / np.sqrt(N)
sterc = scorec.std(axis=0) / np.sqrt(N)
sterd = scored.std(axis=0) / np.sqrt(N)

sco = [scores.mean(axis=0),scorec.mean(axis=0),scored.mean(axis=0)]
ste = [sters,sterc,sterd]
colors = ['k','k','r']
style = ['solid','dotted','dashed']
lab = ['total','competitor distance','supply distance']
plt.figure()
for r in range(len(sco)):
    plt.plot(idx,sco[r],color=colors[r],linestyle=style[r],label=lab[r])
    plt.fill_between(idx, sco[r] - ste[r], sco[r] + ste[r],color=colors[r], alpha=0.1)
    
plt.xlabel('$t$',fontsize=15)
plt.ylim(0,1)
plt.legend(fontsize=13)
plt.ylabel('variance explained',fontsize=15)




"""
plt.figure()
ax = sns.scatterplot(x="idx", y="pred", data=df_hist)
ax.set_xlabel("time (generations)", fontsize = 10)
ax.set_ylabel("R^2", fontsize = 10)
plt.yscale('linear')
"""