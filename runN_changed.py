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


N = 50
S = 10
R = 3
#d_list = [1e-7] #[1e-7,5e-7,1e-6,5e-6,1e-5] #1e-7,5e-7,taking two slowest out
d_list = [1e-7,1e-6,1e-5]
numt = 100000#10000000
tend = 100000 #100000
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
            #get functinal diversity at final and initial times
            fd[j] = get_fd_and_centroid(aeq)[0]
            fd_init[j] = get_fd_and_centroid(a0)[0]
            #get normalized shannon diversity (evenness)
            sd[j,i] = shannon_diversity(neq) / shannon_diversity(1/S*np.ones(S))
            #structure generated, variability from traits (SSI)
            struct_generated[j,i] = 1 - (sd[j,i] / initial_sd[j])
            #get transient times for traits and abundances (resource flattening describes abundances)
            eq_time_c[j,i] = avg_eq_time(c.c,c.t,rel_tol=0.003)
            eq_time_a[j,i] = avg_eq_time(c.a,c.t,rel_tol=0.003) 
            #get jacobian matrix
            jacob = c.getJacobian()
            jacob_ar = jacob
            #different subsets of jacobian matrix 
            eigst[j,i,:] = np.linalg.eig(jacob[:S+R+S*R,:S+R+S*R])[0]
            eigsa = np.linalg.eig(jacob[:S,:S])[0]
            eigsar = np.linalg.eig(jacob[:S+R,:S+R])[0]
            eigstr = np.linalg.eig(jacob[S+R:S+R+S*R,S+R:S+R+S*R])[0]
            
            #leading eigenvalue
            leading_eigenvalue[j,i] = eigst[j,i,:].real.max()            
            le_a[j,i] = eigsa.real.max()
            le_ar[j,i] = eigsar.real.max()
            le_t[j,i] = eigstr.real.max()
            
            #ratio of positive to negative eigenvalues
            total_eig_ratio[j,i] = len(eigst[j,i,eigst[j,i,:]<=0]) / (S+R+S*R)
            a_eig_ratio[j,i] = len(eigsa[eigsa<=0]) / len(eigsa)
            ar_eig_ratio[j,i] = len(eigsar[eigsar<=0]) / len(eigsar)
            tr_eig_ratio[j,i] = len(eigstr[eigstr<=0]) / len(eigstr)
                       
  
            #comment back later
            # for one plastic species use just first index (distance)
            #init_shift_trait[j,i] = np.linalg.norm(bary2cart((aeq[0,:]/c.E0[0, None]),corners=simplex_vertices(R-1))[0]-bary2cart((a0[0,:]/c.E0[0, None]),corners=simplex_vertices(R-1))[0])
            #total distance of trait shuft
            init_shift_trait[j,i] = np.linalg.norm(bary2cart((aeq/c.E0[:, None]),corners=simplex_vertices(R-1))[0]-bary2cart((a0/c.E0[:, None]),corners=simplex_vertices(R-1))[0])
            #score0_og[j,i] = pred_rad_from_traits_noscale(a0,np.log(neq),s,c.E0)[0]
            #score_og[j,i] = pred_rad_from_traits_noscale(aeq,np.log(neq),s,c.E0)[0]
            #a_fitted[j,i], b_fitted[j,i] = pred_rad_from_traits_noscale(a0,np.log(neq),s,c.E0)[4]
            #scored[j,i] = pred_rad_from_dist_noscale(a0, np.log(neq), s, c.E0)
            #scorec[j,i] = pred_rad_from_comp_noscale(a0,np.log(neq),s,c.E0)
            #define a threshold for presence of a species if its at least 0.1% of the abundance of the community
            #richness[j,i] = (neq>1e-3).sum()
            
            #get ranked abundances
            ranks[j,i,:] = c.getRanks()
            #get scores predicting how much traits -> abundances
            score0[j,i],scorec0[j,i],scored0[j,i] = pred_rad_multiple(a0,np.log(neq),s,c.E0)
            score[j,i],scorec[j,i],scored[j,i] = pred_rad_multiple(aeq,np.log(neq),s,c.E0)
            #null model abundance prediction
            abund_pred[j,i] = pred_abund_from_abund(c.n0, neq)
            #order parameter for fitness variance
            ordT[j,i,:] = orderParameter(c.n)
            ordTCV[j,i,:] = orderParameterCV(c.n)
            #peak[j,i] = ordT[j,i,:].max()
            #when[j,i] = c.whenInHull()
            
            
            #lypN[j,i] = (c.getLyapunovExp(N=2)).max()
            #lypA[j,i] = (c.getLyapunovExpA0(N=2)).max()
        else:
            #j-=1
            print('Failed run on j=',str(j),'and d=',str(d))

df_d1 = pd.DataFrame({'d':dd.flatten(),'peak':peak.flatten(),'eigratioab':a_eig_ratio.flatten(),'eigratioar':ar_eig_ratio.flatten(),'eigratio':total_eig_ratio.flatten(),'eigratiotraits':tr_eig_ratio.flatten(),'eig':leading_eigenvalue.flatten(),'eiga':le_a.flatten(),'eigar':le_ar.flatten(),'eigt':le_t.flatten(),'struct':struct_generated.flatten(),'abundP':abund_pred.flatten(),'lypN':lypN.flatten(),'lypA':lypA.flatten(),'tovera':eq_time_a.flatten() / eq_time_c.flatten(),'fd':fd.flatten(),'hull':in_out.flatten(),'fdinit':fd_init.flatten(),'distplast':dist_plast.flatten(),'richness':richness.flatten(),'dist_init':dist_com_s_init.flatten(),'dist_final':dist_com_s_final.flatten(),'plast':init_shift_trait.flatten()*S,'pred':score.flatten(),'pred0':score0.flatten(),'predc0':scorec0.flatten(),'predd0':scored0.flatten(),'predd':scored.flatten(),'predc':scorec.flatten(),'pred_diff':(score0.flatten()-score.flatten()), 'c_eq':eq_time_c.flatten()*c.dlta[0],'a_eq':eq_time_a.flatten()*c.dlta[0],'shannon':sd.flatten(),'a':a_fitted.flatten(),'b':b_fitted.flatten(),'ba':b_fitted.flatten()/a_fitted.flatten(),'tva':score0.flatten()-abund_pred.flatten()})
#df_eigtest = df_eigtest[df_eigtest.richness != 0]
#df_d230F.to_csv('N100_S30R3_d1e-7_1e-5F_final.csv')


plt.figure()
ax = sns.lineplot(x="d", y="a_eq",label='traits',color='tab:cyan', data=df_d1)
ax = sns.lineplot(x="d", y="c_eq",label='abundances',color='tab:red', data=df_d1)
ax.set_xlabel("$d$", fontsize = 16)
ax.set_ylabel("transient time $(1/\delta)$", fontsize = 14)
ax.loglog()
plt.legend(fontsize=14)
plt.show()

ineq = np.concatenate((np.zeros(N*len(d_list)),np.ones(N*len(d_list))))
dl = np.concatenate((dd.flatten(),dd.flatten()))
predall = np.concatenate((score0.flatten(),score.flatten()))
predallc = np.concatenate((scorec0.flatten(),scorec.flatten()))
predalld = np.concatenate((scored0.flatten(),scored.flatten()))

df_d3 = pd.DataFrame({'d':dl,'pred':(predall),'predc':(predallc),'predd':(predalld),'init':ineq})
plt.figure()
ax = sns.boxenplot(x="init", y="pred", data=df_d3[df_d3.d == 1e-6])
ax.set_xlabel(" ", fontsize = 10)
ax.set_ylabel("$R^2$", fontsize = 10)
plt.yscale('linear')
ax.set_xticklabels(['Initial traits','Equilibrium traits'])

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
ax = sns.scatterplot(x='dist_init',y='shannon',hue='d',palette='crest',data=df_d1)
ax.set_xlabel("$iCSVD$", fontsize = 12)
ax.set_ylabel("evenness", fontsize = 12)

sterr = ranks.std(axis=0) / np.sqrt(N)

plt.figure()
for r in range(len(d_list)):
    plt.semilogy(np.arange(1,S+1),ranks.mean(axis=0)[r,:],label='d='+str(d_list[r]),color=col[r])
    plt.fill_between(np.arange(1,S+1), ranks.mean(axis=0)[r,:] - sterr[r,:], ranks.mean(axis=0)[r,:] + sterr[r,:], alpha=0.2,color=col[r])
plt.xticks(np.arange(5,S+1,5))
plt.xlabel('ranks')
plt.ylabel('abundance')
plt.ylim([1e-4,1])
plt.legend()
plt.show()
