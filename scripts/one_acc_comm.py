#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:37:07 2022

@author: brendonmcguinness
"""

# 1 plastic
import numpy as np
from scipy.integrate import odeint
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from utils import model
from utils import bary2cart
from utils import get_rank_dist_save_ind
from utils import simplex_vertices
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.ticker as mticker



class FormatScalarFormatter(mticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        mticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % mticker._mathdefault(self.format)


class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)
    
def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)
    


def chop_cmap_frac(cmap: LinearSegmentedColormap, frac: float) -> LinearSegmentedColormap:
    """
    Trims a colormap by removing the beginning fraction specified by `frac`.

    Parameters:
    cmap (LinearSegmentedColormap): The input colormap to be trimmed.
    frac (float): The fraction of the colormap to chop off from the start.

    Returns:
    LinearSegmentedColormap: A new colormap with the specified portion removed.
    """    
    cmap_as_array = cmap(np.arange(256))
    cmap_as_array = cmap_as_array[int(frac * len(cmap_as_array)):]
    return LinearSegmentedColormap.from_list(cmap.name + f"_frac{frac}", cmap_as_array)


def plot_simplex_1plast(n_eq,a,s,ind_plast):
    """
    Plots the dynamics of a plastic species in a 2D simplex.

    Parameters:
    n_eq (numpy.ndarray): Equilibrium densities for each species.
    a_eq (numpy.ndarray): Equilibrium traits for each species.
    a (numpy.ndarray): Array representing trait values over time.
    s (numpy.ndarray): Supply vector representing resource availability.
    ind_plast (int): Index of the plastic species whose dynamics will be plotted.

    This function visualizes the movement of a plastic species in trait space
    by projecting it into a simplex. It also displays the trajectories of all species
    and their equilibrium configurations.
    """
    a_eq = a[-1,:,:]
    a_sc = a / (Q[:,None]*dlta[:,None])
    cmap1 = plt.get_cmap('Blues')
    cmap1 = chop_cmap_frac(cmap1, 0.4)
    #ind_del = 4
    #ind_samp = 0
    #get ride of trait point we deleted
    #rank_del = int(df.iloc[ind_samp*10+ind_del]['Rank'])
    aplast_del = np.delete(a_eq,ind_plast,axis=0)
    ainit = np.concatenate((aplast_del, a[0,ind_plast,:,None].T), axis=0)
    #to get trait points organized by rank
    rs,inds = get_rank_dist_save_ind(n_eq)
    ac,corners = bary2cart(aplast_del,corners=simplex_vertices(R-1))
    ac_full = bary2cart(a_eq,corners=simplex_vertices(R-1))[0]

    hull = ConvexHull(ac)
    hull_full = ConvexHull(ac_full)
    hull_total = ConvexHull(corners)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.plot(ac[:,0], ac[:,1], 'o')
    cmaps = np.arange(0,S-1)
    cmapss = matplotlib.colors.ListedColormap(plt.cm.tab10.colors[1:])
    acinit = bary2cart(ainit,corners=simplex_vertices(R-1))[0]
    hull = ConvexHull(acinit)
    for simplex in hull.simplices:
        plt.plot(acinit[simplex, 0], acinit[simplex, 1], 'k',linestyle='dotted',alpha=1,zorder=1)

    for simplex in hull_full.simplices:
        plt.plot(ac_full[simplex, 0], ac_full[simplex, 1], 'k-',alpha=1,zorder=2) #linestyle='dashed'


    sc = bary2cart(s,corners=simplex_vertices(R-1))[0]
    for simplex in hull_total.simplices:
        plt.plot(corners[simplex, 0], corners[simplex, 1], 'k-',zorder=3)
        

    num_points = 200
    num_t = a.shape[0]
    t_plot = np.linspace(0,num_t,num_points,dtype=int)
    a2df = np.zeros((num_points,S,R))
    act = np.zeros((num_points,S,2))
    
    for i,ts in enumerate(t_plot-1):
        if i==0:
            ts+=1
        a2df[i,ind_plast,:] = a_sc[ts,ind_plast,:]
        act[i,ind_plast,:] = bary2cart(a2df[i,ind_plast,:],corners=simplex_vertices(R-1))[0]
    
    plt.scatter(act[:,ind_plast,0], act[:,ind_plast,1],s=50,marker='x',c=np.linspace(0.6,1,num_points),cmap='Blues',alpha=1.0)#,c=t,cmap='viridis')
    s = ax.scatter(act[:,ind_plast,0], act[:,ind_plast,1],s=50,marker='x',label='plastic species',c=np.linspace(0,1,num_points),cmap=cmap1,alpha=1.0,zorder=5)#,c=t,cmap='viridis')
    cbar = fig.colorbar(mappable=s,format=mticker.FuncFormatter(fmt),label="time", orientation="vertical")

    cbar.set_ticks([])
    cbar.set_label('$t$', fontsize=20)

    plt.scatter(sc[0],sc[1],s=250,marker='*',color='k')
    plt.scatter(ac[:,0], ac[:,1],c=cmaps, cmap=cmapss,zorder=4,s=50)

    plt.legend()
    

    legend_elements = [Line2D([0], [0], marker ='X',color='w', label='Plastic',markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Static',
                              markerfacecolor='g', markersize=10),
                        Line2D([0], [0], marker='*', color='w', label='Supply',
                              markerfacecolor='k', markersize=15)
                       ]
    second_elements = [Line2D([0], [0],color='k', lw=2, label='Initial',linestyle='dotted',alpha=1.0)
                       ,Line2D([0], [0],color='k', lw=2, label='Equilibrium',linestyle='solid',alpha=1.0)
                       ]
    first_legend = plt.legend(handles=legend_elements, loc='upper right',fontsize=13)
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=second_elements, loc='upper left',fontsize=13)

    
    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig('attempt_042825_fig2_Eq.pdf')


S = 10
R = 3


cmaps = np.arange(S)
    
v = np.random.uniform(10e7, 10e7, R)
dlta = np.random.uniform(5e-3, 5e-3, S) 
Q = np.ones(S)*10e-5 #random.uniform(10e-5,10e-5)
#Q[0] = 1e-5
E0 = np.random.uniform(Q*dlta, Q*dlta, S)
K = np.random.uniform(10e-6, 10e-6, R)  # 10e-4 * np.ones(R)
u = np.zeros(R)
#s = np.array([7e-2,10e-2,1e-2,4e-3,8e-3,1e-2,6e-2,7e-3]) #np.random.uniform(10e-4, 10e-2, R)  # 100*np.zeros(R)
s = np.random.uniform(10e-6,10e-2,R) #back to 10e-5
#try with and without adaptive strategies
d = 60e-7*np.zeros(S) #10e-6*np.ones(S) * 0
d[0] = 6e-6


# time points
num_t = 200000
t_end = 200000
t = np.linspace(0, t_end, num_t)
#here we have s being drawn randomly too
s = np.random.uniform(10e-6,10e-2,R)

s = np.array([8e-4,3e-3,6e-4]) #comment this line out to get a random sample, but these were the values we used in the figure

#comment these lines out to get a random sample, but these were the values we used in the figure
a0= np.array([[2.22095360e-07, 1.82775111e-07, 9.51295296e-08],
       [8.46448532e-08, 2.57901156e-07, 1.57453990e-07],
       [2.96965917e-07, 1.36785207e-07, 6.62488758e-08],
       [1.93274242e-07, 1.50385394e-07, 1.56340364e-07],
       [8.98773988e-08, 1.82221380e-07, 2.27901221e-07],
       [2.19762992e-07, 6.61728841e-08, 2.14064124e-07],
       [4.01567662e-08, 2.87870313e-07, 1.71972921e-07],
       [1.86885291e-07, 9.92394490e-08, 2.13875260e-07],
       [9.59571533e-08, 3.24498257e-07, 7.95445901e-08],
       [5.01584341e-08, 3.04117415e-07, 1.45724151e-07]])

#comment this back in to get random sample, and choose what parameters you want in your dirichlet distribution
"""
a0 = np.zeros((S, R), dtype=float)
for i in range(0, S):
    a0[i, :] = np.random.dirichlet(3*np.ones(R), size=1) * E0[i]
"""
n0 = np.random.uniform(10e6, 10e6, S)
c0 = np.random.uniform(10e-3, 10e-3, R)
z0 = np.concatenate((n0, c0, a0.flatten(), E0), axis=None)

z = odeint(model,z0,t,args=(S,R,v,d,dlta,s,u,K,Q))

n = z[:,0:S]
c = z[:,S:S+R]
a = z[:,S+R:S+R+S*R]
a = np.reshape(a,(num_t,S,R))
#E = z[:,S+R+S*R:2*S+R+S*R]

n_eq = n[-1,:]
#c_eq = c[-1,:]
#a_eq = a[-1,:,:]
#a_sc = a / (Q[:,None]*dlta[:,None])

plot_simplex_1plast(n_eq,a,s,0)


#n_nacc = z[:,0:S]

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(S):
    plt.semilogy(t,n[:,i]/n[-1,:].sum()) #color=colours[i])
plt.set_cmap('tab10')
plt.ylabel('$\hat{n}_\sigma$',fontsize=22)
plt.xlabel('$t$',fontsize=22)
plt.ylim(0.7e-8,3)
plt.yticks([1e-8,1e-6,1e-4,1e-2,1],fontsize=14)
plt.setp(ax.spines.values(), linewidth=1.5)

plt.xticks([0,50000,100000,150000,200000],fontsize=14)
plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.tight_layout()

d[0] = 0
z = odeint(model,z0,t,args=(S,R,v,d,dlta,s,u,K,Q))

n_nacc = z[:,0:S]

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(S):
    plt.semilogy(t,n_nacc[:,i]/n_nacc[-1,:].sum()) #color=colours[i])
plt.set_cmap('tab10')
plt.ylabel('$\hat{n}_\sigma$',fontsize=22)
plt.xlabel('$t$',fontsize=22)
plt.ylim(0.7e-8,3)
plt.yticks([1e-8,1e-6,1e-4,1e-2,1],fontsize=14)
plt.setp(ax.spines.values(), linewidth=1.5)

plt.xticks([0,50000,100000,150000,200000],fontsize=14)
plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.tight_layout()
