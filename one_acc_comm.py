#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:37:07 2022

@author: brendonmcguinness
"""

# 1 keystone plastic
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
#import plotly.io as pio
#import plotly.express as px
from lag_v_budg_fn import model
from lag_v_budg_fn import pick_inout_hull
from lag_v_budg_fn import bary2cart
from lag_v_budg_fn import get_hats
from lag_v_budg_fn import get_rank_dist_save_ind
from lag_v_budg_fn import avg_eq_time_traits
from lag_v_budg_fn import simplex_vertices
import matplotlib.cm as cm
import imageio
import os
import time
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.ticker as mticker


ti = time.time()
#matplotlib.rcParams.update({'font.size':20})

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
    
def var_unif(a,b):
    return (b-a)**2 / 12

def chop_cmap_frac(cmap: LinearSegmentedColormap, frac: float) -> LinearSegmentedColormap:
    """Chops off the beginning `frac` fraction of a colormap."""
    cmap_as_array = cmap(np.arange(256))
    cmap_as_array = cmap_as_array[int(frac * len(cmap_as_array)):]
    return LinearSegmentedColormap.from_list(cmap.name + f"_frac{frac}", cmap_as_array)


def plot_simplex_1plast(n_eq,a_eq,a,s,ind_plast):
        #make simplex
    #plasma = cm.get_cmap('plasma', 10)
    #want to change this instead of time ratio shows actual time in colorbar
    t_end = 200000
    dlta = np.random.uniform(5e-3, 5e-3, S) 

    cmap1 = plt.get_cmap('Blues')
    cmap1 = chop_cmap_frac(cmap1, 0.4)
    #ind_del = 4
    #ind_samp = 0
    #get ride of trait point we deleted
    #rank_del = int(df.iloc[ind_samp*10+ind_del]['Rank'])
    aplast_del = np.delete(a_eq,ind_plast,axis=0)
    ainit = np.concatenate((aplast_del, a[0,ind_plast,:,None].T), axis=0)
    aplast_full = a[:,ind_plast,:]
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
        plt.plot(acinit[simplex, 0], acinit[simplex, 1], 'tab:blue',alpha=1,zorder=1)
    #plt.fill(ac[hull.vertices,0], ac[hull.vertices,1], 'tab:blue', alpha=0.2)    
    #plt.fill_between(ac[hull.vertices,0], ac[hull.vertices,1], 'tab:blue', alpha=0.2,hatch=r"//")
    for simplex in hull_full.simplices:
        plt.plot(ac_full[simplex, 0], ac_full[simplex, 1], 'tab:orange',alpha=1,zorder=2) #linestyle='dashed'
    #plt.fill(ac_full[hull_full.vertices,0], ac_full[hull_full.vertices,1], 'tab:orange', alpha=0.2)    
    #plt.fill_between(ac[hull.vertices,0], ac[hull.vertices,1], 'tab:orange', alpha=0.2,hatch=r"//")

    #plt.colorbar(label="Rank", orientation="vertical")
    


    #plt.scatter(acinit[:,0], acinit[:,1], c=cmaps, cmap='tab10')

    sc = bary2cart(s,corners=simplex_vertices(R-1))[0]
    for simplex in hull_total.simplices:
        plt.plot(corners[simplex, 0], corners[simplex, 1], 'k-',zorder=3)
        
    #plt.scatter(ac[:,0], ac[:,1],c=cmaps, cmap=cmapss)

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
    #plt.scatter(act[0,ind_plast,0], act[0,ind_plast,1],marker='o')
    v1 = np.linspace(0, t_end, 10, endpoint=True)
    #fmt = FormatScalarFormatter("%.2f")
    cbar = fig.colorbar(mappable=s,format=mticker.FuncFormatter(fmt),label="time", orientation="vertical")
    #cbar.set_ticks(np.arange(0,t_end*dlta[0]+1, 300))
    #cbar.set_ticks([s.colorbar.vmin + t*(s.colorbar.vmax-s.colorbar.vmin) for t in cbar.ax.get_yticks()])
    #cbar.set_ticklabels((np.arange(0,t_end+1, 40000,dtype=int)))
    cbar.set_ticks([])
    cbar.set_label('$t$', fontsize=20)

    #cbar.set_ticks(v1)
    plt.scatter(sc[0],sc[1],s=250,marker='*',color='k')
    plt.scatter(ac[:,0], ac[:,1],c=cmaps, cmap=cmapss,zorder=4,s=50)

    #cbar.set_ticklabels(["{:4.2f}".format(i) for i in v1]) # add the labels
    plt.legend()
    

    legend_elements = [Line2D([0], [0], marker ='X',color='w', label='plastic trait',markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='static traits',
                              markerfacecolor='g', markersize=10),
                        Line2D([0], [0], marker='*', color='w', label='supply vector',
                              markerfacecolor='k', markersize=15)
                       ]
    second_elements = [Line2D([0], [0],color='tab:blue', lw=2, label='initial',linestyle='solid',alpha=1.0)
                       ,Line2D([0], [0],color='tab:orange', lw=2, label='equilibrium',linestyle='solid',alpha=1.0)
                       ]
    first_legend = plt.legend(handles=legend_elements, loc='upper right',fontsize=13)
    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)

# Create another legend for the second line.
    #plt.legend(handles=[line2], loc='lower right')
    plt.legend(handles=second_elements, loc='upper left',fontsize=13)

    #plt.legend(['First List', 'Second List'], loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig('one_plast_simplex_blueorange.pdf')


    
"""
def plot_simplex_1plast_gif(n_eq, a_eq, a, s, ind_plast, filename='simplex_1plast.gif'):
    # Make simplex
    cmap1 = plt.get_cmap('Blues')
    cmap1 = chop_cmap_frac(cmap1, 0.4)
    aplast_del = np.delete(a_eq, ind_plast, axis=0)
    ainit = np.concatenate((aplast_del, a[0, ind_plast, :, None].T), axis=0)
    aplast_full = a[:, ind_plast, :]
    rs, inds = get_rank_dist_save_ind(n_eq)
    ac, corners = bary2cart(a_eq)

    # Set up figure and plot initial points
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, np.sqrt(3) / 2])
    cmaps = np.arange(S)
    ax.scatter(ac[:, 0], ac[:, 1], c=cmaps, cmap='tab10')
    hull = ConvexHull(ac)
    for simplex in hull.simplices:
        ax.plot(ac[simplex, 0], ac[simplex, 1], 'k--', alpha=0.55)
    acinit = bary2cart(ainit)[0]
    hull = ConvexHull(acinit)
    for simplex in hull.simplices:
        ax.plot(acinit[simplex, 0], acinit[simplex, 1], 'k:', alpha=0.3)
    sc = bary2cart(s)[0]
    ax.scatter(sc[0], sc[1], s=250, marker='*', color='k')
    for simplex in hull_total.simplices:
        ax.plot(corners[simplex, 0], corners[simplex, 1], 'k-')

    # Initialize empty list to store images for gif
    images = []
    
    # Loop over time steps and plot each point on the simplex
    num_points = 10000
    num_t = a.shape[0]
    t_plot = np.linspace(0, num_t, num_points, dtype=int)
    a2df = np.zeros((num_points, S, R))
    act = np.zeros((num_points, S, 2))
    for i, ts in enumerate(t_plot - 1):
        if i == 0:
            ts += 1
        a2df[i, ind_plast, :] = a_sc[ts, ind_plast, :]
        act[i, ind_plast, :] = bary2cart(a2df[i, ind_plast, :])[0]
        
        # Clear the axis, redraw the points and hull, and plot the current point
        ax.clear()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, np.sqrt(3) / 2])
        ax.scatter(ac[:, 0], ac[:, 1], c=cmaps, cmap='tab10')
        for simplex in hull.simplices:
            ax.plot(ac[simplex, 0], ac[simplex, 1], 'k--', alpha=0.55)
        for simplex in hull_total.simplices:
            ax.plot(corners[simplex, 0], corners[simplex, 1], 'k-')
        ax.scatter(sc[0], sc[1], s=250, marker='.')
"""                

"""    
def plot_gif_1plast(n_eq,a_eq,a,s,ind_plast):  
        
    cmap1 = plt.get_cmap('Blues')
    cmap1 = chop_cmap_frac(cmap1, 0.4)
    #ind_del = 4
    #ind_samp = 0
    #get ride of trait point we deleted
    #rank_del = int(df.iloc[ind_samp*10+ind_del]['Rank'])
    aplast_del = np.delete(a_eq,ind_plast,axis=0)
    ainit = np.concatenate((aplast_del, a[0,ind_plast,:,None].T), axis=0)
    aplast_full = a[:,ind_plast,:]
    #to get trait points organized by rank
    rs,inds = get_rank_dist_save_ind(n_eq)
    ac,corners = bary2cart(a_eq)
    
    hull = ConvexHull(ac)
    hull_total = ConvexHull(corners)
    plt.figure()
    #plt.plot(ac[:,0], ac[:,1], 'o')
    cmaps = np.arange(S)

    num_frames = 20
    num_points = num_frames
    num_t = a.shape[0]
    #t_plot = np.linspace(0,num_t,num_points,dtype=int)
    a2df = np.zeros((num_points,S,R))
    act = np.zeros((num_points,S,2))
    t_plot = np.logspace(1,5.5,num_frames,dtype=int)
    t_plot = np.append(t_plot,num_t-2)
    filenames = []
    print(a_sc.shape)
    for i,ts in enumerate(t_plot-1):
        if i==0:
            ts+=1
        a2df[i,ind_plast,:] = a_sc[ts+1,ind_plast,:]
        act[i,ind_plast,:] = bary2cart(a2df[i,ind_plast,:])[0]
        plt.scatter(ac[:,0], ac[:,1], c=cmaps, cmap='tab10')
    for simplex in hull.simplices:
        plt.plot(ac[simplex, 0], ac[simplex, 1], 'k--',alpha=0.55) #linestyle='dashed'
    #plt.colorbar(label="Rank", orientation="vertical")
    
    acinit = bary2cart(ainit)[0]
    hull = ConvexHull(acinit)
    for simplex in hull.simplices:
        plt.plot(acinit[simplex, 0], acinit[simplex, 1], 'k:',alpha=0.3)
    #plt.scatter(acinit[:,0], acinit[:,1], c=cmaps, cmap='tab10')

    sc = bary2cart(s)[0]
    plt.scatter(sc[0],sc[1],s=250,marker='*',color='k')
    for simplex in hull_total.simplices:
        plt.plot(corners[simplex, 0], corners[simplex, 1], 'k-')
        
    plt.scatter(act[:,ind_plast,0], act[:,ind_plast,1],s=100,marker='x',c=np.linspace(0.7,1,num_points),cmap='Blues',alpha=1.0)#,c=t,cmap='viridis')
    plt.scatter(act[:,ind_plast,0], act[:,ind_plast,1],s=100,marker='x',label='plastic species',c=np.linspace(0,1,num_points),cmap=cmap1,alpha=1.0)#,c=t,cmap='viridis')
    #plt.scatter(act[0,ind_plast,0], act[0,ind_plast,1],marker='o')

    #plt.colorbar(label="Time Ratio", orientation="vertical",ticks=None)
    plt.legend()
    

    legend_elements = [Line2D([0], [0], marker ='X',color='w', label='Plastic',markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Static',
                              markerfacecolor='g', markersize=10),
                        Line2D([0], [0], marker='*', color='w', label='Supply',
                              markerfacecolor='k', markersize=15)
                       ]
    second_elements = [Line2D([0], [0],color='k', lw=2, label='Initial',linestyle='dotted',alpha=0.3)
                       ,Line2D([0], [0],color='k', lw=2, label='Final',linestyle='dashed',alpha=0.55)
                       ]
    first_legend = plt.legend(handles=legend_elements, loc='upper right')
    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)           # create file name and append it to a list
    filename = f'{i}.png'
    filenames.append(filename)
        
        # save frame
    plt.savefig(filename)
    plt.close()# build gif
    with imageio.get_writer('one_acc.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
"""


def plot_simplex_1plast_gif(n_eq, a_eq, a, s, ind_plast, filename):
    # Define colormap
    cmap1 = plt.get_cmap('Blues')
    cmap1 = chop_cmap_frac(cmap1, 0.4)
    
    # Delete trait point of plastic species
    aplast_del = np.delete(a_eq, ind_plast, axis=0)
    ainit = np.concatenate((aplast_del, a[0, ind_plast, :, None].T), axis=0)
    aplast_full = a[:, ind_plast, :]
    
    # Get trait points organized by rank
    rs, inds = get_rank_dist_save_ind(n_eq)
    ac, corners = bary2cart(a_eq)
    
    # Create initial plot
    hull = ConvexHull(ac)
    hull_total = ConvexHull(corners)
    fig, ax = plt.subplots()
    scat = ax.scatter(ac[:,0], ac[:,1], c=np.arange(S), cmap='tab10')
    for simplex in hull.simplices:
        ax.plot(ac[simplex, 0], ac[simplex, 1], 'k--',alpha=0.55)
    axinit = bary2cart(ainit)[0]
    hull = ConvexHull(axinit)
    for simplex in hull.simplices:
        ax.plot(axinit[simplex, 0], axinit[simplex, 1], 'k:',alpha=0.3)
    sc = bary2cart(s)[0]
    ax.scatter(sc[0],sc[1],s=250,marker='*',color='k')
    for simplex in hull_total.simplices:
        ax.plot(corners[simplex, 0], corners[simplex, 1], 'k-')
    ax.set_title('Plastic Species on Simplex')
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    plt.colorbar(scat, ax=ax, label="Rank", orientation="vertical")
    plt.tight_layout()
    
    # Define function to update plot for each frame of gif
    def update(i):
        ts = i
        if i == 0:
            ts += 1
        a2df = np.zeros((S, R))
        a2df[ind_plast, :] = a[ts, ind_plast, :]
        act = bary2cart(a2df)[0]
        scat.set_offsets(act)
        scat.set_array(np.arange(S))
        return scat,
    
    # Create animation object and save as gif
    anim = FuncAnimation(fig, update, frames=a.shape[0], interval=100)
    anim.save(filename, writer='imagemagick', fps=30)
    
    # Show plot after creating gif
    plt.show()

def plot_simplex_noplast(n_eq,a0,a,s):
        #make simplex
    #plasma = cm.get_cmap('plasma', 10)
    
    cmap1 = plt.get_cmap('Blues')
    cmap1 = chop_cmap_frac(cmap1, 0.4)

    #to get trait points organized by rank
    rs,inds = get_rank_dist_save_ind(n_eq)
    ac,corners = bary2cart(a0,corners=simplex_vertices(R-1))
    
    hull = ConvexHull(ac)
    hull_total = ConvexHull(corners)
    plt.figure()
    #plt.plot(ac[:,0], ac[:,1], 'o')
    cmaps = np.arange(S)

    plt.scatter(ac[:,0], ac[:,1], c=cmaps, cmap='tab10')
    for simplex in hull.simplices:
        plt.plot(ac[simplex, 0], ac[simplex, 1], 'k:',alpha=0.3) #linestyle='dashed'

    sc = bary2cart(s,corners=simplex_vertices(R-1))[0]
    plt.scatter(sc[0],sc[1],s=250,marker='*',color='k')
    for simplex in hull_total.simplices:
        plt.plot(corners[simplex, 0], corners[simplex, 1], 'k-')



    legend_elements = [#Line2D([0], [0], marker ='X',color='w', label='Plastic',markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Static',
                              markerfacecolor='g', markersize=10),
                        Line2D([0], [0], marker='*', color='w', label='Supply',
                              markerfacecolor='k', markersize=15)
                       ]
    second_elements = [Line2D([0], [0],color='k', lw=2, label='Initial',linestyle='dotted',alpha=0.3)
                       #,Line2D([0], [0],color='k', lw=2, label='Final',linestyle='dashed',alpha=0.55)
                       ]
    first_legend = plt.legend(handles=legend_elements, loc='upper right')
    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)

# Create another legend for the second line.
    #plt.legend(handles=[line2], loc='lower right')
    plt.legend(handles=second_elements, loc='upper left')
    plt.axis('off')

    #plt.legend(['First List', 'Second List'], loc='upper right')

S = 10
R = 3

plasma = cm.get_cmap('plasma', 10)
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
s = np.array([8e-4,3e-3,6e-4])
#s = np.array([3e-3,6e-4,6e-4])

#as of now the initial stragies are random in each community/remember to change back

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

"""
a0 = np.array([[1.06933009e-07, 1.27701507e-07, 2.65365484e-07],
       [1.54555133e-07, 2.41056542e-07, 1.04388325e-07],
       [1.84071966e-07, 1.22725651e-07, 1.93202383e-07],
       [2.18869635e-07, 2.70062609e-08, 2.54124104e-07],
       [1.66014392e-07, 1.47592711e-07, 1.86392897e-07],
       [2.09566946e-07, 9.85157398e-08, 1.91917314e-07],
       [2.33534554e-07, 1.43658036e-07, 1.22807410e-07],
       [2.03966290e-07, 1.00415767e-07, 1.95617943e-07],
       [1.80138234e-07, 1.15997871e-07, 2.03863895e-07],
       [2.25104768e-07, 1.72923009e-07, 1.01972223e-07],
       [1.14496966e-07, 1.82854036e-07, 2.02648998e-07],
       [2.00852671e-07, 2.42861555e-07, 5.62857743e-08],
       [2.26503132e-07, 1.44867867e-07, 1.28629000e-07],
       [1.49595244e-07, 1.39709699e-07, 2.10695057e-07],
       [1.70482413e-07, 2.61012127e-07, 6.85054604e-08],
       [1.40327843e-07, 8.61416276e-08, 2.73530530e-07],
       [2.16726646e-07, 1.19962480e-07, 1.63310873e-07],
       [2.05769982e-07, 1.88824649e-07, 1.05405369e-07],
       [9.75652083e-08, 1.01397583e-07, 3.01037209e-07],
       [1.96516086e-07, 1.60032925e-07, 1.43450990e-07],
       [1.00669098e-07, 3.04168257e-07, 9.51626443e-08],
       [1.37285100e-07, 2.96691925e-07, 6.60229745e-08],
       [1.16542862e-07, 1.97198980e-07, 1.86258158e-07],
       [2.49950105e-07, 6.17679502e-08, 1.88281945e-07],
       [5.08354964e-08, 3.79733252e-07, 6.94312515e-08],
       [2.82468313e-07, 1.23006336e-07, 9.45253508e-08],
       [9.64628082e-08, 1.55086814e-07, 2.48450378e-07],
       [2.04650893e-07, 2.01085516e-07, 9.42635908e-08],
       [1.94576658e-07, 4.68124193e-08, 2.58610922e-07],
       [1.41796614e-07, 1.98761104e-08, 3.38327275e-07]])
"""
"""
a0 = np.zeros((S, R), dtype=float)
for i in range(0, S):
    a0[i, :] = np.random.dirichlet(3*np.ones(R), size=1) * E0[i]
"""
#s,a0 = pick_inout_hull(S, R, E0, a=10e-6, b=10e-2, inout=False)
n0 = np.random.uniform(10e6, 10e6, S)
c0 = np.random.uniform(10e-3, 10e-3, R)
#n0 = np.random.normal(10e6,var_n0,S)
z0 = np.concatenate((n0, c0, a0.flatten(), E0), axis=None)

z = odeint(model,z0,t,args=(S,R,v,d,dlta,s,u,K,Q))

n = z[:,0:S]
c = z[:,S:S+R]
a = z[:,S+R:S+R+S*R]
a = np.reshape(a,(num_t,S,R))
#E = z[:,S+R+S*R:2*S+R+S*R]

n_eq = n[-1,:]
#c_eq = c[-1,:]
a_eq = a[-1,:,:]
a_sc = a / (Q[:,None]*dlta[:,None])
dt = t[1]-t[0]

"""
plt.figure()
for i in range(S):
    plt.semilogy(t,n[:,i]) #color=colours[i])
plt.set_cmap('tab10')
plt.ylabel('density (cells/mL)')
plt.xlabel('time')
plt.ylim(1,10e8)
#plt.title('starting alpha='+str(a0[0]/E0))
"""

s0_hat,a0_hat,s_hat,a_hat_eq = get_hats(n_eq,a_eq,a0,dlta,v,s,E0,Q)
sc,corners = bary2cart(s,corners=simplex_vertices(R-1))
hull = ConvexHull(corners)
plt.figure()
#color=cm.hsv(i/30.0)
num_points = 100
t_plot = np.linspace(0,num_t,num_points,dtype=int)
a2df = np.zeros((num_points,S,R))
act = np.zeros((num_points,S,2))

for j in range(S):
    for i,ts in enumerate(t_plot-1):
        if i==0:
            ts+=1
        a2df[i,j,:] = a_sc[ts,j,:]
        act[i,j,:] = bary2cart(a2df[i,j,:],corners=simplex_vertices(R-1))[0]
    
plot_simplex_1plast(n_eq,a_eq,a,s,0)

#print('Smoothness: ',smooth_score(a_sc,t))
#plot_contour(f, x1bound, x2bound, resolution, ax)

#df = pd.DataFrame(a2df, columns=['a','b','c'])

#fig = px.scatter_ternary(df, a="a", b="b", c="c")
#fig.show()

#z = odeint(model,z0,t,args=(S,R,v,d,dlta,s,u,K,Q))
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
#.xlim([-30,1030])
plt.setp(ax.spines.values(), linewidth=1.5)
#option 1
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=True)
#option 2
plt.xticks([0,50000,100000,150000,200000],fontsize=14)
plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.tight_layout()

#plt.ticklabel_format(axis='x',style='sci',useMathText=True)
#plt.xticks([0,5e4,1e5,1.5e5,2e5])
#plt.savefig('one_plast_rescue_noplast_fontupdate.pdf')
#plot_simplex_noplast(n_eq,a0,a,s)
#plot_gif_1plast(n_eq,a_eq,a,s,0)
s30_a104 = get_rank_dist_save_ind(n_eq)[0]
elapsed = time.time() - ti
print('Time took:',elapsed)