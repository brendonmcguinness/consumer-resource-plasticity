#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 12:13:54 2021

@author: brendonmcguinness
"""

# lag vs enzyme budget

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from scipy.spatial import ConvexHull
from scipy import stats
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from scipy.optimize import approx_fprime
from scipy.optimize import lsq_linear
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm





def monod(c, k):
    return c/(k+c)

def typeIII(c, k, n):
    return (c**n)/(k+(c**n))
    
def constraint(a, Q, dlta, sig):
    return sum(a[sig, :])/(Q[sig]*dlta[sig])-1


def heavyside_approx(a, Q, dlta, sig):
    """
    if constraint(a,E,sig) < 0:
        return 0.0
    else:
        return 1.0


    if constraint( a, E, sig) >0:
        print("contraint is > 0")
        print(constraint( a, E, sig))
        return 1.0
    else:
        return math.exp(10e5 * constraint(a, E, sig)) 

    """
    try:
        ans = 1/(1+math.exp(- (constraint(a, Q, dlta, sig) * 1000000)))
    except OverflowError:
        ans = 1/float('inf')
    return ans


def growth(c, K, a, v, sigma):
    return sum(v * monod(c, K) * a[sigma, :])




def model(y, t, S, R, v, d, dlta, s, K, Q, fr=monod):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    
    
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    K : Monod parameter
    Q : scaling parameter

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    
    
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    
    E0 = np.sum(a,axis=1)

    for sig in range(0, S):
        dndt[sig] = n[sig] * (np.sum(v * monod(c, K) * a[sig, :]) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n * a[:, i]))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                1/(E0[sig])*np.sum(v*monod(c, K)*a[sig, :])))


    dzdt = np.concatenate((dndt, dcdt, dadt.flatten()), axis=None)
    return dzdt

# def model(y, t, S, R, v, d, dlta, s, u, K, Q, fr=monod):
#     """
#     runs model from Pacciani paper adaptive metabolic strategies


#     Parameters
    
    
#     ----------
#     y : array keeping all the state variables
#     t : time array
#     S : num of species
#     R : num of resources
#     v : nutritional value of resource i
#     d : adaptation rate
#     dlta : death rate of species sigma / outflow of chemostat
#     s : supply of resource i
#     u : degradation rate of resource i
#     K : Monod parameter
#     Q : scaling parameter

#     Returns
#     -------
#     dzdt : state variables all packed into same array

#     """
    
    
#     n = y[0:S]
#     c = y[S:S+R]
#     a = y[S+R:S+R+S*R]
#     E = y[S+R+S*R:2*S+R+S*R]

#     a = np.reshape(a, (S, R))

#     dndt = np.zeros(S, dtype=float)
#     dcdt = np.zeros(R, dtype=float)
#     dadt = np.zeros(shape=(S, R), dtype=float)
#     dEdt = np.zeros(S, dtype=float)

#     if isinstance(Q, np.ndarray):
#         # chill
#         abcd = 1
#     else:
#         temp = Q*np.ones(S)
#         Q = temp

#     for sig in range(0, S):
#         dndt[sig] = n[sig] * (np.sum(v * monod(c, K) * a[sig, :]) - dlta[sig])

#     for i in range(0, R):
#         # assuming no resource outflow (u is 0 for all i)
#         dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
#                           a[:, i])) - u[i]*c[i]  # -10*c[i]
#     for sig in range(S):
#         for i in range(R):
#             dadt[sig, i] = d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
#                 1/R*np.sum(v*monod(c, K)*a[sig, :])))

#     # for sig in range(0,S):
#     #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

#     for sig in range(S):
#         for i in range(R):
#             dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
#                 heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))


# #    for sig in range(S):
# #        for i in range(R):
# #            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
# #                1/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))


#     dEdt = np.sum(dadt, 1)

#     dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
#     return dzdt

    

def compute_jacobian(func, y, *args):
    num_vars = len(y)
    num_outputs = len(func(y, 0, *args))
    jacobian = np.zeros((num_outputs, num_vars))

    # Compute the Jacobian using finite differences
    for i in range(num_vars):
        jacobian[:, i] = approx_fprime(y, lambda y: func(y, 0, *args)[i], epsilon=1e-6) #change to make smaller
        #jacobian[:, i] = derivative(lambda y: func(y, 0, *args)[i],y, dx=1e-6,args=(*args)) #change to make smaller

    return jacobian

def compute_jacobian_centDiff(func, y, *args):
    num_vars = len(y)
    num_outputs = len(func(y, 0, *args))
    jacobian = np.zeros((num_outputs, num_vars))

    # Compute the Jacobian using finite differences
    for i in range(num_vars):
        jacobian[:, i] = approx_fprime(y, lambda y: func(y, 0, *args)[i], epsilon=1e-10) #change to make smaller
        #jacobian[:, i] = derivative(lambda y: func(y, 0, *args)[i],y, dx=1e-6) #change to make smaller

    return jacobian

def shannon_diversity(counts):
    p = counts / np.sum(counts)
    H = - np.sum(p*np.log2(p))

    if math.isnan(H):
        return 0
    else:
        return H

def orderParameter(n):
    numt, S = n.shape
    diff = np.zeros((numt,S))
    #need to decide whether to normalize or not
    for i in range(S):
        diff[:,i] = np.gradient(n[:,i])
    return diff.var(axis=1)
        
def orderParameterCV(n):
    numt, S = n.shape
    diff = np.zeros((numt,S))
    #need to decide whether to normalize or not
    for i in range(S):
        diff[:,i] = np.gradient(n[:,i] / n.sum(axis=1))
    return diff.var(axis=1)
        
def plot2Dsimplex(shat, ahat, string='Equilibrium'):
    S, R = ahat.shape
    plt.figure()

    a_hat_eq_1 = ahat[:, 0]*20

    plt.hlines(1, 1, 20, colors='black')  # Draw a horizontal line
    plt.xlim(0, 21)
    plt.ylim(0.5, 1.5)

    y = np.ones(np.shape(a_hat_eq_1))   # Make all y values the same
    y2 = np.ones(np.shape(shat[0]))

    for sig in range(S):
        # Plot a line at each location specified in a
        plt.plot(a_hat_eq_1[sig]+1, y[sig], '.', ms=30,
                 label='\N{GREEK SMALL LETTER SIGMA}='+str(sig+1))

    plt.plot(shat[0]*20+1, y2, '*', ms=20,
             label='Supply Vector', color='black')
    plt.axis('off')
    plt.legend()
    titl = string + \
        " rescaled \N{GREEK SMALL LETTER ALPHA}'s and supply vector"
    plt.title(titl)
    #plt.xticks(np.arange(0, 20, step=20))
    plt.show()


def isSupplyinConvexHull2D(shat, ahat):
    S, R = ahat.shape
    count1 = 0
    count2 = 0
    for sig in range(S):
        if shat[0] < ahat[sig, 0]:
            count1 += 1
        if shat[0] > ahat[sig, 0]:
            count2 += 1
    if count1 == S or count2 == S:
        return False
    else:
        return True


def get_rank_dist_save_ind(n_eq):
    ranks = np.argsort(n_eq)[::-1]
    rd = n_eq[ranks]
    rd[rd < 0] = 1e-60
    #rdlog = np.log10(rd)
    idx = np.arange(n_eq.size)
    ind = idx[np.argsort(n_eq[idx])[::-1]]
    return rd, ind


def get_cartesian_from_barycentric(b, t):
    return t.dot(b)


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def full_point_in_hull(s, a0):
    R = s.shape[0]
    # cant do convex rhull because we can represent simplex in N-1 dimensions
    #trying basis vector on stack exchange
    if R>2:
        return point_in_hull(bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0], ConvexHull(bary2cart(a0,corners=simplex_vertices(R-1))[0]))
    else:
        #R==2
        return not ((s[0] > a0[:,0]).all() or (s[0] < a0[:,0]).all())

def polycorners(ncorners=3):
    '''
    Return 2D cartesian coordinates of a regular convex polygon of a specified
    number of corners.
    Args:
        ncorners (int, optional) number of corners for the polygon (default 3).
    Returns:
        (ncorners, 2) np.ndarray of cartesian coordinates of the polygon.
    '''

    center = np.array([0.5, 0.5])
    points = []

    for i in range(ncorners):
        angle = (float(i) / ncorners) * (np.pi * 2) + (np.pi / 2)
        x = center[0] + np.cos(angle) * 0.5
        y = center[1] + np.sin(angle) * 0.5
        points.append(np.array([x, y]))

    return np.array(points)

 
def v(n):
    '''returns height of vertex for simplex of n dimensions
    '''
    return np.sqrt((n+1)/(2*n))

      
def simplex_vertices(n):
    '''
    from https://www.tandfonline.com/doi/pdf/10.1080/00207390110121561?needAccess=true
    maybe or maybe not scaled by sqrt(3)/2
    Parameters
    ----------
    n : number of vertices

    Returns
    -------
    vert : vertices of simplex in R^n+1 cartesian space

    
    '''
    vert = np.zeros((n+1,n))
    for i in range(n+1):
        for j in range(n):
            if i - j == 1:
                vert[i,j] = np.sqrt(3)/2 * np.sqrt((j+2)/(2*(j+1))) #v(j+1)
            if i - j > 1:
                vert[i,j] = np.sqrt(3)/2 * np.sqrt((j+2)/(2*(j+1))) / (j+2) # v(j+1)/(j+2)
    return vert
 
       


def bary2cart(bary, corners=None):
    '''
    Convert barycentric coordinates to cartesian coordinates given the
    cartesian coordinates of the corners.
    Args:
        bary (np.ndarray): barycentric coordinates to convert. If this matrix
            has multiple rows, each row is interpreted as an individual
            coordinate to convert.
        corners (np.ndarray): cartesian coordinates of the corners.
    Returns:
        2-column np.ndarray of cartesian coordinates for each barycentric
        coordinate provided.
    '''

    if np.array((corners == None)).any():
        corners = polycorners(bary.shape[-1])

    cart = None

    if len(bary.shape) > 1 and bary.shape[1] > 1:
        cart = np.array([np.sum(b / np.sum(b) * corners.T, axis=1)
                        for b in bary])
        
    else:
        cart = np.sum(bary / np.sum(bary) * corners.T, axis=1)

    return cart,corners

def cart2bary(cart):
    #can only do triangle
    x = cart[0]
    y = cart[1]
    x1,x2,x3 = simplex_vertices(2).T[0]
    y1,y2,y3 = simplex_vertices(2).T[1]
        
    l1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3))
    l2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3))
    l3 = 1-l1-l2
    return l1,l2,l3

def ndim_simplex(n):
    vert = np.zeros((n+1,n+1)) 
    e = np.identity(n+1)
    for i in range(n+1):
        vert[i,:] = 1/(np.sqrt(2))*e[i,:] + (1/(n*np.sqrt(2))*(1+1/np.sqrt(n+1))*np.ones(n+1)) + 1/np.sqrt(2*(n+1))*np.ones(n+1)
    return vert



def slope(n1, n2, t1, t2):
    return (n2-n1) / (t2-t1)


def find_eqt_of_n(n, t, eps):
    sl = np.zeros(t.shape[0])
    for i in range(t.shape[0]-1):
        sl[i] = slope(n[i], n[i+1], t[i], t[i+1])
    for i in range(t.shape[0]-30):
        if np.sum(np.abs(sl[i:i+30])) < 30*eps*n[i] or np.sum(n[i:i+30]) < 30*eps:
            return i
    return t.shape[0]-1


def community_resilience(n, t, eps):
    S = n.shape[1]
    res = []
    if len(n.shape) == 3:
        S = S*n.shape[2]
        n = n.reshape(*n.shape[:-2], -1)
        res = []        
        #print(S)
    for sig in range(S):
        if n[-1, sig] > 1.0:
            res.append(t[find_eqt_of_n(n[:, sig], t, eps)])
    return np.mean(res)

def avg_eq_time(c,t,rel_tol=0.001):
    stable_index = np.zeros(c.shape[1])
    for i in range(c.shape[1]):
        try:
            #try
            stable_index[i] = np.where(np.abs(c[-1,i] - c[:,i]) > (c[-1,i]*rel_tol))[0][-1] + 1                
        except:
            print('not finding eq time')
            stable_index[i] = t.shape[0]-1#t[-1] #maybe delete this
            
        if stable_index[i] == t.shape[0]:
            stable_index[i] = t.shape[0]-1#t[-1]
    return np.mean(t[stable_index.astype(int)])

def avg_eq_time_traits(a,t,rel_tol=0.001):
    S,R = a.shape[1:]
    stable_index = np.zeros((S,R))
    for i in range(S):
        for j in range(R):
            try:
                stable_index[i,j] = np.where(np.abs(a[-1,i,j] - a[:,i,j]) > (a[-1,i,j]*rel_tol))[0][-1] + 1
            except:
                stable_index[i,j] = t[-1]
    return np.mean(t[stable_index.flatten().astype(int)])

        
def pick_inout_hull(S, R, E0, a=10e-6, b=10e-2, inout=False,di=None):
    if di.any() == None:
        di = np.ones(R)
    a0 = np.zeros((S, R), dtype=float)
    for i in range(0, S):
        a0[i, :] = np.random.dirichlet(di*np.ones(R), size=1) * E0[i]
    a0_scaled = a0/E0[:, None]
    s = np.random.uniform(a, b, R)
    if full_point_in_hull(s, a0_scaled) == inout:
        return s, a0
    else:
        return pick_inout_hull(S, R, E0, inout=inout,di=di)
    
def pick_inout_hull_a0(s, S, R, E0, inout=True):
    a0 = np.zeros((S, R), dtype=float)
    for i in range(0, S):
        #dira = np.random.randint(1,dr+1,size=R)
        a0[i, :] = np.random.dirichlet(np.ones(R), size=1) * E0[i]
    a0_scaled = a0/E0[:, None]
    if full_point_in_hull(s, a0_scaled) == inout:
        return a0
    else:
        return pick_inout_hull_a0(s, S, R, E0, inout=inout)


def pick_inout_hull_s(E0, a0, a=10e-6, b=10e-2, inout=True):
    S, R = a0.shape
    a0_scaled = a0/E0[:, None]
    s = np.random.uniform(a, b, R)
    if full_point_in_hull(s, a0_scaled) == inout:
        return s
    else:
        return pick_inout_hull_s(E0, a0, inout=inout)


def get_rank_key(mode, key):
    ix, = np.where(mode.flatten() == np.argmax(key))
    if ix.size > 0:
        return ix[0]
    return get_rank_key(mode, np.delete(key, np.argmax(key)))



def centeroidnp(arr):
    length,dim = arr.shape
    summ = np.zeros(dim)
    for i in range(dim):
        summ[i] = np.sum(arr[:, i])
        
    return summ/length

def weighted_centroid(a,w):
    length,dim = a.shape
    summ = np.zeros(dim)
    for i in range(dim):
        summ[i] = np.sum((a[:, i]*w) /w.sum())
        
    return summ #/length
    #R = a.shape[1]
    #ac,corners = bary2cart(a,corners=simplex_vertices(R-1))    
    #return centeroidnp((a*w[:,None])/w.sum())[:,None] 
    
    

def get_fd_and_centroid(a):
        #make simplex
    #ind_del = 4
    #ind_samp = 0
    #get rank of trait point we deleted
    #rank_del = int(df.iloc[ind_samp*10+ind_del]['Rank'])
    #ares_del = np.delete(ares,ind_del,axis=2)
    #to get trait points organized by rank
    #rs,inds = get_rank_dist_save_ind(nres[ind_samp,ind_del,:])
    
    S,R = a.shape
    ac,corners = bary2cart(a,corners=simplex_vertices(R-1))
    ac1,corners1 = bary2cart(a,corners=None)

    if R>S or R<3:
        return -1, centeroidnp(ac), -1
    else:
        hull = ConvexHull(ac)
    
        
        return hull.volume, centeroidnp(ac), S-len(hull.vertices)

def supply_to_centroid(s,a,E0):
    S,R = a.shape
    if a.sum() < S-1:
        a = a/E0[:,None]
    ac = bary2cart(a,corners=simplex_vertices((R-1)))[0]
    sc = bary2cart(s,corners=simplex_vertices((R-1)))[0]
    cent = centeroidnp(ac)
    return np.sqrt(np.sum((cent-sc)**2))

def supply_to_weighted_centroid(s,a,n,E0):
    S,R = a.shape
    if a.sum() < S-1:
        a = a/E0[:,None]
    ac = bary2cart(a,corners=simplex_vertices((R-1)))[0]
    sc = bary2cart(s,corners=simplex_vertices((R-1)))[0]
    cent = weighted_centroid(ac,n)
    return np.sqrt(np.sum((cent-sc)**2))    

def distance(s,a,E0):
    R = s.shape[0]
    sc = bary2cart(s,corners=simplex_vertices((R-1)))[0]
    ac = bary2cart(a/E0[0],corners=simplex_vertices((R-1)))[0]
    return  np.sqrt(np.sum((ac-sc)**2))

def distanceN0(n01,n02):
    return np.sqrt(np.sum((n01-n02)**2))
    
def distanceA0(a01,a02,E0):
    S, R = a02.shape
    ac1 = bary2cart(a01/E0[0],corners=simplex_vertices((R-1)))[0]
    ac2 = bary2cart(a02/E0[0],corners=simplex_vertices((R-1)))[0]
    return np.sqrt(np.sum((ac1-ac2)**2))

def get_area(ac):
    hull = ConvexHull(ac)
    return hull.volume


def get_cen_dn(a_eq,E0,s):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1)) #.sum(axis=1)#a_bar = bary2cart()
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
            
    comp_eff = comp.sum(axis=2).sum(axis=0)
    cen = comp_eff #/comp_eff.max()
    dn = dist #/dist.max()
    #un normalized
    return cen,dn 

def get_comp_std(a_eq,E0,s):
    #variation of compeitive distances
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
            
    A = comp.sum(axis=2)
    
    comp_var = np.std(A[A!=0])
    #un normalized
    return comp_var

def comp_dist(a_eq,n_eq,s,E0):
    #gets competitive distances
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    return comp.sum(axis=2)

def pred_rad_from_traits(a_eq,n_eq,s,E0):
    #predicts slope of relative abundance distribution from traits (initial or final)
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),dist]).T
    Y=n_eq.T
    reg = LinearRegression().fit(X,Y)
    response = reg.predict(X)
    scaled_regcoef = reg.coef_/(np.abs(reg.coef_).max())
    
#    return reg.score(X,Y),(scaled_regcoef[0]*comp_eff/(S-1))+(dist*scaled_regcoef[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef
    return reg.score(X,Y),(reg.coef_[0]*comp_eff/(S-1))+(dist*reg.coef_[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef

def pred_rad_from_traits_noscale(a_eq,n_eq,s,E0):
    #does the same of pred_rad_from_traits() but coefficients are not scaled
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]

    
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),-dist]).T
    Y=n_eq.T
    reg = LinearRegression(fit_intercept=True,positive=True).fit(X,Y)
    #reg = Ridge(alpha=1.0).fit(X,Y)
    response = reg.predict(X)
    
#    return reg.score(X,Y),(scaled_regcoef[0]*comp_eff/(S-1))+(dist*scaled_regcoef[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef
    return reg.score(X,Y),(reg.coef_[0]*comp_eff/(S-1))+(dist*reg.coef_[1]),(comp_eff/(S-1)),dist,reg.coef_,reg,response

def pred_rad_from_comp_noscale(a_eq,n_eq,s,E0):
    #just from competitive distances
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1)]).T
    Y=n_eq.T
    reg = LinearRegression(fit_intercept=True).fit(X,Y)
    #reg = Ridge(alpha=1.0).fit(X,Y)
    response = reg.predict(X)
    
#    return reg.score(X,Y),(scaled_regcoef[0]*comp_eff/(S-1))+(dist*scaled_regcoef[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef
    return reg.score(X,Y)

def pred_rad_from_dist_noscale(a_eq,n_eq,s,E0):
    #just from supply vector distance
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))

    X = np.array([-dist]).T
    Y=n_eq.T
    reg = LinearRegression(fit_intercept=True).fit(X,Y)
    #reg = Ridge(alpha=1.0).fit(X,Y)
    response = reg.predict(X)
    
#    return reg.score(X,Y),(scaled_regcoef[0]*comp_eff/(S-1))+(dist*scaled_regcoef[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef
    return reg.score(X,Y)

def pred_rad_multiple(a_eq,n_eq,s,E0):
    #predicts rand abundance distribution but gets total score, as well as from just competitive and just supply distance 
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]

    
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),-dist]).T
    
    df = pd.DataFrame(X,columns = ['x1','x2'])
    df['y'] = n_eq.T 
    lm = ols('y ~ x1 + x2',df).fit()
    anova_table = anova_lm(lm)
    score = anova_table['sum_sq'][:-1].sum() / anova_table['sum_sq'].sum()
    scorec = anova_table.loc['x1','sum_sq'] / anova_table['sum_sq'].sum()
    scored = anova_table.loc['x2','sum_sq'] / anova_table['sum_sq'].sum()
    
    return score, scorec, scored, lm.resid

def pred_rad_multiple_nointercept(a_eq,n_eq,s,E0):
    #predicts rand abundance distribution but gets total score, as well as from just competitive and just supply distance 
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]

    
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),-dist]).T
    
    df = pd.DataFrame(X,columns = ['x1','x2'])
    df['y'] = n_eq.T 
    lm = ols('y ~ x1 + x2 -1',df).fit()
    anova_table = anova_lm(lm)
    score = anova_table['sum_sq'][:-1].sum() / anova_table['sum_sq'].sum()
    scorec = anova_table.loc['x1','sum_sq'] / anova_table['sum_sq'].sum()
    scored = anova_table.loc['x2','sum_sq'] / anova_table['sum_sq'].sum()
        
    return score, scorec, scored, lm.resid


def pred_ranks(a_eq,n_eq,s,E0):
    #predicts ranks from traits
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]

    ranks, ind = get_rank_dist_save_ind(n_eq)
    
    
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),-dist]).T
    
    df = pd.DataFrame(X,columns = ['x1','x2'])
    df['y'] = n_eq.T 
    lm = ols('y ~ x1 + x2',df).fit()
    c,b,a = lm.params
    yy = a*comp_eff - b*dist + c
    indp = get_rank_dist_save_ind(yy)[1]

    return ind, indp, n_eq, yy

def chooseAbundWeights(a,s):
    a = bary2cart(a,corners=simplex_vertices(a.shape[1]-1))[0]
    s = bary2cart(s,corners=simplex_vertices(s.shape[0]-1))[0]
    x = lsq_linear(a.T, s, bounds=(0, 1), lsmr_tol='auto', verbose=0).x
    return x / x.sum()
    

def pickPointInitDist(s, dist, count=0):
    #will only work for R=3
    if count > 500:
        return -1

    rad = np.random.uniform(0, 2*np.pi)
    cart = bary2cart(s, corners=simplex_vertices(2))[0]
    
    p = cart2bary((cart[0] + dist * np.cos(rad), cart[1] + dist * np.sin(rad)))
    
    if full_point_in_hull(np.array(p), np.identity(s.shape[0])):
        return np.array(p)
    else:
        #print(count)
        return pickPointInitDist(s, dist, count + 1)
    

def pred_abund_from_abund(neq1,neq2):
    #predicts equilibrium abundances from initial abundances, gets at whether trait driven selection (sorting) occurs
    df = pd.DataFrame(neq1,columns = ['x'])
    df['y'] = neq2.T 
    lm = ols('y ~ x',df).fit()
    anova_table = anova_lm(lm)
    score = anova_table['sum_sq'][:-1].sum() / anova_table['sum_sq'].sum()
    return score

def pred_abund_from_abund_log(neq1, neq2):
    # Predicts equilibrium abundances from initial abundances after log-transforming data.
    # Adds a small constant to handle zeros before log transformation.
    epsilon = 1e-9
    df = pd.DataFrame({
        'x': np.log(neq1 + epsilon),
        'y': np.log(neq2 + epsilon)
    })
    lm = ols('y ~ x', df).fit()
    anova_table = anova_lm(lm)
    score = anova_table['sum_sq'][:-1].sum() / anova_table['sum_sq'].sum()
    
     # Extract slope and intercept
    c = lm.params['x']       # slope
    b = lm.params['Intercept']  # intercept
    
    return score, c, b, lm.resid

def pred_abund_from_abund_log_fixed_slope(neq1, neq2):
    """
    Predicts equilibrium abundances from initial abundances assuming fixed slope = 1 in log-log space.
    Fits only intercept. Computes R^2 in log space.

    Args:
        neq1: array-like, initial abundances
        neq2: array-like, final abundances

    Returns:
        score: R^2 in log space
        intercept: fitted intercept (b)
        residuals: model residuals (log(n_final) - predicted log(n_final))
    """
    epsilon = 1e-9  # small constant to avoid log(0)

    # Log-transform initial and final abundances
    log_init = np.log(neq1 + epsilon)
    log_final = np.log(neq2 + epsilon)

    # Create response variable: difference in logs
    log_diff = log_final - log_init
    df = pd.DataFrame({'log_diff': log_diff})

    # Fit intercept-only model
    model = ols('log_diff ~ 1', data=df).fit()
    intercept = model.params['Intercept']

    # Predicted log(final) abundance
    log_final_pred = intercept + log_init

    # Compute R^2 manually
    ss_res = np.sum((log_final - log_final_pred) ** 2)
    ss_tot = np.sum((log_final - np.mean(log_final)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Residuals in log space
    residuals = log_final - log_final_pred

    return r_squared, intercept, residuals
def pred_abund_from_abund_log_nointercept(neq1, neq2):
    # Predicts equilibrium abundances from initial abundances after log-transforming data.
    # Adds a small constant to handle zeros before log transformation.
    epsilon = 1e-9
    df = pd.DataFrame({
        'x': np.log(neq1 + epsilon),
        'y': np.log(neq2 + epsilon)
    })
    lm = ols('y ~ x-1', df).fit()
    anova_table = anova_lm(lm)
    score = anova_table['sum_sq'][:-1].sum() / anova_table['sum_sq'].sum()
    
    
    return score, lm.resid


def pred_rad_from_weighted_traits(a_eq,n_eq,s,E0):
    
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),dist]).T
    Y=n_eq.T
    
    #params = np.polyfit(X, np.log(Y), 1, w=np.sqrt(Y))
    reg = LinearRegression().fit(X,np.log(Y))
    response = reg.predict(X)
    scaled_regcoef = reg.coef_/(np.abs(reg.coef_).max())
    
    return reg.score(X,Y),(reg.coef_[0]*comp_eff/(S-1))+(dist*reg.coef_[1]),(comp_eff/(S-1)),dist,reg.coef_,reg,response