
'''
This file containts the scripts for contour generation under the elastica theory.

The elastica energy is estimated using the integration method outlined in Sharon et al.

I.e. it's assumed to be a polynomial f = function, and 
'''

from __future__ import division
import pylab as pl
import holoviews as hv
from scipy.integrate import simps
from scipy.optimize import fsolve

## set parameters for single estimation
#N = 10 # number of edges to simulate
#n = 0 # number of variables in polynomial (n+1)
#nn = 4 # number of spots to use for integration
#p = 3 # integration type (3 = simpsons method)

# functions
def Psi(a,s,psi1,psi2):
    ''' 
    Basic function for the polynomial for the curve
    # a = polynomical constants (needs to be in list/array form so len() command works)
    # s = position along polynomial
    # psi1,psi2: start and end orientation
    '''
    psi = (1-s)*psi1 + s*psi2 + s*(1-s)*pl.polyval(a,s)
    # note: the function polyval takes its arguments backwards, so that output is e.g. a1*x**3 + a2*x**2 + a1*x etc.
    return psi

def dPsi(a,s,psi1,psi2):
    ''' 
    First derivative of polynomial
    # a = polynomical constants (needs to be in list/array form so len() command works)
    # s = position along polynomial
    # psi1,psi2: start and end orientation
    '''
    
    n = len(a) # number of variables in polynimal
    ns = len(s) # number of parts to use for the contour
    a1 = a*(pl.arange(n-1,-1,-1)+1)
    a2 = pl.zeros(n+1)
    a2[:-1] = a*(pl.arange(n-1,-1,-1)+2)
    
    psi = -psi1 + psi2 + pl.polyval(a1,s) -pl.polyval(a2,s)
   
    return psi

def ddPsi(a,s):
    '''
    # second derivative of polynomial
    # a = polynomical constants (needs to be in list/array form so len() command works)
    # s = position along polynomial (needs to be in list/array form so len() command works)
    '''
    n = len(a) # number of variables in polynimal
    a1 = a[:-1]*(pl.arange(n-1,0,-1)+1)*pl.arange(n-1,0,-1)
    a2 = a*(pl.arange(n-1,-1,-1)+2)*(pl.arange(n-1,-1,-1)+1)
    
    psi = pl.polyval(a1,s) - pl.polyval(a2,s)
   
    return psi
        
def simpssin(nn,psi1,psi2,a):
    '''
    # finds the sum(w_j sin(Psi_j)) integral
    # a = polynomical constants (needs to be in list/array form so len() command works)
    # s = position along polynomial 
    # psi1,psi2: start and end orientation
    # nn: number of integration points
    '''
    s = pl.linspace(0,1,nn)
    psi = Psi(a,s,psi1,psi2)
    simpsum = simps(pl.sin(psi),s) 
    return simpsum

def simpscos(nn,psi1,psi2,a):
    '''
    # finds the sum(w_j sin(Psi_j)) integral
    # a = polynomical constants (needs to be in list/array form so len() command works)
    # s = position along polynomial 
    # psi1,psi2: start and end orientation
    # nn: number of interation points
    '''
    s = pl.linspace(0,1,nn)
    psi = Psi(a,s,psi1,psi2)
    simpsum = simps(pl.cos(psi),s) 
    return simpsum

def errors(a,psi1,psi2,nn):
    '''
    find the Newton errors, vector of
    ddPsi(i+1/n+2) + lambda cos(Psi(i+1/n+2)) = 0 (0 <= i < n)
    and
    um(w_j sin(Psi_j)) = 0  
    
    Inputs
    ----------
    # a = polynomical constants (needs to be in list/array form so len() command works)
    # s = position along polynomial (needs to be in list/array form so len() command works)
    # psi1,psi2: start and end orientation
    # nn: number of interation points
    
    Outputs
    ----------
    # v: vector of 
    ddPsi(i+1/n+2) + lambda cos(Psi(i+1/n+2)) = 0 (0 <= i < n)
    and
    um(w_j sin(Psi_j)) = 0  
    '''
    n = len(a)-1
    v = pl.zeros(n+1)
    s = (pl.arange(n)+1)/(n+1)
    v[:-1] = ddPsi(a[:-1],s)+a[-1]*pl.cos(Psi(a[:-1],s,psi1,psi2))
    v[-1]  = simpssin(nn,psi1,psi2,a[:-1])
    return v 


def D(c,f,X):
    '''
    The derived elastica energy as in sharon
    E= a**2 etc.
    '''    
    # either c or f can be an array! not both
    # c: current bar orientation
    # f: flanker orientation
    # x: relative flanker distance and position orientation             
    x = X[0]
    y = X[1]

    ## 'Affinity' D
    # D = Ba^2 + Bb^2 -BaBb
    # Here Ba is the angle of the flanker with the line connecting it with the center
    # and Bb is the reverse for the center
    # See figure 5 in Leung & Malik (1998) for intuitive figure

    # flanker positional angles
    theta = pl.arctan2(x,y)
#        if theta > pi/2: theta-=pi   
#        if theta < -pi/2: theta+=pi

    # B values normalized within -pi to pi
    Ba = pl.arctan(pl.tan(0.5*(-f+theta)))*2     
    Bb = pl.arctan(pl.tan(0.5*(c-theta)))*2


    D = 4*(Ba**2 + Bb**2 - Ba*Bb)
    return D
    
def findcurve(psi1,psi2,n=3,nn_fit=4,nn_out=100):
    '''
    Function to find the elastica curve for start and end orientations
    psi1 and psi2. It finds the best curve across all directions from start
    and end, i.e. the direction independent elastica curve.
    
    Inputs
    ------------
    psi1,psi2: start and end orientations.
    n:     degree of estimation polynomial.
    nn:    number of points on the curve.
             - nn_fit: for fittin purposes
             - nn_out: for the output
    
    Outputs
    ------------
    Returns a tuple (s,psi). 
    s:   points on the curve.
    psi: curvature of the curve as a function of s.
    E:   curvature energy of the curve
    '''
    # 
    
    # define the starting conditions
    a0 = pl.zeros(n+1) 
    
    # Set a high energy: 
    E_best = 10000  
    
    # and predfine output curve
    s       = pl.linspace(0,1,nn_out) # points on the curve
    psi_out = pl.zeros(nn_out)        # curvature at points in curve
    
    
    # across all the start and end directions find the curve with the lowest energy    
    for dpsi1 in (-pl.pi,0,pl.pi):
        for dpsi2 in (-pl.pi,0,pl.pi):
            # For the starting variables,
            # the first two polygon variables can be estimated from the Sharon paper derivation
            # For different starting variables the solution can be hard to find            
            a0[-2] = 4*(   pl.arcsin(- (pl.sin(psi1+dpsi1)+ pl.sin(psi2+dpsi2))/4)    -(psi1+dpsi1+psi2+dpsi2)/2       )
            a0[-1] = 2*a0[-2]/pl.cos( (psi1+dpsi1+psi2+dpsi2)/2 + a0[-2]/4  )               
            
            # find the best variables to minimize the elastica energy
            fit = fsolve(errors,a0,args=(psi1+dpsi1,psi2+dpsi2,nn_fit))
    
            # find the curve and its derivative for the fitted variables
            a    = fit[:-1]
            psi  = Psi(a,s,psi1+dpsi1,psi2+dpsi2)
            dpsi = dPsi(a,s,psi1+dpsi1,psi2+dpsi2)
    
            # find the energy of this curve
            E = sum(dpsi**2)*s[1]
            
            # check against the lowest energy
            if E_best > E:
                E_best = E
                psi_out[:] = pl.copy(psi)    
    
    return (s,psi_out,E_best)
    
def plotbar(x,y,th,color='k',width=2,l=1):
    ''' 
    Plot a single bar 
    x,y: location middle of bar
    th(eta): orientation
    color: color (default = black)
    width: linewidth (default = 2)
    l:     line length (default = 1)
    Returns a holoviews curve object
    '''
#    th += pl.pi/4 # so that the orientation is relative to the vertical
    hl = l/2 # half length bar
    
    # define x and y points of bar
    X = [x-pl.sin(th)*hl,x+pl.sin(th)*hl]
    Y = [y-pl.cos(th)*hl,y+pl.cos(th)*hl]
    
    # return holoviews curve
    return hv.Curve(zip(X,Y))    
    
    