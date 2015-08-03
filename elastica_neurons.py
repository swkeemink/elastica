
'''
This file containts the scripts for contour generation under the elastica theory.

The elastica energy is estimated using the integration method outlined in Sharon et al.

I.e. it's assumed to be a polynomial f = function, and 

author: sanderkeemink@gmail.com
'''


from __future__ import division
import holoviews as hv
import pylab as pl
from pylab import exp,cos,sin,pi,tan

def mises(a,k,ref,x):
    ''' basic von mises function
    Inputs
    -----------------------
    - a: determines magnitude
    - k: determines width (low k is wider width)
    - ref: reference angle
    - x: input angle
    '''
    return a*exp(k*(cos(2*(ref-x))))

class scene:
    """ Scene class. Define a scene, for which the neural responses to each bar can then be found """
    def __init__(self,N,O,X,dim,Kc,Ac,a,E0):
        self.n = len(O) # number of bars
        self.N = N      # number of neurons per location
        self.O = O      # bar orientations
        self.X = X      # bar locations
        self.dim = dim  # image dimension [x,y] 
        self.ang = pl.linspace(-pi/2,pi/2-pi/N,N) # orientations
        self.Kc = Kc    # determines width of tuning to bars in location (lower Kc is wider tuning)
        self.Ac = Ac    # determines response magnitude
        self.a  = a     # determines strength of modulation
        self.E0 = E0    # offset for modulation
        self.FRc = pl.zeros((self.n,N))   # 'firing rates' or 'probabilities' for local bars
        self.FRs = pl.zeros((self.n,N))   # modulation from smoothest targets (net effect)
        self.FR = pl.zeros((self.n,N))   # final 'firing rates' or 'probabilities' for each bar
        self.est = pl.zeros(self.n)   # to store the orientation estimates
  
    def popvec(self,X):
        ''' Population vector for the set of responses X, with each value in 
        the vector X corresponding to an angle in self.ang
        X is a 1D vector of length len(self.ang)
        Returns the angle of the population vector.
        '''
        # define vector coordinates
        v = pl.zeros((2,self.N))
        v[0,:] = cos(2*self.ang)
        v[1,:] = sin(2*self.ang)

        # find population vector
        vest0 = pl.sum(((X-min(X))/max(X))*v,1)
        
        # return the angle of the population vector
        return 0.5*pl.arctan2(vest0[1],vest0[0])
        
    def plotscene(self,length=1,oriens = 'NA',alphas='NA',colors='off'):
        ''' Plot the scene

        Inputs
        ----------------
        - length: length of each plotted bar
        - oriens: orientations of each bar, if 'NA' use the orientations as defined in self.O
        - alphas: the alpha values for each bar, if 'NA' set all to one
        - colors: the color of each bar, if 'NA' all are black
        oriens and alphas should all be arrays of length self.n if 'off'
        
        '''
        # check for alphas/oriens inputs
        if alphas == 'NA': alphas = pl.ones(self.n)
        if oriens == 'NA': oriens = self.O
            
        # initiate image
        img = hv.Curve([])
        
        for i in range(self.n):
            # get bar location
            x = self.X[i,0]
            y = self.X[i,1]
            # get bar orientation
            f = oriens[i]
            # check colors
            if colors == 'off':
                c = 'k'
            else:
                if alphas[i]<pl.mean(alphas): # if below average make blue
                    c = 'b'
                else: # otherwise make red
                    c = 'r'
            # plot the bar
            img *= hv.Curve(zip([x-sin(f)*length,x+sin(f)*length],[y-cos(f)*length,y+cos(f)*length]))#,color=c,linewidth=4,alpha = alphas[i]/max(alphas))
        
        # return img object
        return img        
        
    def findE(self,c,f,X):
        ''' find E, the approximated sharon energy
        Inputs
        ----------
        - c: current bar orientation
        - f: flanker orientation
        - X: relative flanker distance and position orientation         
        '''
        # define x and y 
        x = X[0]
        y = X[1]      
        
        # flanker positional angle
        theta = pl.arctan2(x,y)
        
        # find and return D
        Ba = pl.arctan(tan(0.5*(-f+theta)))*2     
        Bb = pl.arctan(tan(0.5*(c-theta)))*2
        return  4*(Ba**2 + Bb**2 - Ba*Bb)
        
    def E(self,c,f,X):
        ''' find the direction invariant approximated sharon energy
        Inputs
        ----------
        - c: current bar orientation (can be double or array, if array D returns an array of D values)
        - f: flanker orientation  (double)
        - X: relative flanker distance and position orientation         
        '''
        
        # check if c is an array, and assign length (note, f cannot be array)
        if isinstance(c,pl.ndarray):
            length = len(c)
        else:
            length = 1        
        
        # find D candiates across all directions
        E = pl.zeros((length,4))  
        E[:,0] = self.findE(c,f,X)
        E[:,1] = self.findE(c+pi,f,X)
        E[:,2] = self.findE(c,f+pi,X)
        E[:,3] = self.findE(c+pi,f+pi,X)
            
        # return the minimum energy
        return E.min(1)    

    def plotlocalmod(self,iLoc,torus='on'):
        ''' Plot contributions to local modulation
        iLoc: location id (which bar to look at)
        torus: whether image is on a torus or not
        '''
        # make mask to select all locations except iLoc
        mask = pl.ones(self.n)
        mask[iLoc] = 0
        
        # find orientations at all other locations
        F = self.O[mask==1]
        
        # find positions of all other locations
        X = self.X[mask==1,:]  ##### set up properly, now based on [x,y], should be [r,theta]
        # and set coordinates to be relative to iLoc
        X[:,0]-=self.X[iLoc,0] # update x coordinate to make current the center
        X[:,1]-=self.X[iLoc,1] # update y coordinate to make current the center
        
        # transform other coordinates according to torus
        # does not work when there are only two bars present! 
        if torus == 'on':
            mx = max(self.X[:,0])
            X[X[:,0]<-mx,0]+=pl.sqrt(self.n)*mx/pl.floor(pl.sqrt(self.n)/2)
            X[X[:,1]<-mx,1]+=pl.sqrt(self.n)*mx/pl.floor(pl.sqrt(self.n)/2)
            X[X[:,0]>mx,0]-=pl.sqrt(self.n)*mx/pl.floor(pl.sqrt(self.n)/2)
            X[X[:,1]>mx,1]-=pl.sqrt(self.n)*mx/pl.floor(pl.sqrt(self.n)/2)
            
        # find the distances from every other bar to the 'center'
        R = pl.sqrt(X[:,0]**2+X[:,1]**2) 
        
        
        # initiate image        
        img = hv.Curve([])

        # find the total modulation, and plot each individual contribution
        for i in range(self.n-1):
            # find curvature energy
            E  =  self.E(self.ang,F[i],X[i,:])
            # find resulting modulation
            h  = exp(-self.a*(E-self.E0)/R[i])
            # plot the modulation curve
            img *= hv.Curve(zip(self.ang/pi,h))#,alpha=0.25,label=i)

        # return img
        return img

    def simulate(self,iLoc,torus='on'):
        '''
        For a location iLoc simulate the responses given the scene:
        - find the orientation at iLoc to get the drive
        - find the orientation and relative position of all other bars, to determine the modulation
        
        Updates FRc, FRs, FR, S , est for iLoc
        
        Inputs
        -------------------------
        - iLoc: the index of the location to simulate
        - torus: if 'on', put the scene on a torus
        '''        
        # find the orientation of the bar in iLoc
        c = self.O[iLoc] # 'centre' flanker
        
        # make mask to select all locations except iLoc
        mask = pl.ones(self.n)
        mask[iLoc] = 0
        
        # find orientations at all other locations
        F = self.O[mask==1]
        
        # find positions of all other locations
        X = self.X[mask==1,:]  ##### set up properly, now based on [x,y], should be [r,theta]
        # and set coordinates to be relative to iLoc
        X[:,0]-=self.X[iLoc,0] # update x coordinate to make current the center
        X[:,1]-=self.X[iLoc,1] # update y coordinate to make current the center
        
        # transform other coordinates according to torus, if necessary
        # does not work when there are only two bars present! 
        if torus == 'on':
            mx = max(self.X[:,0])
            X[X[:,0]<-mx,0]+=pl.sqrt(self.n)*mx/pl.floor(pl.sqrt(self.n)/2)
            X[X[:,1]<-mx,1]+=pl.sqrt(self.n)*mx/pl.floor(pl.sqrt(self.n)/2)
            X[X[:,0]>mx,0]-=pl.sqrt(self.n)*mx/pl.floor(pl.sqrt(self.n)/2)
            X[X[:,1]>mx,1]-=pl.sqrt(self.n)*mx/pl.floor(pl.sqrt(self.n)/2)
            
        # find the distances from every other location to iLoc
        R = pl.sqrt(X[:,0]**2+X[:,1]**2) 

        # find the drive from the orientation in iLoc       
        rc =  mises(self.Ac,self.Kc,c,self.ang)
        
        # find the modulation from all other lcoations
        rs=1
        for i in range(self.n-1): # for all locations except iLoc
            # find D across all preferred orientations
            E =  self.E(self.ang,F[i],X[i,:])
            
            # update the modulation
            rs*=exp(-self.a*(E-self.E0)/R[i])
            
        # find final response in location iLoc across preferred oreintations
        rf = rc*rs
        
        # update relevant variables
        self.FRc[iLoc,:] = rc#/sum(rc)
        self.FRs[iLoc,:] = rs
        self.FR[iLoc,:] = rf#/sum(rf)
        self.est[iLoc] = self.popvec(rf)

    def simulate_all(self):
        '''
        Simulate all locations
        '''
        for i in range(self.n):
            self.simulate(i)
            
    def saliency(self,base):
        '''
        Calculate the saliency for all location either based on:
        - base = 'mean': the mean responses
        - base = 'max' : the max responses
        '''
        sal = pl.zeros(self.n)
        if base == 'mean':
            for i in range(self.n):
                sal[i] = pl.mean(self.FR[i,:])
            self.sal = sal/max(sal)
            
        if base == 'max':
            for i in range(self.n):
                sal[i] = max(self.FR[i,:])
            self.sal = sal/max(sal)
     
        
        
        
    