import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import scipy
float_type=np.longdouble
class Log:
    def __init__(self):
        self.log={'grad': [],'x':[],'f':[]}
    def __getitem__(self,k):
        return np.asarray(self.log[k])
    def to_numpy(self):
        a=np.stack(np.asarray(v) for v in self.log.values())
        return a,self.log.keys()
    def __call__(self,f,x): 
        
        self.log['grad'].append(np.linalg.norm(f.grad(x))**2)
        self.log['x'].append(np.linalg.norm(x)**2)
        self.log['f'].append(np.linalg.norm(f(x)))
        

class BetaQuadratic:
    def __init__(self,r=1,sigma=1,n=600,tau=1/2,xi=-1/2):
        evs=scipy.stats.beta(a=xi+1,b=tau+1).rvs(size=n)
        u=scipy.stats.ortho_group.rvs(n)
        
        self.A=(u@np.diag(evs)@u.T)
        self.x0=np.random.normal(size=(n,1)).astype(float_type)
        self.L=eigh(self.A,eigvals_only=True,subset_by_index=[n-1,n-1])
        
        
    def plot(self):
        ev,_=np.linalg.eigh(self.A)
        plt.hist(ev[ev<100],bins=30,density=True)
        plt.show()
    
    def __call__(self,x):
        return 1/2* x.T@self.A@x
    
    def grad(self,x):
        return self.A@x
    

class RandomQuadratic:
    def __init__(self,r=1,sigma=1,n=600):
        m=int(n*r)
        X=np.random.normal(size=(m,n),scale=sigma)
        self.A=1/n*(X@X.T)
        self.x0=np.random.normal(size=(m,1))
        eigs=eigh(self.A,eigvals_only=True)
        self.L=eigs[-1]
        self.l=eigs[eigs>1e-4][0]
        
    def plot(self):
        ev,_=np.linalg.eigh(self.A)
        plt.hist(ev[ev<100],bins=30,density=True)
        plt.show()
    
    def __call__(self,x):
        return 1/2* x.T@self.A@x
    
    def grad(self,x):
        return self.A@x
    
    
class FeaturesQuadratic:
    def __init__(self,X,y,torch=False):
        self.X=X/np.sqrt(X.shape[0])
        self.A=self.X.T@self.X
        if torch:
            self.x0=torch.randn((self.X.shape[1],1),device='cuda')
        else:
            self.x0=np.random.normal(size=(self.X.shape[1],1))
        self.y=y[:,None]
        m=len(self.A)
        self.L=eigh(self.A,eigvals_only=True,subset_by_index=[m-1,m-1])
        
    def plot(self):
        ev,_=np.linalg.eigh(self.A)
        plt.hist(ev[ev<100],bins=30,density=True)
        plt.show()
    
    def __call__(self,x):
        return 1/2*np.linalg.norm(self.X@x-self.y)**2
    
    def grad(self,x):
        return self.X.T@(self.X@x-self.y)


