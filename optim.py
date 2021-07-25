import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import scipy


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
        self.log['f'].append(np.linalg.norm(f(x)**2))
        

class BetaQuadratic:
    def __init__(self,r=1,sigma=1,n=600,a=1/2,b=1):
        evs=4*scipy.stats.beta(a=a,b=b).rvs(size=n)
        u=scipy.stats.ortho_group.rvs(n)
        
        self.A=u@np.diag(evs)@u.T
        self.x0=np.random.normal(size=(n,1))
        self.L=eigh(self.A,eigvals_only=True,subset_by_index=[n-1,n-1])
        
        
    def plot(self):
        ev,_=np.linalg.eigh(self.A)
        plt.hist(ev[ev<100],bins=30)
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
        self.L=eigh(self.A,eigvals_only=True,subset_by_index=[m-1,m-1])
        
        
    def plot(self):
        ev,_=np.linalg.eigh(self.A)
        plt.hist(ev[ev<100],bins=30)
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
        plt.hist(ev[ev<100],bins=30)
        plt.show()
    
    def __call__(self,x):
        return 1/2*np.linalg.norm(self.X@x-self.y)**2
    
    def grad(self,x):
        return self.X.T@(self.X@x-self.y)

    
        
def nesterov(f,niter=200,L=4):
    x=0
    y=f.x0
    alpha=1/L
    log=Log()
    log(f,y)
    for k in tqdm(range(1,niter+1)):
        beta=k/(k+3)
        x1=y-alpha*f.grad(y)
        y=x1+beta*(x1-x)
        x=x1
        #x=x-alpha*f.grad(x)
        ##log
        log(f,x)
        
    return log,x

def gd(f,niter=200,L=4):
    x=f.x0
    alpha=1/L
    log=Log()
    log(f,x)
    for k in tqdm(range(1,niter+1)):
        x=x-alpha*f.grad(x)
        log(f,x)
        
    return log,x

def cg(f,niter=200,L=None):
    ## as in https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf p.32
    x=f.x0
    d=r=-f.grad(x)
    A=f.A
    log=Log()
    log(f,x)
    for i in range(1,niter+1):
        alpha=r.T@r/(d.T@A@d)
        x=x+alpha*d
        
        r1=-f.grad(x)
        beta=r1.T@r1/(r.T@r)
        r=r1
        
        d=r+beta*d
        
        log(f,x)
    
    return log,x

