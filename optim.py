import numpy as np
from utils import Log
import math

def fom(f,gen,niter=200):
    x1=x=f.x0
    log=Log()
    log(f,x)
     
    for _ in range(niter):
        a,b,_=next(gen)
        aux=x
        x=x+(a-1)*(x-x1)+b*f.grad(x)
        x1=aux
        
        log(f,x)
        
    return log,x

def jacobi_basegen(alpha,beta):
    
    yield (alpha-beta)/2,(alpha+beta+2)/2,0
    
    n=2
    while True:
        
        at=(alpha**2-beta**2)*(2*n+alpha+beta-1)/ \
            (2*(n)*(n+alpha+beta)*(2*n+alpha+beta-2))
        bt=(2*n+alpha+beta-1)*(2*n+alpha+beta)/ \
            (2*(n)*(n+alpha+beta))
        ct=-(n+alpha-1)*(n+beta-1)*(2*n+alpha+beta)/ \
            ((n)*(n+alpha+beta)*(2*n+alpha+beta-2))
        
        
        yield at,bt,ct
        n+=1
def laguerre_basegen(alpha):
    n=1
    while True:
        bt=-1/(n+1)
        at=(2*n+alpha+1)/(n+1)
        ct=-(n+alpha)/(n+1)
        
        yield at,bt,ct
        n+=1
        
def residual_wrapgen(gen):
    delta=np.array([0],dtype=np.longdouble)
    while True:
        alpha,beta,gamma=next(gen)
        delta=1/(alpha+gamma*delta)
        yield alpha*delta,delta*beta,1-alpha*delta
        
def shift_wrapgen(gen,a=2,b=2):
    while True:
        alpha,beta,gamma=next(gen)
        alpha=alpha+b*beta
        beta=a*beta

        yield alpha,beta,gamma
    
def jacobi_momentum(f,alpha=1/2,beta=5/2,niter=200,L=None):
    if hasattr(f,'L'):
        L=f.L
    if not L:
        raise Error()
    gen=residual_wrapgen(shift_wrapgen(jacobi_basegen(alpha,beta),2/f.L,-1))
    return fom(f,gen,niter)
def jm_decorator(alpha,beta):
    def  aux(*args,**kwargs):
        return jacobi_momentum(*args,alpha=alpha,beta=beta,**kwargs)
    return aux


def gd(f,niter=200,L=None):
    
    def gen():
        while True:
            yield 1,-1/f.L,0
    return fom(f,gen(),niter)

def polyak(f,niter=200):
    l,L=f.l,f.L
    m=(math.sqrt(L)-math.sqrt(l))/(math.sqrt(L)+math.sqrt(l))
    bt=-4/(math.sqrt(L)+math.sqrt(l))**2
    
    def gen():
        while True:
            yield m+1,bt,0
            
    return fom(f,gen(),niter)


        
def nesterov(f,niter=200,L=4):
    x=0
    y=f.x0
    alpha=1/L
    log=Log()
    log(f,y)
    for k in range(1,niter+1):
        beta=k/(k+3)
        x1=y-alpha*f.grad(y)
        y=x1+beta*(x1-x)
        x=x1
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

