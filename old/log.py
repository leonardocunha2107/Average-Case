def mp_momentum(f,niter=200,sigma=1,r=1):
    x1=x=f.x0
    delta=0
    rho=(1+r)/np.sqrt(r)
    log=Log()
    log(f,x)
    for t in range(1,niter+1):
        delta=-1/(rho+delta)
        
        aux=x
        x=x+(1+rho*delta)*(x1-x)+delta/(np.sqrt(r)*sigma**2)*f.grad(x)
        x1=aux
        
        log(f,x)        
    return log,x


def jacobi_momentum(f,niter=200,b=2,L=4):
    x1=x=f.x0
    delta=0
    log=Log()
    log(f,x)
    for t in tqdm(range(1,niter+1)):
        t2=t**2
        alpha=-(b+2*t)*(2*b**2+4*b*t+b+4*t2-1)
        alpha/=2*t*(b+t+1)*(b+2*t-1)
        
        beta=(b+2*t)*(b+2*t+1)/(L*t*(b+t+1))
        
        gamma=-(t-1/2)*(b+t-1/2)*(b+2*t+1)
        gamma/=t*(b+t+1)*(b+2*t-1)
        
        delta=1/(alpha+gamma*delta)
        
        a,bt=delta*alpha,delta*beta
        
        aux=x
        x=x+(1-a)*(x1-x)+bt*f.grad(x)
        x1=aux
        
        log(f,x)        
    return log,x