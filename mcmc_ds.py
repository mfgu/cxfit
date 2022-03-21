from pylab import *
from scipy import interpolate, special
"""
usage: python mcmc.py
"""
def voigt(alpha, x):
    v = x/1.41421
    a=zeros(8)
    b=zeros(8)
    c=zeros(8)
    
    a[1]=122.607931777104326
    a[2]=214.382388694706425
    a[3]=181.928533092181549
    a[4]=93.155580458134410
    a[5]=30.180142196210589
    a[6]=5.912626209773153
    a[7]=0.564189583562615
    
    b[1]=122.607931773875350
    b[2]=352.730625110963558
    b[3]=457.334478783897737
    b[4]=348.703917719495792
    b[5]=170.354001821091472
    b[6]=53.992906912940207
    b[7]=10.479857114260399
    
    c[1]=0.5641641
    c[2]=0.8718681
    c[3]=1.474395
    c[4]=-19.57862
    c[5]=802.4513
    c[6]=-4850.316
    c[7]=8031.468
    
    n = len(v)
    H = zeros(n)
    vb = 2.5
    if (alpha <= .001):
        w = where(abs(v) >= vb)[0]
        if (len(w) > 0):
            v2   = v[w]* v[w]
            v3   = 1.0
            fac1 = c[1]
            fac2 = c[1] * (v2 - 1.0)
            for i in range(1,8):
                v3     = v3 * v2
                fac1 = fac1 + c[i] / v3
                fac2 = fac2 + c[i] / v3 * (v2 - i)
                
            H[w] = exp(-v2)*(1. + fac2*alpha**2 * (1. - 2.*v2)) + fac1 * (alpha/v2);
        w = where(abs(v) < vb)
    else:
        w = arange(0,n)
        
    if (len(w) > 0):
        p1 = alpha
        vw = v[w]
        o1 = -vw
        p2 = (p1 * alpha + o1 * vw)
        o2 = (o1 * alpha - p1 * vw)
        p3 = (p2 * alpha + o2 * vw)
        o3 = (o2 * alpha - p2 * vw)
        p4 = (p3 * alpha + o3 * vw)
        o4 = (o3 * alpha - p3 * vw)
        p5 = (p4 * alpha + o4 * vw)
        o5 = (o4 * alpha - p4 * vw)
        p6 = (p5 * alpha + o5 * vw)
        o6 = (o5 * alpha - p5 * vw)
        p7 = (p6 * alpha + o6 * vw)
        o7 = (o6 * alpha - p6 * vw)

        q1 = a[1] + p1 * a[2] + p2 * a[3] + p3 * a[4] + p4 * a[5] + p5 * a[6] + p6 * a[7];
        r1 =        o1 * a[2] + o2 * a[3] + o3 * a[4] + o4 * a[5] + o5 * a[6] + o6 * a[7];
        q2 = b[1] + p1 * b[2] + p2 * b[3] + p3 * b[4] +  p4 * b[5] + p5 * b[6] + p6 * b[7] + p7;
        r2 =        o1 * b[2] + o2 * b[3] + o3 * b[4] + o4 * b[5] + o5 * b[6] + o6 * b[7] + o7;

        H[w] = (q1 * q2 + r1 * r2) / (q2 * q2 + r2 * r2);
        
    return H;

def vprof(xlo, xhi, c, s, a, n):
    x = arange(-20.0, 1e-5, 0.01)
    y = voigt(a, x)
    y = cumsum(y)
    y = log(y*0.5/y[-1])
    fxy = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0],y[-1]))
    def cprof(xx):
        w = xx<=0
        w0 = where(w)[0]
        yy = zeros(len(xx))
        if len(w0) > 0:
            yy[w0] = exp(fxy(xx[w0]))
        w1 = where(logical_not(w))[0]
        if len(w1) > 0:
            yy[w1] = 1-exp(fxy(-xx[w1]))
        return yy
    return n*(cprof((xhi-c)/s) - cprof((xlo-c)/s))

#this computes the -ln(likelyhood)
def mlnlikely(p, d):
    xlo = d['xlo']
    xhi = d['xhi']
    nx = len(xlo)
    y0 = d['y0']
    y1 = d['y1']
    lny0 = d['lny0']
    lny1 = d['lny1']
    yb = d['yb']
    np = d['np']
    a = exp(p[0])
    s0 = exp(p[1])
    s1 = exp(p[2])
    c = p[3:np+3]
    n0 = exp(p[np+3:2*np+3])
    n1 = exp(p[2*np+3:3*np+3])
    ym0 = zeros(nx)
    ym1 = zeros(nx)
    for i in range(np):
        ym0 += vprof(xlo, xhi, c[i], s0, a, n0[i])
        ym1 += vprof(xlo, xhi, c[i], s1, a, n1[i])
    ym0 += yb
    ym1 += yb
    d['ym0'] = ym0
    d['ym1'] = ym1
    return -sum(y0*log(ym0)-ym0-lny0)-sum(y1*log(ym1)-ym1-lny1)    

def mcmc(d, p0, s0, nm):
    np = len(p0)
    nq = np+2
    #first dim of p is parameter index, 2nd dim is the chain iteration index
    #p[0:np,i] store the np parameters of i-th iteration
    #p[np,i] store the -ln(likelyhood) of i-th iteration
    #p[np+1,i] store the rejection probability i-th iteration
    p = zeros((nq,nm))
    #copy the inital parameter into p[:np,0]
    for ip in range(np):
        p[ip,0] = p0[ip]
    #compute the -ln(likelyhood) for the initial parameters
    p[np,0] = mlnlikely(p0, d)
    #iterate along the chain
    for i in range(1,nm):
        #random jump from i-1 iteration to new parameters pnew
        #x is the np random variables uniformly distributed from -1 to 1
        x = 2*random(np)-1.0
        pnew = zeros(np)
        for ip in range(np):
            pnew[ip] = p[ip,i-1] + x[ip]*s0[ip]
        #compute the -ln(likelyhood) for the new parameters, store in p[np,i]
        p[np,i] = mlnlikely(pnew, d)
        if p[np,i] <= p[np,i-1]:
            #if p[np,i] <= p[np,i-1], the new parameters are accepted
            for ip in range(np):
                p[ip,i] = pnew[ip]
        else:
            #if p[np,i] > p[np,i-1], the new parameters are accepted with probability of r = exp(-(p[np,i]-p[np,i-1]))
            r = exp(-(p[np,i]-p[np,i-1]))
            #1-r is the rejection probability, store in p[np+1,i]
            p[np+1,i] = 1-r
            y = random()
            if (y > r):
                #reject pnew, copy p[:,i-1] into p[:,i]
                for ip in range(np+1):
                    p[ip,i] = p[ip,i-1]
            else:
                #accept pnew, copy pnew into p[:,i]
                for ip in range(np):
                    p[ip,i] = pnew[ip]
                
        if (i+1)%100 == 0:
            print('%6d %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E'%(i,p[np+1,i],p[np,i],p[0,i],p[1,i],p[2,i],mean(p[np+1,:i+1])))
    return p

def run_mcmc(nm, df0, df1):
    d0 = loadtxt(df0, unpack=1)
    d1 = loadtxt(df1, unpack=1)
    w = where((d0[0]>6.6e3)&(d0[0]<7e3))
    w = w[0]
    xlo = d0[0][w]
    xhi = d0[0][w+1]
    yb = 0.05
    y0 = d0[1][w]+yb
    y1 = d1[1][w]+yb
    
    lny0 = special.gammaln(y0+1.0)
    lny0 -= log(sqrt(2*pi*y0))
    lny1 = special.gammaln(y1+1.0)
    lny1 -= log(sqrt(2*pi*y1))
    np = 6
    npar = 3 + np*3
    p = zeros(npar)
    s = zeros(npar)
    p[0] = log(0.1)
    p[1] = log(2.5)
    p[2] = log(2.5)
    p[3:3+np] = [6.635e3,6.666e3,6.6815e3,6.6985e3,6.9515e3,6.9735e3]
    p[3+np:3+2*np] = log(array([500.0,300.0,300.0,250.0,150.0,200.0]))
    p[3+2*np:3+3*np] = log(array([500.0,300.0,300.0,250.0,150.0,200.0]))
    s[0] = 0.5
    s[1] = 0.1
    s[2] = 0.1
    s[3:3+np] = 0.05
    s[3+np:3+3*np] = 0.1
    d = {'xlo':xlo, 'xhi':xhi, 'y0':y0, 'y1':y1, 'lny0':lny0, 'lny1':lny1, 'np':np, 'yb':yb}
    mp = mcmc(d, p, s, nm)
    return d,mp
