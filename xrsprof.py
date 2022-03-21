from pylab import *
from scipy.stats import norm
from scipy.optimize import curve_fit
import mcmc

npix = 32
ws = 0.15
ps = 0.25
sig = 2.8

isig = norm.rvs(loc=sig, scale=sig*ws, size=npix)
ipos = norm.rvs(loc=0.0, scale=ps, size=npix)

x = arange(-25.0,25.01, 0.05)
nx = len(x)
y = zeros(nx)
y0 = norm.pdf(x, loc=0.0, scale=sig)
for i in range(npix):
    y += norm.pdf(x, loc=ipos[i], scale=isig[i])

ma = 5e4/max(y)
mb = 10.0
y = ma*y + 10.0
y0 = ma*y0 + 10.0
ye = sqrt(y)
yd = normal(loc=y, scale=ye)

def xpf(x,p):
    s0 = p[0]
    s1 = p[1]
    a = p[2]
    y0 = (1+p[5]*x)*norm.pdf(x, loc=0.0, scale=s0)
    y1 = (1+p[5]*x)*norm.pdf(x, loc=0.0, scale=s0*s1)
    yd = y0*(1-a)+y1*a
    yd = yd*p[3]+p[4]
    return yd

#p,c = curve_fit(xpf, x, y, bounds=([0.5*sig,1.0,0.0],[1.5*sig,2.0,1.0]))
d = {'xd':x,
     'yd':yd,
     'ye':ye,
     'ftp':0,
     'mp':[sig,1.5,0.5,5e4,10.0,0.0],
     'mp0':[0.1*sig,1.0,0.0,1e3,1.0,-1.0],
     'mp1':[5*sig,2.0,2.0,1e6,100.0,1.0],
     'smp':[0.1*sig,0.1,0.1,100.0,1.0,0.1]}
     
mcmc.mcmc(xpf, d, 5000)
p = d['mpa']
ym = xpf(x, p)

    
