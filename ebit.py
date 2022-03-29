from pfac import fac, crm, const
from scipy import integrate, interpolate, special
from textio import *

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

def convd(xd, yd, s, lw=None, x0=None, x1=None):
    dx = s/3.0
    xd = array(xd)
    yd = array(yd)
    if x0 is None:
        x0 = min(xd)-20.0*s
    if x1 is None:
        x1 = max(xd)+20.0*s
    x = arange(x0, x1, dx)
    y = zeros(len(x))
    p = 1.0/sqrt(2*pi)/s
    if lw is None:
        for i in range(len(xd)):
            t = (x-xd[i])/s
            w = nonzero((t > -20.0) & (t < 20.0))[0]
            if (len(w) > 0):
                y += p*yd[i]*exp(-0.5*t*t)
    else:
        for i in range(len(xd)):
            t = (x-xd[i])/s
            a = lw[i]/(1.414*s)
            y += p*yd[i]*voigt(a, t)
            
    return (x, y)

def e2v(e, m=0):
    if (m == 0):
        k = e/const.Hartree_eV;
        k = 2.0*k*(1.0 + 0.5*const.Alpha**2*k);
        k = const.Alpha**2*k;
        k = sqrt(k/(1.0+k));
        k /= const.Alpha;
        k *= const.RBohr*1e-8*const.Rate_AU*1E-10;
    else:
        m = const.Mp_keV*m/const.Me_keV;
        k = e/const.Hartree_eV;
        k = 2.0*m*k*(1.0+0.5*const.Alpha**2*k/m);
        k = const.Alpha**2*k;
        k = sqrt(k/(m*m+k));
        k /= const.Alpha;
        k *= const.RBohr*1e-8*const.Rate_AU*1E-10;  

    return k

def sawtooth(nt, emin, emax, e0, x):
    t = linspace(0.0, 1.0, nt)
    w0 = where(t <= x)
    w1 = where(t > x)
    e = zeros(nt)
    if (len(w0[0]) > 0):
        e[w0] = t[w0]*(e0-emin)/x + emin
    if (len(w1[0]) > 0):
        e[w1] = emax

    return e

def intmaxwell(x):
    return special.erf(x) - 2*x*exp(-x*x)/sqrt(pi)

def maxwell(nt, t0, emin0, emax0, t1, emin1, emax1, x):
    t = linspace(0.0, 1.0, nt)
    e0 = linspace(emin0, emax0, nt)
    e1 = linspace(emin1, emax1, nt)
    x0 = sqrt(e0/t0)
    x1 = sqrt(e1/t1)
    t0 = intmaxwell(x0)
    t1 = intmaxwell(x1)
    t0 = (t0-t0[0])/(t0[-1]-t0[0])
    t1 = (t1-t1[0])/(t1[-1]-t1[0])
    i0 = interpolate.interp1d(t0, e0)
    i1 = interpolate.interp1d(t1, e1)
    w0 = nonzero(t <= x)
    w1 = nonzero(t > x)
    e = zeros(nt)
    if (len(w0[0]) > 0):
        e[w0] = i0(t[w0]/x)
    if (len(w1[0]) > 0):
        e[w1] = i1((t[w1]-x)/(1.0-x))
    return e

def cirate(z, k, e):
    n = len(e)
    r = zeros(n)
    for i in range(n):
        r[i] = crm.CColFit(z, k, e[i])[0]*e2v(e[i])
    return r

def rrrate(z, k, e):
    e = array(e)
    if k <= 3:
        n0 = 1
        v = k
    elif k <= 10:
        n0 = 2
        v = k-2
    elif k <= 28:
        n0 = 3
        v = k-10
    w = double(v)/(2.0*n0*n0)
    n0 = n0 + w - 0.3
    ze = 0.5*(z + z-k)
    x = 2*ze*ze*const.Ryd_eV/e
    r = 8*pi*const.Alpha*(0.3861**2)*x/(3.0*sqrt(3.0)) * log(1+x/(2*n0*n0))
    r = r * e2v(e)

    return r

class rates:
    def __init__(self):
        self.z = None
        self.nele = None
        self.e = None
        self.rts = None
        
def readrate(r0):
    if (type(r0) == type('')):
        r = readcol(r0, range(4), format=['I','I','F','F'])
        a = rates()
        a.z = r[0]
        a.nele = r[1]
        a.e = r[2]
        a.rts = r[3]
        return a
    else:
        return r0

def interprate(z, k, e, r, t):
    n = len(e)
    if (r == None):
        if (t == 'rr'):
            return rrrate(z, k, e)
        elif (t == 'ci'):
            return cirate(z, k, e)
        elif (t == 'dr'):
            return 0.0
    else:
        a = zeros(n)
        if (r.e == None):
            a[:] = r.rts
            return a
        if (r.z != None):
            w = where((r.z == z) & (r.nele == k))[0]
        else:
            w = where(r.nele == k)[0]
        if (len(w) > 1):
            ri = interpolate.interp1d(r.e[w], r.rts[w])
            s = where((e >= r.e[w[0]]) & (e <= r.e[w[-1]]))[0]
            if (len(s) > 0):
                a[s] = ri(e[s])
        else:
            if (t == 'rr'):
                a = rrrate(z, k, e)
            elif (t == 'ci'):
                a = cirate(z, k, e)
        return a    
    
def chex(z, a):
    n = z + 1
    c = matrix(zeros((n,n)))
    for i in range(z):
        c[i+1,i] = a * (double(z-i)/double(z))**1.17
    return c

def crderiv(y, t, tp, ri, cx, rloss):
    n = ri.shape
    x = t - 2*tp*floor(t/(2*tp))
    if x < tp:
        x = x/tp
        i = floor(x*n[0])
    else:
        x = (x-tp)/tp
        i = n[0]-1 - floor(x*n[0])

    r = ri[i,:,:]
    r = matrix(r.reshape((n[1],n[2])))
    r = r + cx
    tr = sum(r, 0)
    tr = diag(array(tr).flatten()+rloss)
    r -= tr
    
    return array(r*y.reshape((n[1],1))).flatten()

def avgderiv(y, t, r):
    n = r.shape
    return array(r*y.reshape((n[0],1))).flatten()

class ionev:
    def __init__(self):
        self.tgrid = None
        self.sweep = None
 
    def setsweep(self, sweep=None,
                 nsweep=2048, tsweep=0.01, args=()):
        self.sweep = sweep
        self.nsweep = nsweep
        self.tsweep = tsweep
        self.sweepargs = args
        if (sweep != None):
            self.esweep = sweep(nsweep, *args)
        else:
            self.esweep = array([emin])
            self.nsweep = 1
        
    def setri(self, z, kmin=0, kmax=0, rr=None, dr=None, ci=None):
        self.z = z
        self.kmin = kmin
        if kmax > 0:
            self.kmax = kmax
        else:
            self.kmax = z

        rr = readrate(rr)
        dr = readrate(dr)
        ci = readrate(ci)

        e = self.esweep
        nt = len(e)
        n = self.kmax - self.kmin + 1
        c = zeros((nt,n,n))            
        for i in range(n):
            ip = i + 1
            im = i - 1
            if (ip <= z):
                c[:,ip,i] = c[:,ip,i] + interprate(z, i, e, dr, 'dr')
                c[:,ip,i] = c[:,ip,i] + interprate(z, i, e, rr, 'rr')
            if im >= 0:
                c[:,im,i] = c[:,im,i] + interprate(z, i, e, ci, 'ci')
        self.rimat = c
        self.ria = sum(self.rimat, 0)/self.nsweep
        self.rcx = chex(z, 1.0)

    def solveavg(self, tgrid=None, ne=1e2, cex=0.01, rloss=0.0, xci=1.0):
        self.ne = ne
        self.cex = cex
        self.rloss = rloss
        self.xci = xci
        
        r = self.ria * ne + self.rcx*cex*ne
        r[0,1] = r[0,1] * xci
        r[1,2] = r[1,2] * xci
        tr = sum(r, 0)
        tr = diag(array(tr).flatten()+rloss)
        r -= tr
         
        n = self.kmax - self.kmin + 1
        p0 = zeros(n)
        p0[-1] = 1.0

        if (tgrid != None):
            self.tgrid = tgrid
        if (self.tgrid == None):
            self.tgrid = arange(0.0, 10.0, 0.01)
            
        self.pop = integrate.odeint(avgderiv, p0, self.tgrid,
                                    args=(r,), rtol=1e-3, atol=1e-5,
                                    hmax=0.01, mxstep=1000000)        
        
    def solvecr(self, tgrid=None, ne=1e2, cex=0.01, rloss=0.0):

        self.ne = ne
        self.cex = cex
        self.rloss = rloss
        
        n = self.kmax - self.kmin + 1
        p0 = zeros(n)
        p0[-1] = 1.0
        ri = self.rimat * ne
        cx = self.rcx*cex*ne

        if (tgrid != None):
            self.tgrid = tgrid
        if (self.tgrid == None):
            self.tgrid = arange(0.0, 10.0, self.tsweep/20.0)
            
        self.pop = integrate.odeint(crderiv, p0, self.tgrid,
                                    args=(self.tsweep, ri, cx, rloss),
                                    rtol=1e-3, atol=1e-5,
                                    hmax=self.tsweep*0.2,
                                    mxstep=100000)

    def savepop(self, fn):
        f = open(fn, 'w')
        n = self.z + 1
        nt = len(self.tgrid)
        for i in range(n):
            for j in range(nt):
                f.write('%3d %11.4E %11.4E\n'%(i, self.tgrid[j], self.pop[j,i]))
        f.close()
    
        
