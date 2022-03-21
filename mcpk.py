from pylab import *
from scipy import integrate, interpolate, special, stats, linalg
import pickle
import time, os, sys
from collections import namedtuple

SpecType = namedtuple('SpecType',
                      ['xlo', 'xhi', 'yd', 'eff', 'lnyd', 'yb',
                       'ym', 'inp', 'tr'])

def read_spec(fn, emin, emax, yb, feff=''):
    r = loadtxt(fn, unpack=1)
    w = where((r[0]>=emin)&(r[0]<=emax))
    w = w[0]
    xlo = r[0][w]
    xhi = append(xlo[1:],2*xlo[-1]-xlo[-2])
    yd = r[1][w] + yb
    nx = len(yd)
    eff = zeros(nx)
    eff[:] = 1.0
    if feff != '':
        e = loadtxt(feff, unpack=1)
        fie = interpolate.interp1d(e[0], e[1], bounds_error=False,
                                   fill_value='extrapolate')
        eff = fie(0.5*(xlo+xhi))
        
    lnyd = special.gammaln(yd+1.0)-log(sqrt(2*pi*yd))
    s = SpecType(xlo, xhi, yd, eff, lnyd, yb,
                 ['ymt', 'ymi', 'ymb', 'xb', 'xw'],
                 ['ip0', 'np', 'nt', 'ni', 'nb', 'ib', 'nw', 'iw'],
                 ['ke','i0','i1','tp','e','s','w'])    
    return s

def cc_unresolved(p, nt, ni, ifx, ifp, fwhm):
    np = len(p)
    cc = zeros((np,4))
    iw = arange(0, nt*ni, dtype=int32)
    ix = iw[::ni]
    xc = p[ix]
    for i in range(nt):
        ip = ix[i]
        w = where((abs(xc-xc[i]) < fwhm/2)&(ifx[ix] < 0))
        w = w[0]
        for j in w:
            jp = ix[j]
            if jp == ip:
                continue
            if ifx[jp] >= 0:
                continue
            ifx[jp] = ip+ifp
            while(ifx[ifx[jp]-ifp] >= 0):
                ifx[jp] = ifx[ifx[jp]-ifp]
                if ifx[jp] == jp+ifp:
                    ifx[jp] = -1
                    break
            if ifx[jp] < 0:
                continue
            cc[jp,0] = 1.0
            cc[jp,1] = p[jp] - p[ifx[jp]-ifp]
            cc[jp,2] = 1.0
            cc[jp,3] = 1.0
    return cc

def pick_lines(x, y, smin, fwhm):
    n = len(x)
    iz = repeat(False, n)
    for i in range(n):
        e = x[i]
        k = where(abs(x-e) <= fwhm)
        k = k[0]
        if ((y[i] > smin*nanmax(y[k])).any()):
            iz[i] = True
    return (where(iz))[0]

def init_params(s, lns, smin, fwhm, dx, ifp, xb, xw, nwp=2):
    e0 = s.xlo[0]
    e1 = s.xhi[-1]
    nf = len(lns)
    xs = []
    ys = []
    ds = []
    kes = []
    i0s = []
    i1s = []
    tps = []
    ni = 2
    if type(xb) == int:
        nb = xb
        if nb == 1:
            xb = [0.5*(s.xlo[0]+s.xhi[-1])]
        else:
            db = (s.xhi[-1]-s.xlo[0])/(nb-1)
            xb = arange(s.xlo[0], s.xhi[-1]+db*0.1, db)
    else:
        nb = len(xb)
    if type(xw) == int:
        nw = xw
        if nw == 1:
            xw = [0.5*(s.xlo[0]+s.xhi[-1])]
        else:
            dw = (s.xhi[-1]-s.xlo[0])/(nw-1)
            xw = arange(s.xlo[0], s.xhi[-1]+dw*0.1, dw)
    else:
        nw = len(xw)
    nt = 0
    for i in range(nf):
        r = loadtxt(lns[i], unpack=1)
        w = where((r[4]>=e0)&(r[4]<=e1))
        ke = int32(r[0][w])
        i0 = int32(r[1][w])
        i1 = int32(r[2][w])
        tp = int32(r[3][w])
        x = r[4][w]
        y = r[6][w]
        d = r[7][w]/2
        m = argmax(y)
        w = where((s.xlo >= x[m]-fwhm)&(s.xhi <= x[m]+fwhm))
        w = w[0]
        ym = sum(s.yd[w])
        sm = y[m]
        k = pick_lines(x, y, smin, fwhm)
        x = x[k]
        y = y[k]*ym/sm
        d = d[k]
        ke = ke[k]
        i0 = i0[k]
        i1 = i1[k]
        tp = tp[k]
        xs.append(x)
        ys.append(y)
        ds.append(d)
        kes.append(ke)
        i0s.append(i0)
        i1s.append(i1)
        tps.append(tp)
        nt += len(x)
    
    ib = nt*ni
    iw = ib+nb
    np = iw+nw*nwp
    p = zeros(np)
    p0 = zeros(np)
    p1 = zeros(np)
    sig = zeros(np)
    ifx = repeat(-1, np)
    cc = zeros((np,4))
    tr = zeros(nt)
    ax = zeros(nt)
    ay = zeros(nt)
    ake = zeros(nt, dtype=int32)
    ai0 = zeros(nt, dtype=int32)
    ai1 = zeros(nt, dtype=int32)
    atp = zeros(nt, dtype=int32)
    ip0 = 0
    it0 = 0
    for i in range(nf):
        nti = len(xs[i])
        npi = nti*ni
        ii = arange(ip0, ip0+npi, dtype=int32)
        ix = ii[::ni]
        p[ix] = xs[i]
        p0[ix] = p[ix]-dx
        p1[ix] = p[ix]+dx
        sig[ix] = dx*0.1
        ix = ii[1::ni]
        p[ix] = ys[i]
        p0[ix] = 1e-5*p[ix]
        p1[ix] = 1e5*p[ix]
        sig[ix] = sqrt(p[ix])*0.5
        tr[it0:it0+nti] = ds[i]
        ax[it0:it0+nti] = xs[i]
        ay[it0:it0+nti] = ys[i]
        ake[it0:it0+nti] = kes[i]
        ai0[it0:it0+nti] = i0s[i]
        ai1[it0:it0+nti] = i1s[i]
        atp[it0:it0+nti] = tps[i]
        ip0 += npi
        it0 += nti
    s.inp[0] = ifp
    s.inp[1] = np
    s.inp[2] = nt
    s.inp[3] = ni
    s.inp[4] = nb
    s.inp[5] = ifp+ib
    s.inp[6] = nw
    s.inp[7] = ifp+iw
    w = where(s.yd>=1)
    sy = sort(s.yd[w])
    bk = 0.5*sy[int(0.2*len(sy))]
    p[ib:iw] = bk
    p0[ib:iw] = p[ib:iw]*1e-5
    p1[ib:iw] = p[ib:iw]*1e10
    sig[ib:iw] = p[ib:iw]*0.1
    iw1 = iw+nw
    iw2 = iw1+nw
    if nwp >= 4:
        iw3 = iw2+nw
        iw4 = iw3+nw
        if nwp == 5:
            iw5 = iw4+nw
    p[iw:iw1] = 0.1*fwhm
    p0[iw:iw1] = 0.0
    p1[iw:iw1] = fwhm
    sig[iw:iw1] = p[iw:iw1]*0.1
    p[iw1:iw2] = fwhm/2.35
    p0[iw1:iw2] = p[iw1:iw2]*1e-2
    p1[iw1:iw2] = p[iw1:iw2]*1e2
    sig[iw1:iw2] = p[iw1:iw2]*0.1
    if nwp >= 4:
        p[iw2:iw3] = 1.3
        p0[iw2:iw3] = 1.0
        p1[iw2:iw3] = 2.0
        sig[iw2:iw3] = 0.05
        p[iw3:iw4] = 0.15
        p0[iw3:iw4] = 0.0
        p1[iw3:iw4] = 0.5
        sig[iw3:iw4] = 0.05
        if nwp == 5:
            p[iw4:iw5] = 0.0
            p0[iw4:iw5] = -0.25
            p1[iw4:iw5] = 0.25
            sig[iw4:iw5] = 0.01
    s.tr[0] = ake
    s.tr[1] = ai0
    s.tr[2] = ai1
    s.tr[3] = atp
    s.tr[4] = ax
    s.tr[5] = ay
    s.tr[6] = tr
    s.ym[1] = zeros((np,len(s.xlo)))
    s.ym[3] = xb.copy()
    s.ym[4] = xw.copy()
    
    cc = cc_unresolved(p, nt, ni, ifx, ifp, fwhm)
    ii = arange(np, dtype=int32)+ifp
    w = where(ifx == ii)
    w = w[0]
    if len(w) > 0:
        ifx[w] = -1
    return p,p0,p1,sig,ifx,cc
                
def voigt(alpha, x):
    v = x/1.414213562373
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

def vprof(xlo, xhi, c, s, a):
    x = arange(-35.0, 1e-5, 0.025)
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
    return (cprof((xhi-c)/s) - cprof((xlo-c)/s))

#this computes the -ln(likelyhood)
def lnlikely(d, ip):
    s = d['sp']
    nd = len(s)
    id = d['id']
    ifx = d['ifx']
    cc = d['cc']
    lnlk = d['lnlk']
    p = d['mp']
    nwp = d['nwp']
    def update_line(k, j):
        tr = s[k].tr[6]
        ni = s[k].inp[3]
        ip0 = s[k].inp[0]
        nw = s[k].inp[6]
        iw = s[k].inp[7]
        iw1 = iw + nw
        iw2 = iw1 + nw
        if nwp >= 4:
            iw3 = iw2 + nw
            iw4 = iw3 + nw
            if nwp == 5:
                iw5 = iw4 + nw
        i0 = ip0+j*ni
        i1 = i0+1
        if nw == 1:
            a = p[iw]
            w = p[iw1]
            if nwp >= 4:
                ws = p[iw2]
                b = p[iw3]
                if nwp == 5:
                    c = p[iw4]
        else:            
            fia = interpolate.interp1d(sk.ym[4], p[iw:iw1],
                                       bounds_error=False,
                                       fill_value='extrapolate')
            fiw = interpolate.interp1d(sk.ym[4], p[iw1:iw2],
                                       bounds_error = False,
                                       fill_value='extrapolate')
            if nwp >= 4:
                fis = interpolate.interp1d(sk.ym[4], p[iw2:iw3],
                                           bounds_error = False,
                                           fill_value='extrapolate')
                fib = interpolate.interp1d(sk.ym[4], p[iw3:iw4],
                                           bounds_error = False,
                                           fill_value='extrapolate')
                if nwp == 5:
                    fic = interpolate.interp1d(sk.ym[4], p[iw4:iw5],
                                               bounds_error = False,
                                               fill_value='extrapolate')
            a = fia(p[i0])
            w = fiw(p[i0])
            if nwp >= 4:
                ws = fis(p[i0])
                b = fib(p[i0])
                if nwp == 5:
                    c = fic(p[i0])
        a += tr[j]
        a /= 1.414213562373*w
        v = vprof(s[k].xlo, s[k].xhi, p[i0], w, a)
        if nwp >= 4:
            v2 = vprof(s[k].xlo, s[k].xhi, p[i0], w*ws, a/ws)
            v = (1-b)*v + b*v2
            if nwp == 5:
                xw = (0.5*(s[k].xlo+s[k].xhi)-p[i0])/w
                xm = 1+c*xw*exp(-0.001*xw*xw)
                wm = where(xm < 0.01)
                xm[wm] = 0.01
                v *= xm
        s[k].ym[1][j] = p[i1]*v

    def update_bkgd(k):
        ib = s[k].inp[5]
        nb = s[k].inp[4]
        iw = s[k].inp[7]

        if nb == 1:
            s[k].ym[2] = repeat(p[ib], len(s[k].xlo))
        else:
            fib = interpolate.interp1d(s[k].ym[3], p[ib:iw],
                                       bounds_error=False,
                                       fill_value='extrapolate')
            s[k].ym[2] = fib(0.5*(s[k].xlo+s[k].xhi))
                         
    if ip >= 0:
        w = (where(ifx == ip))[0]
        iu = append(w, ip)        
        for i in iu:
            if ip >= 0 and i != ip:
                uc = cc[i]
                xp = p[ip]
                if uc[2]:
                    xp = xp**uc[2]
                if uc[0]:
                    xp = uc[0]*xp
                if uc[1]:
                    xp += uc[1]
                if uc[3]:
                    xp = xp**uc[3]
                p[i] = xp
            k = id[i]
            ib = s[k].inp[5]
            iw = s[k].inp[7]
            if i >= ib and i < iw:
                update_bkgd(k)
            elif i >= iw:
                nt = s[k].inp[2]
                for j in range(nt):
                    update_line(k, j)            
            else:
                ip0 = s[k].inp[0]
                ni = s[k].inp[3]
                j = int((i-ip0)/ni)
                update_line(k, j)
            s[k].ym[0] = sum(s[k].ym[1], 0)
            ym = s[k].eff*(s[k].ym[0]+s[k].ym[2])+s[k].yb
            lnlk[k] = sum(s[k].yd*log(ym)-ym-s[k].lnyd)
    else:
        for k in range(nd):
            nt = s[k].inp[2]
            for j in range(nt):
                update_line(k, j)
            update_bkgd(k)

            s[k].ym[0] = sum(s[k].ym[1], 0)
            ym = s[k].eff*(s[k].ym[0]+s[k].ym[2]) + s[k].yb
            lnlk[k] = sum(s[k].yd*log(ym)-ym-s[k].lnyd)
    return sum(lnlk)

def logqx(x):
    x = abs(x)
    b1 =  0.319381530
    b2 = -0.356563782
    b3 =  1.781477937
    b4 = -1.821255978
    b5 =  1.330274429
    t = 1.0/(1+0.2316419*x)
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    t5 = t4*t
    zx = -0.5*x*x + log(b1*t + b2*t2 + b3*t3 + b4*t4 + b5*t5) - 0.918938533205
    return zx

def mcmc(d, nm, sav=[], racc=0.4, npr=10):        
    xin = arange(-30.0, 0.05, 0.05)
    xin[-1] = 0.0
    yin = integrate.cumtrapz(stats.norm().pdf(xin), xin, initial=0.0)
    yin[0] = yin[1]*(yin[1]/yin[2])
    fin0 = interpolate.interp1d(xin, yin, kind='linear',
                                bounds_error=False, fill_value=(yin[0],0.5))
    fin1 = interpolate.interp1d(log(yin), xin, kind='linear',
                                bounds_error=False, fill_value=(-30.0,0.0))
    def cnorm(x):
        if (x > 0):
            if x > 30:
                z = logqx(x)
                if z < -300:
                    return 1.0
                else:
                    return 1-exp(z)
            return 1-fin0(-x)
        elif (x < 0):
            if x < -30:
                z = logqx(x)
                if z < -300:
                    return 0.0
                else:
                    return exp(z)
            return fin0(x)
        else:
            return 0.5
        
    def inorm(y):
        if y > 0.5:            
            return -fin1(log(1-y))
        elif y < 0.5:
            return fin1(log(y))
        else:
            return 0.0
           
    def rand_cg(x0, x1):
        r = rand()
        r0 = cnorm(x0)
        r1 = cnorm(x1)
        r = r0 + r*(r1-r0)
        return (inorm(r),r,r0,r1)

    mp0 = d['mp0']
    mp1 = d['mp1']
    smp = d['smp']
    mp = d['mp']
    id = d['id']
    ifx = d['ifx']
    s = d['sp']
    ni = s[0].inp[3]
    t0 = time.time()
    np = len(mp)
    frej = zeros((nm,np),dtype=int8)
    rrej = zeros((nm,np))    
    rrej[:,:] = -1.0
    fmp = zeros(np)
    fmp[:] = 1.0
    hmp = zeros((nm,np))
    ene = zeros(nm)
    hmp[0] = mp
    trej = 0.0
    ttr = 0.0
    r0 = lnlikely(d, -1)
    ene[0] = r0
    nburn = max(500,int32(0.25*nm))
    if len(sav) == 3:
        fsav = sav[0]
        nsav = sav[1]
        tsav = sav[2]
    if len(sav) == 2:
        fsav = sav[0]
        nsav = sav[1]
        tsav = fsav+'.trigger'
    elif len(sav) == 1:
        fsav = sav[0]
        nsav = 0
        tsav = fsav+'.trigger'
    else:
        fsav = None
        nsav = 0
        tsav = None

    for i in range(1, nm):
        i1 = i-1
        trej = 0.0
        nrej = 0
        for ip in range(np):
            if ifx[ip] >= 0:
                continue
            if mp1[ip] <= mp0[ip]:
                continue
            k = id[ip]
            j = int((ip - s[k].inp[0])/ni)
            xp0 = (mp0[ip]-hmp[i1,ip])/smp[ip]
            xp1 = (mp1[ip]-hmp[i1,ip])/smp[ip]
            rn,yp,y0,y1 = rand_cg(xp0, xp1)
            mp[ip] = hmp[i1,ip] + rn*smp[ip]
            yi0 = cnorm((mp0[ip]-mp[ip])/smp[ip])
            yi1 = cnorm((mp1[ip]-mp[ip])/smp[ip])
            r = lnlikely(d, ip)
            dr = r + log(y1-y0)-log(yi1-yi0)
            rej = 0
            rp = 0.0
            if (dr < r0):
                rp = 1-exp(dr-r0)
                if (rand() < rp):
                    mp[ip] = hmp[i1,ip]
                    r = lnlikely(d, ip)
                    rej = 1
            frej[i,ip] = rej
            if not rej:
                r0 = r
            rrej[i,ip] = rp
            trej += rp
            nrej += 1
        hmp[i] = mp
        ene[i] = r0
        if i >= 50 and i <= nburn and i%25 == 0:
            im = i-25
            im0 = max(i-100, 10)
            for ip in range(np):
                mr = mean(rrej[im:i+1,ip])
                xa = ((1-mr)/racc)**2
                fa = fmp[ip]*xa
                fa = min(fa, 1e2)
                fa = max(fa, 1e-2)
                fa = 0.25*fmp[ip]+0.75*fa
                xst = fa*std(hmp[im0:i+1,ip])
                if xst > 0:
                    smp[ip] = xst
                fmp[ip] = fa
        trej /= nrej
        ttr = (ttr*(i-1)+trej)/i
        if (i+1)%npr == 0:
            pp = [i, trej, ttr, r0, r0-ene[i1], rrej[i,0]]
            sx = ''
            for ix in range(len(s)):
                iw = s[ix].inp[7]
                iw1 = s[ix].inp[6]+iw
                sx += '%10.3E %10.3E'%(mp[iw],mp[iw1])
            pp.append(sx)
            pp.append(time.time()-t0)
            print('imc: %6d %10.3E %10.3E %10.3E %10.3E %10.3E %s %10.3E'%tuple(pp))
            sys.stdout.flush()

        savenow = False
        if i == nm-1:
            savenow = True
        elif nsav > 0 and (i+1)%nsav == 0:
            savenow = True
        elif tsav != None and os.path.isfile(tsav):
            savenow = True
            os.system('rm '+tsav)            
        if savenow:
            d['hmp'] = hmp
            d['fmp'] = fmp        
            d['smp'] = smp
            d['ene'] = ene
            d['frej'] = frej
            d['rrej'] = rrej    
            print('pickling: %s %10.3E'%(fsav, time.time()-t0))
            fs = open(fsav, 'wb')
            pickle.dump(d, fs)
            fs.close()

def load_pkl(fn):
    with open(fn, 'rb') as f:
        d = pickle.load(f)
        p = d['hmp']
        w=where(d['ene'] < 0)
        w = w[0]
        nw = len(w)
        i0 = min(1000,int(nw/2))
        d['mp'] = mean(p[i0:nw,],0)
        r = lnlikely(d, -1)
    return d
    
def run_mcpk(nm, dfs, feff='', emin=6.88e3, emax=7.05e3,
             lns=['spec1/Fe01a.ln'], sav='',
             smin=1e-2, fwhm=6.5, dx=1.5, fixlam=1,
             fixwd0=None, fixwd1=None, fixwd2=None,
             fixwd3=None, fixwd4=None, rsp=[],
             yb=1e-5, racc=0.4, npr=10, xb=1, xw=1, nwp=2, nofit=0):
    nd = len(dfs)
    sp = []
    ips = []
    ip0 = 0
    for i in range(nd):
        s0 = read_spec(dfs[i], emin, emax, yb, feff=feff)
        sp.append(s0)
        ips.append(init_params(s0, lns, smin, fwhm, dx, ip0, xb, xw, nwp=nwp))
        ip0 += s0.inp[1]
    np = ip0
    p = zeros(np)
    p0 = zeros(np)
    p1 = zeros(np)
    s = zeros(np)
    ifx = zeros(np, dtype=int32)
    id = zeros(np, dtype=int32)
    cc = zeros((np,4))
    ip0 = 0
    it0 = 0
    
    for i in range(nd):
        if len(rsp) == nd:
            rw = loadtxt(rsp[i], unpack=1)
            fixwd0 = rw[0][nwp]
            fixwd1 = rw[0][nwp+1]
            fixwd2 = rw[0][nwp+2]
            fixwd3 = rw[0][nwp+3]
            fixwd4 = rw[0][nwp+4]
        ip1 = ip0+sp[i].inp[1]
        ib = sp[i].inp[5]
        iw = sp[i].inp[7]
        nw = sp[i].inp[6]
        iw1 = iw+nw
        iw2 = iw1+nw
        if nwp >= 4:
            iw3 = iw2+nw
            iw4 = iw3+nw
            if nwp == 5:
                iw5 = iw4+nw
        if i == 0:
            onw = nw
            oiw = iw
            oiw1 = iw1
            oiw2 = iw2
            if nwp >= 4:
                oiw3 = iw3
                oiw4 = iw4
                if nwp == 5:
                    oiw5 = iw5
        p[ip0:ip1] = ips[i][0]
        p0[ip0:ip1] = ips[i][1]
        p1[ip0:ip1] = ips[i][2]
        s[ip0:ip1] = ips[i][3]
        ifx[ip0:ip1] = ips[i][4]
        if i > 0:
            ifx[iw:iw1] = arange(oiw, oiw1, dtype=int32)
            if nwp >= 4:
                ifx[iw2:iw3] = arange(oiw2, oiw3, dtype=int32)
                ifx[iw3:iw4] = arange(oiw3, oiw4, dtype=int32)
                if nwp == 5:
                    ifx[iw4:iw5] = arange(oiw4, oiw5, dtype=int32)
            if fixlam == 1:
                ifx[ip0:ib:2] = arange(0,(ib-ip0),2,dtype=int32)
        if fixwd0 != None:
            p[iw:iw1] = fixwd0
            p0[iw:iw1] = fixwd0
            p1[iw:iw1] = fixwd0
        if fixwd1 != None:
            p[iw1:iw2] = fixwd1
            p0[iw1:iw2] = fixwd1
            p1[iw1:iw2] = fixwd1
        if fixwd2 != None:
            p[iw2:iw3] = fixwd2
            p0[iw2:iw3] = fixwd2
            p1[iw2:iw3] = fixwd2
        if fixwd3 != None:
            p[iw3:iw4] = fixwd3
            p0[iw3:iw4] = fixwd3
            p1[iw3:iw4] = fixwd3
        if fixwd4 != None:
            p[iw4:iw5] = fixwd4
            p0[iw4:iw5] = fixwd4
            p1[iw4:iw5] = fixwd4
        cc[ip0:ip1] = ips[i][5]
        id[ip0:ip1] = i
        ip0 += sp[i].inp[1]
        it0 += sp[i].inp[2]
    
    d = {'dfs': dfs,
         'nwp': nwp,
         'sp': sp,
         'id': id,
         'ifx': ifx,
         'cc': cc,
         'mp': p,
         'smp': s,
         'mp0': p0,
         'mp1': p1,
         'lnlk': zeros(nd)}
    if not nofit:
        mcmc(d, nm, racc=racc, npr=npr, sav=sav)
    return d

def plot_spec(d, i, sav='', op=0, res=0, xr=[], ylog=0):
    if not op:
        clf()
    s = d['sp'][i]
    xm = 0.5*(s.xlo+s.xhi)
    y = s.yd
    ym = s.eff*(s.ym[0]+s.ym[2])+s.yb
    if res == 1:
        errorbar(xm, y-ym, yerr=sqrt(1+ym),
                 marker='.', capsize=3, fmt=' ')
    elif res == 2:
        errorbar(xm, (y-ym)/sqrt(1+ym), yerr=1.0,
                 marker='.', capsize=3, fmt=' ')
    else:
        errorbar(xm, y, yerr=sqrt(1+ym),
                 marker='.', capsize=3, fmt=' ')
        plot(xm, ym)
    if i == 0:
        legend(['beam on fit', 'beam on data'])
    else:
        legend(['beam off fit', 'beam off data'])
    xlabel('Energy (eV)')
    ylabel('Counts')
    if len(xr) == 2:
        xlim(xr[0], xr[1])
    if ylog:
        yscale('log')
    if sav != '':
        savefig(sav)

def plot_rspec(d, i, xr=[], sav=''):
    a1 = plt.subplot2grid((4,1),(0,0),rowspan=3)
    a2 = plt.subplot2grid((4,1),(3,0),rowspan=1)
    s = d['sp'][i]
    xm = 0.5*(s.xlo+s.xhi)
    yd = s.yd
    ym = s.eff*(s.ym[0]+s.ym[2])+s.yb
    ye = sqrt(s.yd+1.0)
    a1.errorbar(xm, yd, yerr=ye, marker='.', capsize=3, fmt=' ')
    a1.plot(xm, ym)
    a2.errorbar(xm, (yd-ym)/ye, yerr=1.0, marker='.', capsize=3, fmt=' ')
    a2.plot([min(xm),max(xm)],[0.0,0.0])
    a2.set_ylim(-4,4)
    a2.set_xlabel('Energy (eV)')
    a1.set_ylabel('Counts')
    a2.set_ylabel(r'$\chi^2$')
    a1.set_xticklabels([])
    if len(xr) == 2:
        a1.set_xlim(xr[0], xr[1])
        a2.set_xlim(xr[0], xr[1])
    if sav != '':
        savefig(sav)
        
def plot_fwhm(d, i0=0, i1=1, sav='', op=0):
    if not op:
        clf()
    w = where(d['ene'] < 0)
    w = w[0]
    nw = len(w)
    im = min(1000,int(nw/2))
    s0 = d['sp'][i0]
    s1 = d['sp'][i1]
    iw0 = s0.inp[6]+s0.inp[7]
    iw1 = s1.inp[6]+s1.inp[7]
    w0 = d['hmp'][im:nw,iw0]*2.35
    w1 = d['hmp'][im:nw,iw1]*2.35
    wb0 = mean(w0)
    ws0 = std(w0)
    wb1 = mean(w1)
    ws1 = std(w1)
    h1 = histogram(w1, bins=25)
    h0 = histogram(w0, bins=25)
    step(h0[1][:-1], h0[0]/sum(h0[0]))
    step(h1[1][:-1], h1[0]/sum(h1[0]))
    legend(['beam on fwhm %4.2f+/-%4.2f'%(wb0,ws0),
            'beam off fwhm %4.2f+/-%4.2f'%(wb1,ws1)])
    xlabel('FWHM (eV)')
    ylabel('Probability (arb. unit)')
    if sav != '':
        savefig(sav)
    return wb0,ws0,wb1,ws1

def plot_dfwhm(d, i0=0, i1=1, sav='', op=0):
    if not op:
        clf()
    w = where(d['ene'] < 0)
    w = w[0]
    nw = len(w)
    im = min(1000,int(nw/2))
    s0 = d['sp'][i0]
    s1 = d['sp'][i1]
    iw0 = s0.inp[6]+s0.inp[7]
    iw1 = s1.inp[6]+s1.inp[7]
    w0 = d['hmp'][im:nw,iw0]*2.35
    w1 = d['hmp'][im:nw,iw1]*2.35
    dw = w0-w1
    wb = mean(dw)
    ws = std(dw)
    h = histogram(dw, bins=25)
    y = h[0]/sum(h[0])
    step(h[1][0:-1], y)
    xlabel('beam on/off FWHM difference (eV)')
    ylabel('Probability (arb. unit)')

    text(wb+0.5*ws, max(y)*0.9, 'mean=%4.2f+/-%4.2f eV'%(wb, ws))
    if sav != '':
        savefig(sav)

def plot_hp(z, i, bins=25, xsc=0, xlab='', op=0):
    if op == 0:
        clf()
    w = where(z['ene'] < 0)
    w = w[0]
    nw = len(w)
    im = min(1000,int(nw/2))
    y = z['hmp'][im:nw,i]
    h = histogram(y, bins=bins)
    x = h[1][:-1]
    if xsc == 0:
        plot(x, h[0]/max(h[0]), drawstyle='steps')
    elif xsc == 1:
        semilogx(10**x, h[0]/max(h[0]), drawstyle='steps')
    xlabel(xlab)
    ym = mean(y)
    ys = std(y)
    text(ym+1.5*ys, 0.9, '%.2f+/-%0.2f'%(ym,ys))
    return ym, ys

def plot_widths(tgts=['H2','H','N2','He'], sav='', op=0):
    if not op:
        clf()
    nd = len(tgts)
    yd0 = zeros(nd)
    ye0 = zeros(nd)
    yd1 = zeros(nd)
    ye1 = zeros(nd)
    for i in range(nd):
        fn = 'xrs_'+tgts[i].lower()+'.txt'
        r = loadtxt(fn, unpack=1)
        yd0[i] = r[0][1]
        ye0[i] = r[1][1]
        yd1[i] = r[0][6]
        ye1[i] = r[1][6]
        
    yd0 *= 2.355
    ye0 *= 2.355
    yd1 *= 2.355
    ye1 *= 2.355
    yd = yd0-yd1
    ye = sqrt(ye0**2+ye1**2)
    errorbar(range(nd), yd, yerr=ye, marker='o', capsize=4, fmt=' ', color='k')
    ymin = -0.4
    ymax = 1.2
    xlabel('Neutral Targets')
    ylabel('Decrease in FWHM (eV)')
    ax = gca()
    ax.set_ylim(ymin, ymax)
    t = ax.set_xticks([0,1,2,3])
    t = ax.set_xticklabels(tgts)
    ay = ax.twinx()
    t0 = 2*9.3e8*mean(yd0)*ymin/7e3**2/2.355**2
    t1 = 2*9.3e8*mean(yd0)*ymax/7e3**2/2.355**2
    ay.set_ylim(t0, t1)
    t = ay.set_yticks(arange(int(t0/10)*10,int(t1/10)*10+1,10))
    ay.set_ylabel('Decrease in Ion Temperature (eV/u)')
    if sav != '':
        savefig(sav)
    return yd,ye,yd0,ye0,yd1,ye1
    
def redte(d, i0=0, i1=1):
    w = where(d['ene'] < 0)
    w = w[0]
    nw = len(w)
    im = min(1000,int(nw/2))
    s0 = d['sp'][i0]
    s1 = d['sp'][i1]
    iw0 = s0.inp[6]+s0.inp[7]
    iw1 = s1.inp[6]+s1.inp[7]
    w0 = d['hmp'][im:nw,iw0]*2.35
    w1 = d['hmp'][im:nw,iw1]*2.35
    wd = mean(w0-w1)
    wb = mean(w0)
    ws = std(w0-w1)
    
    r = 2*9.3e8*wb*wd/7e3**2/2.35**2
    e = r*ws/wd
    return r,e

def hardness(d, k):
    w = where(d['ene'] < 0)
    w = w[0]
    nw = len(w)
    im = min(1000,int(nw/2))
    nd = len(d['sp'])
    r = zeros(nd)
    rs = zeros(nd)
    h = []
    for i in range(nd):
        s = d['sp'][i]
        w0 = (where((s.tr[0] == k)&(s.tr[3] == 201)))[0]
        w1 = (where((s.tr[0] == k)&(s.tr[3] != 201)))[0]
        s0 = sum(d['hmp'][im:nw,s.inp[0]+1+w0*2],1)
        s1 = sum(d['hmp'][im:nw,s.inp[0]+1+w1*2],1)
        r[i] = mean(s1/s0)
        rs[i] = std(s1/s0)
        h.append(histogram(s1/s0,bins=30))
    return r,rs,h

def tablines(d, i, k, ofn, n0=2, n1=16):
    w = where(d['ene'] < 0)
    w = w[0]
    nw = len(w)
    im = min(1000,int(nw/2))
    s = d['sp'][i]
    p = d['hmp'][im:nw,s.inp[0]:]
    fwhm = 2.35*mean(p[:,1+s.inp[7]-s.inp[0]])
    f = open(ofn, 'w')
    for n in range(n0,n1+1):
        tp = n*100+1
        w = where((s.tr[0] == k)&(s.tr[3] == tp))
        w = w[0]
        if len(w) == 0:
            a = '%2d %2d %4d %12.5E %12.5E %12.5E %12.5E'%(k,n,0,0.0,0.0,0.0,0.0)
            f.write(a+'\n')
            continue
        if n == 2:
            uw = zeros(len(w), dtype=int32)
            for jw in range(len(w)):
                if uw[jw] == 1:
                    continue
                j = w[jw]
                u = where(abs(s.tr[4][w]-s.tr[4][j])<fwhm)[0]
                uw[u] = 1
                wf = 2*w[u]
                y = sum(p[:,wf+1],1)
                x = sum(p[:,wf]*p[:,wf+1],1)/y
                dx = std(x)
                dy = std(y)
                x = mean(x)
                y = mean(y)
                i1 = 0
                for iu in range(len(u)):
                    i1 += int(s.tr[2][w[u[iu]]]*10**iu)
                a = '%2d %2d %4d %12.5E %12.5E %12.5E %12.5E'%(k,n,i1,x,dx,y,dy)
                f.write(a+'\n')
        else:
            y = sum(p[:,2*w+1],1)
            x = sum(p[:,2*w]*p[:,2*w+1],1)/y
            dx = std(x)
            dy = std(y)
            x = mean(x)
            y = mean(y)
            a = '%2d %2d %4d %12.5E %12.5E %12.5E %12.5E'%(k,n,s.tr[2][w[0]],x,dx,y,dy)
            f.write(a+'\n')
    f.close()
    
def tab_all(d, tfn=''):
    if tfn != '':
        f1 = open(tfn+'_hlike.tex', 'w')
        f2 = open(tfn+'_helike.tex', 'w')
        df1 = []
        df2 = []
    labs = ['alpha','beta','gamma','delta','epsilon','eta']
    for i in range(len(d['dfs'])):
        fn = d['dfs'][i].split('.')[0]
        for k in [1,2]:
            if k == 1:
                ofn = fn+'_hlike.txt'
            else:
                ofn = fn+'_helike.txt'
            print(ofn)
            tablines(d, i, k, ofn)
            if tfn == '':
                continue
            r = loadtxt(ofn, unpack=1)
            if i == 0:
                if k == 1:
                    df1 = ['']*(2+len(r[0]))
                else:
                    df2 = ['']*(5+len(r[0]))
            for j in range(len(r[0])):
                if k == 1:
                    s = df1[j]
                else:
                    s = df2[j]
                n = int(r[1][j])
                n2 = n-2
                m = int(r[2][j])
                lab = ''
                if n == 2:
                    if k == 1:
                        if m == 12:
                            lab = r'Ly$_\alpha 2$'
                        else:
                            lab = r'Ly$_\alpha 1$'
                    else:
                        if m == 6:
                            lab = 'w'
                        elif m == 1:
                            lab = 'z'
                        elif m == 5:
                            lab = 'x'
                        elif m == 3:
                            lab = 'y'
                elif n2 < len(labs):
                    if k == 1:
                        lab = 'Ly$_\\' + labs[n2] + '$'
                    else:
                        lab = 'K$_\\' + labs[n2] + '$'
                if i == 0:
                    s += '%2d & %15s & %8.1f & %6.1f'%(n, lab, r[5][j], r[6][j])
                else:
                    s += ' & %8.1f & %6.1f'%(r[5][j], r[6][j])
                if k == 1:
                    df1[j] = s
                else:
                    df2[j] = s
            if k == 1:
                j += 1
                rd = r[5][1]/r[5][0]
                re = sqrt((r[6][1]/r[5][1])**2+(r[6][0]/r[5][0])**2)*rd
                if i == 0:
                    df1[j] += '%2s & %15s & %8.2f & %6.2f'%('', 'Ly$_\\alpha$ 1:2',
                                                            rd, re)
                else:
                    df1[j] += ' & %8.4f & %6.4f'%(rd, re)
                j += 1
                rd = sum(r[5][2:])/sum(r[5][:2])
                re = sqrt(sum(r[6][2:]**2)/sum(r[5][2:])**2+sum(r[6][:2]**2)/sum(r[5][:2])**2)*rd
                if i == 0:
                    df1[j] += '%2s & %15s & %8.4f & %6.4f'%('', '$H$', rd, re)
                else:
                    df1[j] += ' & %8.4f & %6.4f'%(rd, re)
            else:
                j += 1
                rd = r[5][2]/sum(r[5][:2])
                re = sqrt((r[6][2]/r[5][2])**2+sum(r[6][:2]**2)/sum(r[5][:2])**2)*rd
                if i == 0:
                    df2[j] += '%2s & %15s & %8.4f & %6.4f'%('', '$R$', rd, re)
                else:
                    df2[j] += ' & %8.4f & %6.4f'%(rd, re)
                j += 1
                rd = sum(r[5][:3])/r[5][3]
                re = sqrt(sum(r[6][:3]**2)/sum(r[5][:3])**2+(r[6][3]/r[5][3])**2)*rd
                if i == 0:
                    df2[j] += '%2s & %15s & %8.4f & %6.4f'%('', '$G$', rd, re)
                else:
                    df2[j] += ' & %8.4f & %6.4f'%(rd, re)
                j += 1
                rd = sum(r[5][:3])/sum(r[5][3:])
                re = sqrt(sum(r[6][:3]**2)/sum(r[5][:3])**2+sum(r[6][3:]**2)/sum(r[5][3:])**2)*rd
                if i == 0:
                    df2[j] += '%2s & %15s & %8.4f & %6.4f'%('', '$G^{\\prime}$', rd, re)
                else:
                    df2[j] += ' & %8.4f & %6.4f'%(rd, re)
                j += 1
                rd = sum(r[5][4:])/sum(r[5][:4])
                re = sqrt(sum(r[6][4:]**2)/sum(r[5][4:])**2+sum(r[6][:4]**2)/sum(r[5][:4])**2)*rd
                if i == 0:
                    df2[j] += '%2s & %15s & %8.4f & %6.4f'%('', '$H$', rd, re)
                else:
                    df2[j] += ' & %8.4f & %6.4f'%(rd, re)
                j += 1
                rd = sum(r[5][4:])/r[5][3]
                re = sqrt(sum(r[6][4:]**2)/sum(r[5][4:])**2+(r[6][3]/r[5][3])**2)*rd
                if i == 0:
                    df2[j] += '%2s & %15s & %8.4f & %6.4f'%('', '$H^{\\prime}$', rd, re)
                else:
                    df2[j] += ' & %8.4f & %6.4f'%(rd, re)
                    
    if tfn != '':
        [f1.write(s+'\\\\\n') for s in df1]
        [f2.write(s+'\\\\\n') for s in df2]
        f1.close()
        f2.close()        
                  
def plot_hr(md, op=0, sav=''):
    if not op:
        clf()
    r0 = loadtxt('pra_hr.txt', unpack=1)
    tgt = ['H2', 'H', 'N2', 'He']
    r2s = ['Rw','Rz','Ri']
    ra0 = r0[0]/(1+r0[0])
    dra0 = ra0*(r0[1]/r0[0])
    szi = r0[6]/(1+r0[6])
    dzi = szi*(r0[7]/r0[6])
    rw0 = (1/r0[4])*szi
    drw0 = rw0*(r0[5]/r0[4])
    rz0 = r0[2]/(1+r0[2])*szi
    drz0 = rz0*(r0[3]/r0[2])
    ri0 = (1/r0[2])/(1/r0[2]+1) * szi
    dri0 = ri0*(r0[3]/r0[2])
    r2r0 = [rw0,rz0,ri0]
    r2e0 = [drw0, drz0, dri0]
    #errorbar(ra0, (rz0+ri0)/rw0, xerr=dra0, yerr=r2e0[md],
    #         marker='o', capsize=3, fmt=' ')
    #labs = ['PRA2014']
    r2r = zeros(3)
    r2e = zeros(3)
    labs = []
    for i in range(len(tgt)):
        fn1 = 'Fe%sNewCX_hlike.txt'%tgt[i]
        fn2 = 'Fe%sNewCX_helike.txt'%tgt[i]
        r1 = loadtxt(fn1, unpack=1)
        r2 = loadtxt(fn2, unpack=1)
        ra = sum(r1[5][:3])/sum(r1[5])
        dra = ra*sqrt(sum(r1[6][:3]**2)/sum(r1[5][3:])**2)
        r2r[0] = r2[5][3]/sum(r2[5])
        r2e[0] = r2r[0]*(r2[6][3]/r2[5][3])
        r2r[1] = r2[5][2]/sum(r2[5])
        r2e[1] = r2r[1]*(r2[6][2]/r2[5][2])
        r2r[2] = sum(r2[5][:2])/sum(r2[5])
        r2e[2] = r2r[2]*sqrt(sum(r2[6][:2]**2)/sum(r2[5][:2])**2)
        errorbar([ra], [r2r[md]], xerr=[dra], yerr=[r2e[md]],
                 marker='o', capsize=3, fmt=' ')
        labs.append('Fe+'+tgt[i])
    legend(labs)
    xlabel('Ra')
    ylabel(r2s[md])
    if sav != '':
        savefig(sav)
        
def save_xrsprof(d, fn):
    w = where(d['ene'] < 0)
    w = w[0]
    nw = len(w)
    im = min(1000,int(nw/2))
    s = d['sp']
    i0 = s[0].inp[-1]
    i1 = s[1].inp[-1]
    p = d['hmp'][im:nw]
    nwp = int((s[0].inp[1]-s[0].inp[-1])/s[0].inp[-2])
    wb = mean(p[:,list(range(i0,i0+nwp))+list(range(i1,i1+nwp))],0)
    ws = std(p[:,list(range(i0,i0+nwp))+list(range(i1,i1+nwp))],0)
    
    savetxt(fn, transpose((wb,ws)))

def save_xrswidth(d, fn):    
    w = where(d['ene'] < 0)
    w = w[0]
    nw = len(w)
    im = min(1000,int(nw/2))
    p = d['hmp'][im:nw]
    s = d['sp']
    wb = zeros(4)
    ws = zeros(4)
    for i in range(4):
        i0 = s[i].inp[-1]+1
        wb[i] = mean(p[:,i0])
        ws[i] = std(p[:,i0])
    savetxt(fn, transpose((wb,ws)))

    
def test(*p, **k):
    print(p)

    print(k)
    
