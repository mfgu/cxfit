from numpy import *
from pylab import *
import os
from collections import namedtuple
from scipy import special, integrate, interpolate, optimize
from pfac import fac
from pfac import rfac
from pfac import crm
import pickle

RadCas = namedtuple('RadCas', ['rd', 'r', 'ir0', 'ir1', 'egy', 'wa', 'ai', 'trm', 'tri', 'z', 'k', 'ide', 'de', 'im', 'nt', 'im1', 'na'])
SpecData = namedtuple('SpecData', ['fns', 'stype', 'er', 'elo', 'ehi', 'em', 'yc', 'yd', 'eff', 'ye'])
Response = namedtuple('Response', ['sig', 'bn', 'ar', 'aj'])
IonData = namedtuple('IonData', ['ddir', 'sdir', 'ps0', 'ds0', 'rd', 'r', 'ir0', 'ir1', 'idx', 'egy', 'ai', 'ad', 'zai',
                                 'z', 'k', 'ns', 'nm', 'kmin', 'kmax', 'an', 'ae', 'anr', 'wk0',
                                 'wk', 'we', 'swk', 'swe', 'xk',
                                 'ide', 'de', 'ws', 'emin', 'emax'])
FitMCMC = namedtuple('FitMCMC', ['imp', 'sp', 'ym', 'mpa', 'mpe', 'ra', 'ds', 'rs', 'ap', 'hmp', 'ene','nde','ide0','ide1','iid', 'ia','iw','frej','rrej'])
    
def rcs(f):
    r = loadtxt(f)
    if len(r) > 0:
        return transpose(loadtxt(f))
    else:
        return r

def rad_cas(z, k, emin, emax, nmin=0, nmax=100, ist='', ddir='data', sdir='spec'):
    a = fac.ATOMICSYMBOL[z]
    ps0 = '%s/%s%02d'%(sdir, a, k)
    ds0 = '%s/%s%02d'%(ddir, a, k)
    rd = rfac.FLEV(ds0+'a.en')
    r = rcs(ps0+'a.r1')
    ir0 = int32(r[1])
    ir1 = int32(r[0])
    im1 = max(ir1)
    im = 1 + im1
    trm = zeros((im,im))
    trm[ir0,ir1] = -r[2]
    for i in range(im):
        trm[i,i] = -sum(trm[:i,i])
    w = []
    if k == 1:
        tpr = crm.TwoPhoton(z, 0)
        w = where(rd.n == '2s+1(1)1')
    elif k == 2:
        tpr = crm.TwoPhoton(z, 1)
        w = where(rd.n == '1s+1(1)1.2s+1(1)0')
    if len(w) == 1:        
        w = w[0]
        if len(w) == 1:
            w = w[0]
            trm[w,w] += tpr
    egy = rd.e[ir1] - rd.e[ir0]
    w = where(logical_and(egy > emin, egy < emax))
    ir0 = ir0[w]
    ir1 = ir1[w]
    egy = egy[w]
    nt = len(ir0)
    tri = pinv(trm[1:,1:],rcond=1e-12)
    ai = matmul(diag(-trm[ir0,ir1]),tri[ir1-1])
    w=where(ai < 1e-10)
    ai[w] = 0
    v = rd.v[range(1,im)]/100
    w = where(logical_or(v < nmin, v > nmax))
    ai[:,w] = 0.0
    nis = len(ist)
    if nis > 0:
        for i in range(1,im):                
            if rd.n[i][:nis] != ist:
                ai[:,i-1] = 0.0
    sa = sum(ai, axis=0)
    wa = where(sa > 0)
    wa = wa[0]
    na = len(wa)
    ai = ai[:,wa]
    wa = wa+1
    ide = zeros(nt, dtype=int8)
    de = zeros(nt)
    return RadCas(rd, r, ir0, ir1, egy, wa, ai, trm, tri, z, k, ide, de, im, nt, im1, na)

def ion_data(z, k, ns, ws0, emin, emax, ddir='data', sdir='spec', kmin=0, kmax=-1):
    a = fac.ATOMICSYMBOL[z]
    ps0 = '%s/%s%02d'%(sdir, a, k)
    ws0 = atleast_1d(array(ws0))
    ws = abs(ws0[0])
    nc = 'n'
    if ws0[0] < 0:
        nc = 'm'
    if k > 0:
        ds0 = '%s/%s%02d'%(ddir, a, k)
        fn = ds0 + 'a.en'
        rd = rfac.FLEV(ds0+'a.en')
        fn = ps0+'a.r1'
        r = rcs(fn)
        ir0 = int32(r[1])
        ir1 = int32(r[0])
        im = 1 + max(ir1)
        idx = zeros((im,im), dtype=int32)
        egy = rd.e[ir1] - rd.e[ir0]
        w = where(logical_and(egy > emin, egy < emax))
        ir0 = ir0[w]
        ir1 = ir1[w]
        egy = egy[w]
        idx[:,:] = -1
        idx[ir0,ir1] = arange(len(ir0), dtype=int32)
    else:
        rd = None
        r = None
        ps = '%sn%02dk%02d'%(ps0, ns[0], 0)
        ofn = ps + 'a.ln'
        d = rcs(ofn)
        egy = atleast_1d(d[4])
        w = where(logical_and(egy > emin, egy < emax))
        nw = len(w[0])
        im = nw+1
        ir0 = zeros(nw, dtype=int32)
        ir1 = 1+arange(nw, dtype=int32)
        egy = egy[w]
        idx = zeros((im,im),dtype=int32)
        idx[:,:] = -1
        idx[ir0,ir1] = arange(len(ir0), dtype=int32)
    nt = len(ir0)
    nn = len(ns)
    nm = int(mean(ns))
    if kmax < 0:
        kmax = nm-1
    nm = kmax-kmin+1
    xk = arange(nm)+kmin
    ai = zeros((nn, nt, nm))
    ad = zeros((nn, nm, im))
    zai = zeros(nm, dtype=int32)
    zai[:] = 1
    for ws1 in ws0:
        wzai = zeros(nm, dtype=int32)
        ws = abs(ws1)
        nc = 'n'
        if ws1 < 0:
            nc = 'm'
        for i in range(nn):
            n = ns[i]
            for kk in range(kmin, kmax+1):
                if kk >= n:
                    continue
                ki = kk-kmin
                if k > 0:
                    ps = '%s%s%02dk%02d'%(ps0, nc, (ws*100+n), kk)
                    ofn = ps + 'a.ln'
                    d = rcs(ofn)
                    if len(d) == 0:
                        wzai[ki] = 1
                        continue
                    it0 = atleast_1d(int32(d[1]))
                    it1 = atleast_1d(int32(d[2]))
                    ofn = ps + 'a.r7'
                    c = rcs(ofn)
                    ic0 = atleast_1d(int32(c[0]))
                    ic1 = atleast_1d(int32(c[1]))
                    w = where(ic0 == min(ic0))
                    w = w[0]
                    ad[i,ki,ic1[w]] += c[2][w]
                else:
                    it0 = ir0
                    it1 = ir1
                si = atleast_1d(d[6]).copy()
                if (k == 0):
                    si[:ki] = 0.0
                    si[ki+1:] = 0.0
                ix = idx[it0, it1]
                w = where(ix >= 0)
                ix = ix[w]
                ai[i,ix,ki] += si[w]
        zai = logical_and(zai, wzai)
    an = zeros(nn+1)
    ae = zeros(nn+1)
    wk0 = fac.LandauZenerLD(z, nm, 5)
    wk = zeros(nm)
    we = zeros(nm)
    swk = zeros(nm)
    swe = zeros(nm)
    ide = zeros(nt, dtype=int8)
    de = zeros(nt)

    return IonData(ddir, sdir, ps0, ds0, rd, r, ir0, ir1, idx, egy, ai, ad, zai, z, k, ns, nm, kmin, kmax, an, ae, an.copy(), wk0, wk, we, swk, swe, xk, ide, de, ws0, emin, emax)

def escape_peak(x, beta, gamma):
    xx = (x+beta)/(1.41421356*gamma)
    yc = zeros(len(xx))
    w = where(xx >= 0)
    yc[w] = log(special.erfcx(xx[w])) - xx[w]*xx[w]
    w = where(xx < 0)
    yc[w] = log(special.erfc(xx[w]))
    y = yc + x*beta
    return y

def response(d, s, sig):
    elo = s.elo
    ehi = s.ehi
    em = 0.5*(elo+ehi)
    nr = len(em)
    nt = len(d.ir0)
    nn = len(d.ns)
    ar = zeros((nr,nt))
    rd = d.rd
    ir1 = d.ir1
    ir0 = d.ir0
    nm = d.nm
    if rd == None:
        egy = d.egy.copy()
    else:
        egy = rd.e[ir1] - rd.e[ir0]
    w = where(d.ide > 0)
    if len(w[0]) > 0:
        egy[w] += d.de[w]
    w = where(d.ide < 0)
    w = w[0]
    for i in w:
        j = -d.ide[i]-1
        egy[i] = egy[j] + d.de[i]
    bn = 0.0
    fxy = None
    ns = len(sig)
    wf0 = (ehi-elo)/sig[0]
    wf1 = 0.3989423*wf0
    if (ns >= 3):
        if ns == 3:
            gamma = 1.0
        else:
            gamma = sig[3]
        x = arange(-50.0/sig[1], 50.0*gamma, 0.025)
        y = escape_peak(x, sig[1], gamma)
        yi = zeros(len(y))
        w = where(y > -300)
        yi[w] = exp(y[w])
        bn = integrate.simps(yi*0.3989423, x)
        fxy = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0],y[-1]))
    emid = em[0]
    #emid = mean(em)
    erng = 1e3
    for i in range(nt):        
        e = egy[i]        
        if ns == 2 or ns == 5:
            #esig = sig[0]*pow(e/emid, sig[-1])
            fsig = 1 + sig[-1]*(e-emid)/erng
            esig = sig[0]*sqrt(fsig)
            rsig = sig[0]/esig
        else:
            esig = sig[0]
            rsig = 1.0
        de = (em-e)/esig
        ade = abs(de)
        w = where(ade < 10.0)
        if len(w[0]) > 0:
            ar[w,i] = wf1[w]*rsig*exp(-0.5*ade[w]*ade[w])            
            if ns >= 3:
                w = where(de < 30)
                if len(w[0]) > 0:
                    ye = wf1[w]*rsig*exp(fxy(de[w]))/bn
                    ar[w,i] = (ar[w,i] + sig[2]*ye)/(1+sig[2])
                
    aj = zeros((nn,nr,nm))
    for i in range(nn):
        aj[i] = matmul(ar, d.ai[i])
    return Response(sig, bn, ar, aj)

def calc_spec(d, r, ar=None):
    n = len(d.ns)
    if ar == None:
        ar = 0.0
        for i in range(n):
            ar += d.anr[n]*d.anr[i]*r.aj[i]
    y = matmul(ar, d.wk)
    return (y,ar)

def scale_spec(yd0, ye):
    nd = len(yd0)
    yd = zeros(nd)
    yd[:] = yd0
    w = where(ye > 0)
    eff = zeros(nd)
    eff[:] = 1.0
    eff[w] = yd[w]/(ye[w]*ye[w])
    yd[w] = yd[w]*eff[w]
    return (yd, eff)

def mcmc_cmp(z, m):
    np = len(z.mpa)
    ni = len(z.ds)
    nsig = len(z.rs[0].sig)
    ia = z.ia
    iw = z.iw
    iid = z.iid
    z.mpa[:] = z.hmp[m,:]
    for ii in range(ni):
        z.ds[ii].anr[:-1] = z.mpa[ia[ii]:ia[ii+1]]
        z.ds[ii].anr[-1] = z.mpa[iid[ii]]
        z.ds[ii].wk[:] = z.mpa[iw[ii]:iw[ii+1]]
        for jj in range(len(z.ds[ii].ns)):
            nn = z.ds[ii].ns[jj]
            if nn <= z.ds[ii].kmax:
                xw = sum(z.mpa[z.iw[ii]:(z.iw[ii+1]-(z.ds[ii].kmax+1-nn))])
            else:
                xw = 1.0
            z.ds[ii].an[jj] = xw*z.ds[ii].anr[jj]
        xw = sum(z.ds[ii].an[:-1])
        z.ds[ii].an[:-1] /= xw
        z.ds[ii].an[-1] = z.ds[ii].anr[-1]*xw
    
def mcmc_avg(z, m, m1=-1, eps=-1e30, rmin=-1e30):
    np = len(z.mpa)
    ni = len(z.ds)
    nsig = len(z.rs[0].sig)
    ia = z.ia
    iw = z.iw
    iid = z.iid
    if m1 <= 0:
        m1 = z.imp
    r = z.ene[m1-m:m1]
    ra = mean(r)
    rd = std(r)
    wr = m1-m+arange(m, dtype=int32)
    wr0 = wr
    if eps > -1e30:
        wr = where(r > ra+eps*rd)
        wr = wr[0]+m1-m
    elif rmin > -1e30:
        wr = where(r > rmin)
        wr = wr[0]+m1-m
    for ip in range(np):
        x = z.hmp[wr,ip]
        x0 = z.hmp[wr0, ip]
        z.mpa[ip] = mean(x)
        z.mpe[ip] = std(x0)
    for ii in range(ni):
        z.ds[ii].anr[:-1] = z.mpa[ia[ii]:ia[ii+1]]
        z.ds[ii].anr[-1] = z.mpa[iid[ii]]
        z.ds[ii].ae[:-1] = z.mpe[ia[ii]:ia[ii+1]]
        z.ds[ii].ae[-1] = z.mpe[iid[ii]]
        z.ds[ii].wk[:] = z.mpa[iw[ii]:iw[ii+1]]
        z.ds[ii].we[:] = z.mpe[iw[ii]:iw[ii+1]]
        for jj in range(len(z.ds[ii].ns)):
            nn = z.ds[ii].ns[jj]
            if nn <= z.ds[ii].kmax:
                xw = sum(z.mpa[z.iw[ii]:(z.iw[ii+1]-(z.ds[ii].kmax+1-nn))])
            else:
                xw = 1.0
            z.ds[ii].an[jj] = xw*z.ds[ii].anr[jj]
        xw = sum(z.ds[ii].an[:-1])
        z.ds[ii].an[:-1] /= xw
        z.ds[ii].an[-1] = z.ds[ii].anr[-1]*xw
    z.ym[:] = 0.0
    for ii in range(ni):
        z.rs[ii] = response(z.ds[ii], z.sp, z.mpa[:nsig])
        (y,r) = calc_spec(z.ds[ii], z.rs[ii])
        z.ym[:] += z.sp.eff*y
        
def read_spec(df, stype, er=[]):
    if type(df) == tuple:
        if stype == 0:
            elo, ehi, yd, ye = df
        elif stype == 3:
            if (len(er) == 0):
                er = [800.0, 1.6e3]
            d = rcs(df[0])
            t = rcs(df[1])
            w = where(logical_and(d[0] > er[0]/1e3, d[1] < er[1]/1e3))
            elo = d[0][w]*1e3
            ehi = d[1][w]*1e3
            yc = d[2][w]
            fti = interpolate.interp1d(t[0], t[1], kind='linear', bounds_error=False, fill_value=(0.0, 1.0))            
            eff = fti(0.5*(elo+ehi))
            w = where(yc < 0)
            yc[w] = 0
            yd = yc/eff
            ye = sqrt(yc)/eff
        elif stype == 4:
            elo, ehi, yc, eff = df
            yd = yc/eff
            ye = sqrt(yc)/eff
    else:
        d = rcs(df)
        if stype == 0:
            if (len(er) == 0):
                er = [2e3, 4e3]
            w = where(logical_and(d[0] > er[0]/1e3, d[1] < er[1]/1e3))
            elo = d[0][w]*1e3
            ehi = d[1][w]*1e3
            yd = d[2][w]
            ye = d[3][w]
        elif stype == 1:
            if (len(er) == 0):
                er = [2e3, 4e3]
            w = where(logical_and(d[0] > er[0], d[0] < er[1]))
            nw = len(w[0])
            de = zeros(nw)
            de[:nw-1] = d[0][w][1:] - d[0][w][0:-1]
            de[nw-1] = de[nw-2]
            elo = d[0][w]
            ehi = d[0][w]+de
            yd = d[1][w]
            ye = d[2][w]
        elif stype == 2:
            if (len(er) == 0):
                er = [2.25e3, 4.75e3]
            w = where(logical_and(d[0] > er[0], d[0] < er[1]))
            nw = len(w[0])
            de = zeros(nw)
            de[:nw-1] = d[0][w][1:] - d[0][w][0:-1]
            de[nw-1] = de[nw-2]
            elo = d[0][w]
            ehi = d[0][w]+de
            yd = d[1][w]
            ye = sqrt(yd)
        elif stype == 3:
            if (len(er) == 0):
                er = [2e3, 4e3]
            w = where(logical_and(d[0] > er[0], d[1] < er[1]))
            elo = d[0][w]
            ehi = d[1][w]
            yc = d[2][w]            
            ye = sqrt(d[2][w])
            eff = d[3][w]
            w = where(eff <= 0)
            eff[w] = 1.0
            yd = yc/eff
            ye = ye/eff
            
    if stype < 3:
        (yc,eff) = scale_spec(yd, ye)
    return SpecData(df, stype, er, elo, ehi, 0.5*(elo+ehi), yc, yd, eff, ye)

def load_ecf(ds, ecf):
    for d in ds:
        d.ide[:] = 0
        d.de[:] = 0
    if ecf != '':
        r = rcs(ecf)
        k = int32(r[0])
        i1 = int32(r[1])
        i0 = int32(r[2])
        m = int32(r[3])
        n1 = int32(r[4])
        n0 = int32(r[5])
        e = r[6]
        for d in ds:
            w = where(k == d.k)
            w = w[0]
            for i in w:
                s = where(logical_and(d.ir1 == i1[i], d.ir0 == i0[i]))
                s = s[0]
                if len(s) == 1:
                    s = s[0]
                    d.ide[s] = 2
                    d.de[s] = e[i]
                    if n1[i] <= 0:
                        if m[i] == 0:
                            d.de[s] -= d.egy[s]
                    else:
                        t = where(logical_and(d.ir1 == n1[i], d.ir0 == n0[i]))
                        t = t[0]
                        if len(t) == 1:
                            t = t[0]
                            d.ide[s] = -t-1
                            if m[i] == 0:
                                d.de[s] -= d.egy[t]
                            elif m[i] == 2:
                                d.de[s] = d.egy[s] - d.egy[t]
                                
def mcmc_spec(ds, sp, sig, eth, imp, fixld=[], fixnd=[], racc=0.4, wb=[], wsig=[], sav=[], mps=[], nburn=0, nopt=0, sopt=0, fopt='', yb=1e-2, ecf=''):
    yd = sp.yc + yb
    eff = sp.eff
    elo = sp.elo
    ehi = sp.ehi
    nd = len(yd)
    yi = int32(yd)
    nfac = int32(max(yd)+10)
    lnfac = zeros(nfac)
    lnfac[0] = 0.0    
    for i in range(1, nfac):
        lnfac[i] = lnfac[i-1]+log(i)
    lnydi = lnfac[yi] + (yd-yi)*(lnfac[yi+1]-lnfac[yi])
    lnydi -= log(sqrt(2*pi*yd))
    slnydi = sum(lnydi)
    na = 0
    nw = 0
    nde = 0
    ia = [0]
    iw = [0]
    ide0 = []
    ide1 = []
    ti = 0.0
    ni = len(ds)
    d0 = ds[0]
    i0 = 0
    load_ecf(ds, ecf)
    for ii in range(len(ds)):
        d = ds[ii]
        if d.k != d0.k:
            w = where(d0.ide == 1)
            nde0 = nde
            nde = nde + len(w[0])
            for i1 in range(i0, ii): 
                ide0.append(nde0)
                ide1.append(nde)
                if i1 > i0:
                    ds[i1].ide[:] = d0.ide
            d0 = d
            i0 = ii
        na += len(d.ns)
        nw += d.nm
        ia.append(na)
        iw.append(nw)
        si = sum(sum(d.ai,axis=0),axis=1)
        w = where(logical_and(d.egy > elo[0], d.egy < ehi[-1]))
        msi = max(si[w])
        ti += sum(si[w])
        ww = where(logical_and(si[w] > msi*eth, d.ide[w] == 0))
        if (len(ww[0]) > 0):
            d.ide[w[0][ww]] = 1

    w = where(d0.ide == 1)
    nde0 = nde
    nde = nde + len(w[0])
    for i1 in range(i0, len(ds)):
        ide0.append(nde0)
        ide1.append(nde)
        if i1 > i0:
            ds[i1].ide[:] = d0.ide
    nsig = len(sig)
    ide0 = [nsig+x for x in ide0]
    ide1 = [nsig+x for x in ide1]
    nde1 = nde + nsig
    nde2 = nde1 + ni
    iid = range(nde1,nde2)
    ia = [nde2+x for x in ia]
    iw0 = iw
    iw = [ia[-1]+x for x in iw]
    np = nde2+na+nw
    spw = nde2+na
    mp = zeros(np)
    ydm = 1e-5*yd
    yt = sum(yd/eff)
    ye = sqrt(yd)
    yte = sqrt(sum(yd))
    mp[:nsig] = sig
    ai0 = yt/ti
    mp[iid] = ai0    
    for i in range(ni):
        mp[ia[i]:ia[i+1]] = 1.0/(ia[i+1]-ia[i])
        mp[iw[i]:iw[i+1]] = 1.0/(iw[i+1]-iw[i])
        wz = where(ds[i].zai > 0)
        mp[iw[i]+wz[0]] = 0.0
    mp0 = zeros(np)
    mp1 = zeros(np)
    smp = zeros(np)
    smp[:nsig] = 0.25*mp[:nsig]
    smp[nsig:nde1] = 1.0
    smp[iid] = 0.25*ai0
    smp[ia[0]:ia[-1]] = 0.1*mp[ia[0]:ia[-1]]
    smp[iw[0]:iw[-1]] = 0.1*mp[iw[0]:iw[-1]]
    mp0[:nsig] = sig*0.01
    mp1[:nsig] = sig*100.0
    w = where(smp <= 0)
    smp[w] = 1e-5
    if (nsig == 2 or nsig == 5):
        mp0[nsig-1] = 0.0
        mp1[nsig-1] = 3.0
    for i in range(len(wsig)):
        mp0[i] = wsig[i][0]
        mp1[i] = wsig[i][1]
    mp0[nsig:nde1] = -2.0
    mp1[nsig:nde1] = 2.0
    for ii in range(ni):
        if ds[ii].k == 0:
            mp0[ide0[ii]:ide1[ii]] = -100.0
            mp1[ide0[ii]:ide1[ii]] = 100.0
    mp0[iid] = 1e-3*ai0
    mp1[iid] = 1e3*ai0
    mp0[ia[0]:ia[-1]] = 0.0
    mp1[ia[0]:ia[-1]] = 1.0
    mp0[iw[0]:iw[-1]] = 0.0
    mp1[iw[0]:iw[-1]] = 1.0
    for b in wb:
        i = b[0]
        k = b[1]
        mp0[iw[i]+k] = b[2]
        mp1[iw[i]+k] = b[3]
        
    wde = []
    de0 = []
    for i in range(ni):
        ww = where(ds[i].ide == 1)
        wde.append(ww)
        if len(ww[0])>0:
            dd = zeros(len(ww[0]))
            dd[:] = ds[i].de[ww]
            de0.append(dd)
        else:
            de0.append(None)
    ap = zeros((nd,nw))
    rs = [None for i in range(len(ds))]
    rs0 = [None for i in range(len(ds))]
    y = zeros(nd)
    xin = arange(-30.0, 0.05, 0.05)
    xin[-1] = 0.0
    yin = integrate.cumtrapz(normpdf(xin, 0.0, 1.0), xin, initial=0.0)
    yin[0] = yin[1]*(yin[1]/yin[2])
    fin0 = interpolate.interp1d(xin, yin, kind='linear',
                                bounds_error=False, fill_value=(yin[0],0.5))
    fin1 = interpolate.interp1d(log(yin), xin, kind='linear',
                                bounds_error=False, fill_value=(-30.0,0.0))
    if len(mps) > 0:
        for ip in range(len(mps[0])):
            mp[ip] = mps[0][ip]
    if len(mps) > 1:
        for ip in range(len(mps[1])):
            smp[ip] = mps[1][ip]
    optres = [None, None]
    def cnorm(x):
        if (x > 0):
            return 1-fin0(-x)
        elif (x < 0):
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
    
    def getym(ip):
        if (ip < nde1):
            for i in range(ni):
                if ip < nsig or (ip >= ide0[i] and ip < ide1[i]):
                    ww = wde[i]
                    if len(ww[0]) > 0:
                        de0[i][:] = ds[i].de[ww]
                        ds[i].de[ww] = mp[ide0[i]:ide1[i]]
                    rs0[i] = rs[i]
                    rs[i] = response(ds[i], sp, mp[0:nsig])        

        if len(fixld) > 0:
            for i in range(ni):
                ii = fixld[i]
                if ii >= 0 and ii < ni:
                    wki = mp[iw[ii]:iw[ii+1]]
                    nm = min(ds[i].nm, ds[ii].nm)
                    mp[iw[i]:iw[i+1]][:] = 0.0
                    mp[iw[i]:iw[i+1]][:nm] = wki[:nm]
        if len(fixnd) > 0:
            for i in range(ni):
                ii = fixnd[i]
                if ii >= 0 and ii < ni:
                    wki = mp[ia[ii]:ia[ii+1]]
                    mp[ia[i]:ia[i+1]] = 0.0
                    mp[ia[i]:ia[i+1]] = wki
        ap[:,:] = 0.0
        for i in range(ni):
            x = mp[ia[i]:ia[i+1]]
            for k in range(len(ds[i].ns)):
                ap[:,iw0[i]:iw0[i+1]] += rs[i].aj[k]*x[k]*mp[iid[i]]
            wz = where(ds[i].zai > 0)
            mp[iw[i]+wz[0]] = 0.0
        wk = mp[iw[0]:iw[-1]]        
        y[:] = matmul(ap, wk)*eff

    def ilnlikely(ip):
        getym(ip)
        yt = y + yb
        r = yd*log(yt)-yt-lnydi
        return r
    
    def lnlikely(ip):
        r = ilnlikely(ip)
        r = sum(r)
        return r

    def eqanorm(x, ii):
        return sum(x[ia[ii]-nde1:ia[ii+1]-nde1])-1.0

    def eqwnorm(x, ii):
        return sum(x[iw[ii]-nde1:iw[ii+1]-nde1])-1.0

    def chi2m(x):
        mp[nde1:] = x
        getym(nde1)
        dy = zeros(nd+2*ni)
        w = where(ye > 0)
        dy[w] = (y[w]-yd[w])/ye[w]
        for ii in range(ni):
            if len(fixnd) <= ii or fixnd[ii] < 0:
                dy[nd+ii] = yte*eqanorm(x, ii)
        for ii in range(ni):
            if len(fixld) <= ii or fixld[ii] < 0:
                dy[nd+ni+ii] = yte*eqwnorm(x, ii)        
        return dy
    
    def chi2n(x):
        mp[:nde1] = x
        r = -ilnlikely(0)
        nx = np-nde1
        xx = zeros(nx)
        xx[:] = mp[nde1:]
        xb = []
        for ip in range(nde1, np):
            xb.append((mp0[ip],mp1[ip]))
        res = optimize.least_squares(chi2m, xx, jac='3-point', bounds=(mp0[nde1:],mp1[nde1:]), ftol=1e-1/nd, xtol=1e-2)
        mp[nde1:] = res.x
        """
        for ii in range(ni):
            mp[ia[ii]:ia[ii+1]] /= sum(mp[ia[ii]:ia[ii+1]])
            mp[iw[ii]:iw[ii+1]] /= sum(mp[iw[ii]:iw[ii+1]])
        res.x[:] = mp[nde1:]
        """
        r = res.cost
        if (optres[1] == None or optres[1].cost > r):
            optres[1] = res
        return chi2m(res.x)
            
    def update_awp(mode, r0, idx, sar, rar, har):
        trej = 0.0
        nrej = 0
        wi = arange(idx[ii], idx[ii+1])
        for ip in wi:
            for jp in range(idx[ii],ip):
                if mode == 1 and (ds[ii].zai[ip-idx[ii]] or ds[ii].zai[jp-idx[ii]]):
                    continue
                imin = mp0[ip]
                imax = mp1[ip]
                mpi = mp[ip]
                jmin = mp0[jp]
                jmax = mp1[jp]
                mpj = mp[jp]
                sigma = sar[ip-idx[0],jp-idx[0]]
                xmax = min(imax-mpi, mpj-jmin)
                xmin = max(imin-mpi, mpj-jmax)
                xp0 = xmin/sigma
                xp1 = xmax/sigma
                (rn,yp,y0,y1) = rand_cg(xp0, xp1)
                dp = sigma*rn
                mp[ip] += dp
                mp[jp] -= dp
                xmaxi = min(imax-mp[ip], mp[jp]-jmin)
                xmini = max(imin-mp[ip], mp[jp]-jmax)
                xpi0 = xmini/sigma
                xpi1 = xmaxi/sigma
                yi0 = cnorm(xpi0)
                yi1 = cnorm(xpi1)
                r = lnlikely(ip)
                dr = r + log(y1-y0)-log(yi1-yi0)
                rej = 0
                rp = 0.0
                if (dr < r0):
                    rp = 1-exp(dr-r0)
                    if (rand() < rp):
                        mp[ip] = mpi
                        mp[jp] = mpj
                        rej=1
                if not rej:
                    r0 = r
                    hmp[i1,ip] = mp[ip]
                    hmp[i1,jp] = mp[jp]
                    har[i, ip-idx[0],jp-idx[0]] = dp
                trej += rp
                nrej += 1
                rar[i,ip-idx[0],jp-idx[0]] = rp
        return (r0,trej,nrej)
    
    r0 = lnlikely(0)
    frej = zeros((imp,np),dtype=int8)
    rrej = zeros((imp,np))
    rrej[:,:] = -1.0
    arej = zeros((imp,na,na))
    wrej = zeros((imp,nw,nw))
    arej[:,:,:] = -1.0
    wrej[:,:,:] = -1.0
    hda = zeros((imp,na,na))
    hdw = zeros((imp,nw,nw))
    sda = zeros((na,na))
    sdw = zeros((nw,nw))
    fda = zeros((na,na))
    fdw = zeros((nw,nw))
    fmp = zeros(np)
    fmp[:] = 1.0
    fda[:,:] = 1.0
    fdw[:,:] = 1.0
    for ip in range(ia[0],ia[-1]):
        for jp in range(ia[0],ia[-1]):
            sda[ip-ia[0],jp-ia[0]] = sqrt(smp[ip]*smp[jp])
    for ip in range(iw[0],iw[-1]):
        for jp in range(iw[0],iw[-1]):
            sdw[ip-iw[0],jp-iw[0]] = sqrt(smp[ip]*smp[jp])
    hmp = zeros((imp,np))
    ene = zeros(imp)
    hmp[0] = mp
    trej = 0.0
    ene[0] = r0
    if nburn <= 0:
        nburn = 0.25
    if nburn < 1:
        nburn = int32(nburn*imp)
    if nopt > 0 and nopt < 1:
        nopt = int32(nopt*imp)
    if sopt <= 0:
        sopt = nopt+1
    if sopt > 0 and sopt < 1:
        sopt = int32(sopt*imp)
    ttr = 0.0
    mpa = zeros(np)
    mpe = zeros(np)    
    if len(sav) == 2:
        fsav = sav[0]
        nsav = sav[1]
    else:
        fsav = None
        nsav = 0
    xx = zeros(nde1)
    xx0 = zeros(np)
    for i in range(1,imp):
        i1 = i-1
        if (nopt > 0 and i%nopt == 0 and i <= sopt) or os.path.isfile(fopt):
            print('optimizing ...')
            if os.path.isfile(fopt):
                os.system('rm -f %s'%fopt)
            xx0[:] = mp
            xx[:] = mp[:nde1]
            res = optimize.least_squares(chi2n, xx, jac='3-point', bounds=(mp0[:nde1],mp1[:nde1]), ftol=1e-1/nd, xtol=1e-2)
            optres[0] = res
            mp[:nde1] = res.x
            mp[nde1:] = optres[1].x
            for ii in range(ni):
                mp[ia[ii]:ia[ii+1]] /= sum(mp[ia[ii]:ia[ii+1]])
                mp[iw[ii]:iw[ii+1]] /= sum(mp[iw[ii]:iw[ii+1]])
            r = lnlikely(0)
            if r > r0:
                r0 = r
                hmp[i1] = mp
                pp = [i, trej, ttr, r0, r0-ene[i1], mp[0], smp[0]]
                for ii in range(3):
                    if ii < ni:
                        pp.append(mp[iid[ii]])
                        pp.append(smp[iid[ii]])                
                        pp.append(mp[ia[ii]])
                        pp.append(smp[ia[ii]])
                        pp.append(mp[iw[ii]])
                        pp.append(smp[iw[ii]])
                    else:
                        pp.append(0.0)
                        pp.append(0.0)
                        pp.append(0.0)
                        pp.append(0.0)
                        pp.append(0.0)
                        pp.append(0.0)        
                print('opt: %6d %10.3E %10.3E %15.8E %10.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E'%tuple(pp))
            else:
                mp[:] = xx0
                r1 = lnlikely(0)
                print('optimzie fail: %g %g %g'%(r0, r, r1))
        trej = 0.0
        nrej = 0
        for ip in range(nde2):
            xp0 = (mp0[ip]-hmp[i1,ip])/smp[ip]
            xp1 = (mp1[ip]-hmp[i1,ip])/smp[ip]
            (rn,yp,y0,y1) = rand_cg(xp0, xp1)            
            mp[ip] = hmp[i1,ip] + rn*smp[ip]
            yi0 = cnorm((mp0[ip]-mp[ip])/smp[ip])
            yi1 = cnorm((mp1[ip]-mp[ip])/smp[ip])
            r = lnlikely(ip)
            dr = r + log(y1-y0)-log(yi1-yi0)
            rej = 0
            rp = 0.0
            if (dr < r0):
                rp = 1-exp(dr-r0)
                if (rand() < rp):
                    mp[ip] = hmp[i1,ip]
                    if (ip < nde1):                        
                        for ii in range(ni):
                            if ip < nsig or (ip >= ide0[ii] and ip < ide1[ii]):
                                rs[ii] = rs0[ii]
                                if len(wde[ii][0]) > 0:
                                    ds[ii].de[wde[ii]] = de0[ii]
                    rej = 1

            frej[i,ip] = rej
            if not rej:
                r0 = r
            rrej[i,ip] = rp
            trej += rp
            nrej += 1
            
        for ii in range(ni):
            if len(fixnd) > 0 and fixnd[ii] >= 0:
                continue
            (r0,treja,nreja) = update_awp(0, r0, ia, sda, arej, hda)
            nrej += nreja
            trej += treja
            
        for ii in range(ni):
            if len(fixld) > 0 and fixld[ii] >= 0:
                continue
            (r0, treja, nreja) = update_awp(1, r0, iw, sdw, wrej, hdw)
            nrej += nreja
            trej += treja
            
        hmp[i] = mp
        ene[i] = r0
        if i >= 50 and i <= nburn and i%25 == 0:
            im = i-25
            im0 = max(i-100, 10)
            for ip in range(np):
                if ip < nde2:
                    fa = mean(rrej[im:i+1,ip])
                    fa = fmp[ip]*((1-fa)/racc)**2
                    fa = min(fa, 1e2)
                    fa = max(fa, 1e-2)
                    fa = 0.25*fmp[ip]+0.75*fa
                else:
                    fa = 1.0
                xst = fa*std(hmp[im0:i+1,ip])
                fmp[ip] = fa
                if xst > 0:
                    smp[ip] = xst
            for ii in range(ni):
                for ip in range(ia[ii], ia[ii+1]):
                    for jp in range(ia[ii], ip+1):
                        fa = mean(arej[im:i+1,ip-ia[0],jp-ia[0]])
                        fa = fda[ip-ia[0],jp-ia[0]]*((1-fa)/racc)**2
                        fa = min(fa, 1e2)
                        fa = max(fa, 1e-2)
                        fa = 0.25*fda[ip-ia[0],jp-ia[0]]+0.75*fa
                        xst = fa*std(hda[im0:i+1,ip-ia[0],jp-ia[0]])
                        fda[ip-ia[0],jp-ia[0]] = fa
                        if xst > 0:
                            sda[ip-ia[0],jp-ia[0]] = xst                        
                for ip in range(iw[ii], iw[ii+1]):
                    for jp in range(iw[ii], ip+1):
                        fa = mean(wrej[im:i+1,ip-iw[0],jp-iw[0]])
                        fa = fdw[ip-iw[0],jp-iw[0]]*((1-fa)/racc)**2
                        fa = min(fa, 1e2)
                        fa = max(fa, 1e-2)
                        fa = 0.25*fdw[ip-iw[0],jp-iw[0]]+0.75*fa
                        xst = fa*std(hdw[im0:i+1,ip-iw[0],jp-iw[0]])
                        fdw[ip-iw[0],jp-iw[0]] = fa
                        if xst > 0:
                            sdw[ip-iw[0],jp-iw[0]] = xst
                        
        trej /= nrej
        ttr = (ttr*(i-1)+trej)/i
        wr = where(rrej[i] >= 0)
        war = where(arej[i] >= 0)
        wwr = where(wrej[i] >= 0)
        arm = 0.0
        rrm = 0.0
        wrm = 0.0
        if len(war[0]) > 0:
            arm = mean(arej[i,war[0],war[1]])
        if len(wr[0]) > 0:
            rrm = mean(rrej[i,wr[0]])
        if len(wwr[0]) > 0:
            wrm = mean(wrej[i,wwr[0],wwr[1]])
        pp = [i, trej, ttr, rrm, arm, wrm, 
              r0, r0-ene[i1], mp[0], smp[0], fmp[0], rrej[i,0], mean(rrej[max(0,i-25):i+1,0])]
        for ii in range(3):
            if ii < ni:
                pp.append(mp[iid[ii]])
                pp.append(smp[iid[ii]])
                pp.append(mp[ia[ii]])
                if ia[ii+1] > ia[ii]+1:
                    pp.append(sda[ia[ii]-ia[0]+1,ia[ii]-ia[0]])
                else:
                    pp.append(0.0)
                pp.append(mp[iw[ii]])
                if (iw[ii+1] > iw[ii]+1):
                    pp.append(sdw[iw[ii]-iw[0]+1,iw[ii]-iw[0]])
                else:
                    pp.append(0.0)
            else:
                pp.append(0.0)
                pp.append(0.0)
                pp.append(0.0)
                pp.append(0.0)
                pp.append(0.0)
                pp.append(0.0)
        print('imc: %6d %7.1E %7.1E %7.1E %7.1E %7.1E %12.5E %8.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E'%tuple(pp))
        if i == imp-1 or (nsav > 0 and (i+1)%nsav == 0):
            print('pickling: %s'%(fsav))
            if nsav > 0:
                fs = open(fsav, 'w')
            for ii in range(ni):
                if (len(fixnd) > 0):
                    ij = fixnd[ii]
                    if ij >= 0 and ij < ni:
                        hmp[:,ia[ii]:ia[ii+1]] = hmp[:,ia[ij]:ia[ij+1]]
                if (len(fixld) > 0):
                    ij = fixld[ii]
                    if ij >= 0 and ij < ni:
                        hmp[:,iw[ii]:iw[ii+1]] = hmp[:,iw[ij]:iw[ij+1]]
            zi = FitMCMC(i+1, sp, y.copy(), mpa, mpe, r, ds, rs, ap, hmp, ene, nde, ide0, ide1, iid, ia, iw, frej, rrej)
            mcmc_avg(zi, i/2)
            if nsav > 0:
                pickle.dump(zi, fs)
                fs.close()   
    return zi

def rebin_spec(elo, ehi, yd, ye, nr):
    nd = len(elo)
    n1 = nd/nr
    elo1 = zeros(n1)
    ehi1 = zeros(n1)
    yd1 = zeros(n1)
    ye1 = zeros(n1)
    for i in range(n1):
        ir = i*nr
        elo1[i] = elo[ir]
        ehi1[i] = ehi[ir+nr-1]
        yd1[i] = sum(yd[ir:ir+nr])
        ye1[i] = sqrt(sum(ye[ir:ir+nr]**2))
    return (elo1,ehi1,yd1,ye1)

def mod_spec(z, df, ws, stype, er=[]):
    s = read_spec(df, stype, er)
    zm = FitMCMC(z.imp, s, s.yc.copy(), z.mpa, z.mpe, z.ra, z.ds, z.rs, z.ap, z.hmp, z.ene, z.nde, z.ide, z.iid, z.ia, z.iw, z.frej, z.rrej)
    for i in range(len(zm.ds)):
        if i < len(ws):
            iws = ws[i]
        else:
            iws = 0
        zm.ds[i] = ion_data(z.ds[i].z, z.ds[i].k, z.ds[i].ns, iws, elo[0], ehi[-1])
    mcmc_avg(zm, zm.imp/2, zm.imp)
    return zm

def fit_spec(df, z, ks, ns, ws, sig, eth, stype, er=[], nmc=5000, fixld=[], fixnd=[], racc=0.35, kmin=0, kmax=-1,
             wb=[], wsig=[], sav=[], mps=[], nburn=0.25, nopt=0, sopt=0, fopt='', yb=1e-2, ecf='', ddir='data', sdir='spec'):
    s = read_spec(df, stype, er)
    sig = atleast_1d(sig)
    kmin = atleast_1d(kmin)
    kmax = atleast_1d(kmax)
    ds = []
    for i in range(len(ks)):
        if i < len(ws):
            iws = ws[i]
        else:
            iws = 0
        if i < len(kmin):
            k0 = kmin[i]
        else:
            k0 = kmin[-1]
        if i < len(kmax):
            k1 = kmax[i]
        else:
            k1 = kmax[-1]
        if i < len(ks):
            ki = ks[i]
        else:
            ki = ks[-1]
        if i < len(ns):
            nsi = ns[i]
        else:
            nsi = ns[-1]
        di = ion_data(z, ki, nsi, iws, s.elo[0], s.ehi[-1], kmin=k0, kmax=k1, ddir=ddir, sdir=sdir)
        ds.append(di)

    z = mcmc_spec(ds, s, sig, eth, nmc, fixld, fixnd, racc, wb, wsig, sav, mps, nburn, nopt, sopt, fopt, yb, ecf)
    return z

def plot_ldist(z, op=0, sav=''):
    ds = z.ds
    if not op:
        clf()
    ymin = 1e31
    ymax = -1e31
    for i in range(len(ds)):
        if ds[i].k == 0:
            continue
        ymin0 = min(ds[i].wk-ds[i].we)
        ymax0 = max(ds[i].wk+ds[i].we)
        ymin = min(ymin0, ymin)
        ymax = max(ymax0, ymax)
    dy = 0.025*(ymax-ymin)
    ymin -= dy
    ymax += dy
    ylim(ymin, ymax)
    ni = len(ds)
    for i in range(ni):
        d = ds[i]
        if d.k == 0:
            continue
        errorbar(d.xk-(0.2/ni)*(i-ni/2), d.wk, yerr=d.we, marker='o', capsize=3)
    xlabel('L')
    ylabel('Fraction')
    if sav != '':
        savefig(sav)
    
def plot_idist(z, op=0, sav=''):
    if not op:
        clf()
    x0 = arange(len(z.ds))
    y0 = z.mpa[z.iid]/sum(z.mpa[z.iid])
    e0 = z.mpe[z.iid]/sum(z.mpa[z.iid])

    ymin = min(y0-e0)
    ymax = max(y0+e0)
    dy = 0.025*(ymax-ymin)
    ymin -= dy
    ymax += dy
    ylim(ymin, ymax)
    errorbar(x0, y0, yerr=e0, marker='o', capsize=3)

    xlabel('Ion')
    ylabel('Fraction')
    if sav != '':
        savefig(sav)
    
def plot_ndist(z, op=0, sav=''):
    ds = z.ds
    if not op:
        clf()
    ymin = 1e31
    ymax = -1e31
    ni = len(ds)
    for i in range(len(ds)):
        d = ds[i]
        y0 = d.an[:-1]
        e0 = d.ae[:-1]
        ymin0 = min(y0-e0)
        ymax0 = max(y0+e0)
        ymin = min(ymin0, ymin)
        ymax = max(ymax0, ymax)
    dy = 0.025*(ymax-ymin)
    ymin -= dy
    ymax += dy
    ylim(ymin, ymax)
    for i in range(len(ds)):
        d = ds[i]
        x0 = array(d.ns)-(0.2/ni)*(i-ni/2)
        y0 = d.an[:-1]
        e0 = d.ae[:-1]
        errorbar(x0, y0, yerr=e0, marker='o', capsize=3)

    xlabel('N')
    ylabel('Fraction')
    if sav != '':
        savefig(sav)

def plot_ink(z, i, op=0, col=0, xoffset=0, sav=''):
    cols = ['k', 'b','g','r','c','m','y']
    if not op:
        clf()
    if col < 0:
        col=i
    if len(z.ds[i].ns) > 1:
        errorbar(array(z.ds[i].ns)+0.1+xoffset, z.ds[i].an[:-1], yerr=z.ds[i].ae[:-1], capsize=3, marker='o', color=cols[col%len(cols)])
    errorbar(z.ds[i].xk-0.1+xoffset, z.ds[i].wk, yerr=z.ds[i].we, capsize=3, marker='o', color=cols[col%len(cols)])
    xlabel('N/L')
    ylabel('Fraction')
    if sav != '':
        savefig(sav)

def plot_snk(z, sav=''):
    plot_ink(z, 0)    
    plot_ink(z, 1, op=1, col=-1)
    plot_ink(z, 2, op=1, col=-2, sav=sav)
    labs = ['H-Like', 'He-Like Singlet', 'He-Like Triplet']
    legend(labs)
    
def plot_spec(z, res=0, op=0, ylog=0, sav='', ymax=0):
    fm = z.sp
    if not op:
        clf()
    if res == 0:
        if ymax > 0:
            ylim(-ymax*0.01, ymax)
        if ylog > 0:
            ymin = max(fm.yd)*ylog
            semilogy(fm.em, ymin+fm.yd)
            semilogy(fm.em, ymin+z.ym/fm.eff)
        else:
            plot(fm.em, fm.yd)
            plot(fm.em, z.ym/fm.eff)
        ylabel('Intensity')
    else:
        r = fm.yd-z.ym/fm.eff
        w = where(fm.ye > 0)
        r[w] /= fm.ye[w]
        w = where(fm.ye <= 0)
        r[w] = 0
        plot(fm.em, r)
        ylabel('Residual')
    xlabel('Energy (eV)')
    if sav != '':
        savefig(sav)
        
def plot_ene(z, i0=0.25, op=0):
    if not op:
        clf()
    if i0 < 1:
        i0 = int32(i0*z.imp)
    plot(arange(i0, z.imp), z.ene[i0:z.imp])
    
def plot_param(z, ip, i0=0.25, op=0):
    if not op:
        clf()
    if i0 < 1:
        i0 = int32(i0*z.imp)
    plot(arange(i0, z.imp), z.hmp[i0:z.imp,ip])

def sum_cx(d):    
    ni = len(d.ns)
    im = len(d.ad[0,0])
    c = zeros(im)
    e = zeros(im)
    for i in range(ni):
        a = transpose(d.ad[i])
        x = d.anr[-1]*d.anr[i]
        y = matmul(a, d.wk)
        c[:] += y*x
        e[:] += matmul(a*a, d.we*d.we)*(x*x)
        e[:] += y*y*((d.anr[-1]*d.ae[i])**2+((d.ae[-1]*d.anr[i])**2))
    return (c,e)

def sum_tcx(z, k=0):
    c = 0.0
    e = 0.0
    if k == 0:
        k = z.ds[0].k
    for d in z.ds:
        if d.k != k:
            continue
        ci, ei = sum_cx(d)
        c += ci
        e += ei
    return (c,e)

def sel_icx(z, ic, k = 0):
    if k == 0:
        k = z.ds[0].k
    c,e = sum_tcx(z, k)
    tc = sum(c)
    for d in z.ds:
        if d.k == k:
            break
    n = len(c)
    a = d.rd.n[n+ic]
    if d.k > 1:
        for i in range(len(c)):
            if not d.rd.n[i].startswith(a):
                c[i] = 0.0
                e[i] = 0.0
    return (c/tc,sqrt(e)/tc,tc)
    
def sum_icx(z, k = 0):
    if k == 0:
        k = z.ds[0].k
    c,e = sum_tcx(z, k)
    tc = sum(c)
    for d in z.ds:
        if d.k == k:
            break
    n = len(c)
    ic = d.rd.n[n:]
    nc = len(ic)
    r = zeros(nc)
    rs = zeros(nc)
    for i in range(len(c)):
        for j in range(nc):
            if z.ds[0].rd.n[i].startswith(ic[j]):
                r[j] += c[i]
                rs[j] += e[i]
    return (r/tc,sqrt(rs)/tc,tc)

def sum_tnk(z, k=0):
    if k == 0:
        k = z.ds[0].k
    c,e = sum_tcx(z, k)
    for d in z.ds:
        if d.k == k:
            break
    v = d.rd.v[:len(c)]
    vn = v/100
    vk = v%100
    nmax = nanmax(vn)
    r = zeros((nmax,nmax))
    s = zeros((nmax,nmax))
    for n in range(nmax):
        for j in range(nmax):
            w = where(logical_and(vn==n+1, vk==j))
            r[n,j] += sum(c[w])
            s[n,j] += sum(e[w])
    return (r,s)

def avg_tnk(z, k=0, m=0.25):
    if k == 0:
        k = z.ds[0].k
    for d in z.ds:
        if d.k == k:
            break
    im = len(d.ad[0,0])
    v = d.rd.v[:im]
    vn = v/100
    vk = v%100
    nmax = nanmax(vn)
    r = zeros((nmax,nmax))
    s = zeros((nmax,nmax))
    if m < 1:
        m0 = int32(z.imp*m)
    else:
        m0 = int32(m)    
    for i in range(z.imp-m0, z.imp):
        mcmc_cmp(z, i)
        c,e = sum_tnk(z, k)
        r += c
        s += c*c
    r /= m0
    s /= m0
    s = s - r*r
    mcmc_avg(z, z.imp/2)
    return (r,s)

def plot_tnk(z, op=0, nmin=8, nmax=12, kmax=6, pn0=4, pn1=12,
             k=0, md=0, sav=''):
    if not op:
        clf()
    if k == 0:
        k = z.ds[0].k
    cols = ['k', 'b','g','r','c','m','y']
    if md == 0:
        r,s = sum_tnk(z, k)
    else:
        r,s = avg_tnk(z, k)
    tr = sum(r, axis=1)
    ts = sqrt(sum(s, axis=1))
    n = len(r[0])
    y = [r[i,:]/max(1e-30,tr[i]) for i in range(n)]
    ye = [sqrt(s[i,:])/max(1e-30,tr[i]) for i in range(n)]
    xn = arange(n)+1
    xk = arange(n)
    trs = sum(tr)
    pn1 = min(pn1, n+1)
    idx = range(pn0-1,pn1)
    errorbar(xn[idx], (tr/trs)[idx], yerr=(ts/trs)[idx], capsize=3, marker='o', color=cols[0])
    labs = ['N-dist']
    for i in range(nmin, nmax+1):
        errorbar(xk[:kmax+1], y[i-1][:kmax+1], yerr=ye[i][:kmax+1], capsize=3, marker='o', color=cols[1+(i-nmin)%(len(cols)-1)])
        labs.append('L-dist N=%d'%i)
    legend(labs)
    xlabel('N/L')
    ylabel('Fraction')
    if sav != '':
        savefig(sav)
    
def plot_icx(z, op=0, sav='', k=0, nmin=6, nmax=12, xr=[120, 660], yr=[1e-3, 0.25]):
    if not op:
        clf()
    if k == 0:
        k = z.ds[0].k
    for d in z.ds:
        if d.k == k:
            break
    cols = ['k', 'b','g','r','c','m','y']
    c0,e0,tc = sel_icx(z, 0, k)
    if d.k == 10:
        c1,e1,tc = sel_icx(z, 1, k)
        c2,e2,tc = sel_icx(z, 2, k)
    xlim(xr[0], xr[1])
    ylim(yr[0], yr[1])
    semilogy(c0+1e-8, color=cols[0])
    v = d.rd.v/100
    if d.k == 10:
        semilogy(c1+1e-8, marker='o', color=cols[1])
        semilogy(c2+1e-8, marker='o', color=cols[2])
        labs = ['2p+', '2p-', '2s+']
        legend(labs)
        w1 = where(c1 > 5e-3)
        w2 = where(c2 > 5e-3)
        w1 = w1[0][0]
        w2 = w2[0][0]
        n1 = d.rd.n[w1][-8:-5]
        n2 = d.rd.n[w2][-8:-5]
        n1 = '%s J=%d'%(n1, d.rd.j[w1]/2)
        n2 = '%s J=%d'%(n2, d.rd.j[w2]/2)
        text(w1, c1[w1]*1.1, n1, color=cols[1])
        text(w2, c2[w2]*1.1, n2, color=cols[2])
    for n in range(nmin, nmax+1):
        w = where(v == n)
        w = w[0]
        i = argmax(c0[w])
        text(w[i], 1.1*c0[w[i]], 'N=%d'%n, horizontalalignment='center')
    xlabel('Level Index')
    ylabel('Relative Cross Section')
    if sav != '':
        savefig(sav)
