from numpy import *
from pylab import *
import matplotlib.pyplot as plt
import os,sys
import os.path
from collections import namedtuple
from scipy import special, integrate, interpolate, optimize, stats, linalg
from pfac import fac
from pfac import rfac
from pfac import crm
import ebit
import pickle
import time
import mcpk, mcmc

"""cxfit uses MCMC algorithm to fit the charge exchange X-ray spectra to derive n/l distribution of the capture cross sections
"""

RadCas = namedtuple('RadCas', ['rd', 'r', 'ir0', 'ir1', 'egy', 'wa', 'ai', 'trm', 'tri', 'z', 'k', 'ide', 'de', 'im', 'nt', 'im1', 'na'])
"""RadCas contains the radiative cascade model. it is not used in the fit directly, but can be used to examine the cascade spectra of individual captures
rd: the FAC energy level obj read in via pfac.rfac.read_en()
r: the radiative transition rates read in from *.r1 file dumped from fac.crm.DumpRate
ir0: lower level indices of the lines
ir1: upper level indices of the lines
egy: energy of the lines
wa: level indices of the capture states
ai: cascade matrix. ai[i,j] is the intensity of the i-th line from the capture into wa[j] state.
trm: transition rate matrix. trm[i,j] = -A_{i,j} if i<j, trm[i,i]=sum_i(A_{i,j})
tri: inverse of trm
z: atomic number
k: number of electrons
ide,de: optional energy corrections to the lines
im: total number of levels
nt: number of lines
im1: im-1
na: number of capture states, len(wa)
"""

SpecData = namedtuple('SpecData', ['fns', 'stype', 'er', 'elo', 'ehi', 'em', 'yc', 'yd', 'eff', 'ye', 'xm'])
"""Spectra Data structure
fns: file name the data is read from
stype: type of the data file
er: energy range
elo: low edge of the energy grid
ehi: high edge of the energy grid
em: mid of the energy bin
yc: counts per bin
yd: transmission corrected count
eff: transmission coefficients
ye: uncertainty on yd
"""

Response = namedtuple('Response', ['sig', 'es', 'eip', 'bn', 'ar', 'aj'])
"""Spectral response function
sig: response function parameters. List of up to 5 elements.
     sig[0] the standard deviation of the Gaussian profile
     sig[-1] is the energy dependence of the width. sigma=sig[0]*sqrt(1+sig[-1]*(e-em[0])/1e3)
     len(sig)>2, the response function is a mixture of a Gaussian and a Compton scattering wing.
     R(x) = (G(x) + S(x)*sig[2])/(1+sig[2])
     S(x) = bn*exp(x*sig[1])*erfc((x+sig[1])/sqrt(2)*sig[3])
     where x = (e-e0)/sigma
es:  energy scale calibration, x = es[1] + es[2]*(e-es[0]) + es[3]*(e-es[0])**2 + ...
bn:  normalization constant of the S(x) function above.
ar:  ar[:,j] is the response of j-th line, R((e-egy[j])
aj:  response of capture n/l distribution. aj[n,:,l] is the spectra of capture to n/l orbital
"""

IonData = namedtuple('IonData',
                     ['ddir', 'sdir', 'ps0', 'ds0', 'rd', 'r',
                      'ir0', 'ir1', 'idx', 'egy', 'ai', 'ev', 'av',
                      'ad', 'zai','z', 'k', 'ns', 'nm', 'kmin', 'kmax',
                      'an', 'ae', 'anr', 'wk0', 'wk', 'we', 'swk', 'swe',
                      'xk', 'ide', 'de', 'ws', 'emin', 'emax',
                      'df', 'lzd', 'tgt', 'lzf', 'lznd', 'ga'])
"""Ion radiative cascade data container
ddir: data directory containing the FAC atomic data
sdir: spectral directory containing the cascade mode from the crm run
ps0: file prefix of the spectral model files
ds0: file prefix of the FAC data files
rd: the FAC energy level obj read in via pfac.rfac.read_en()
r: the radiative transition rates read in from *.r1 file dumped from fac.crm.DumpRate
ir0: lower level indices of the lines
ir1: upper level indices of the lines
idx: idx[i,j] is the index of the j->i transition in the r array.
egy: energy of the lines
ai: ai[n,i,l] is the intensity of i-th line for capture in to n/l orbital
ad: ad[n,l,i] is the capture cross section into i-th state for a n/l capture
zai: zai[l]=1 means a capture to l orbital does not result in any line for this ion and ws
z: atomic number
k: number of electrons
ns: n-array of the capture
nm: number of l orbitals
kmin: minimum l
kmax: maximum l
an: n-distribution, accounting for wk normalization
ae: error in n-distribution
anr: raw n-distribution
wk0: theoretical low-energy l-distribution
wk: l-distribution
we: error in l-distribution
swk: not used
swe: not used
xk: l-array
ide: index of lines need energy correction
de: energy correction of lines
ws: 2J+1/2S+1 of the ion for spin/j split
emin: minimum energy of the lines
emax: maximum energy of the lines
"""

FitMCMC = namedtuple('FitMCMC', ['imp', 'sp', 'ym', 'mpa', 'mpe', 'ra', 'ds', 'rs', 'ap', 'hmp', 'ene','nde','ide0','ide1','iid', 'ia','iw','ib','bf','frej','rrej', 'ierr', 'fixnd', 'fixld'])
"""Results of MCMC fit to the cx spectra
imp: number mcmc iterations
sp: spectral data
ym: fitted spectrum
mpa: average parameters
mpe: standard deviation of parameters
ra: log of likelyhood in the last iteration
ds: IonData array for included ions
rs: response function for each ion
ap: ap[i,:,l] is the model spetra of i-th ion with capture into l orbital
hmp: hmp[i,:] is the parameter of the i-th iteration
ene: ene[i] is the log of likelyhood of the i-th iteration
nde: number of lines needs energy correction
ide0,ide1: ide0[i]:ide1[i] is the parameter index of the energy correction of i-th ion
iid: parameter index of the total ion normalizaiton
ia: ia[i]:ia[i+1] is the parameter index of the n-dist parameters of the i-th ion
iw: iw[i]:iw[i+1] is the parameter index of the l-dist parameters of the i-th ion
ib: parameter index of the background parameters.
bf: background function
frej: reject probability scal factor
rrej: reject probability array
ierr: (iea,iy) iea is the theoretical error of the model spectra, iy is the resulting errors on the counts
fixnd: flags to fix n-dist of any ion
fixld: flags to fix l-dist of any ion
"""

def rcs(f):
    r = loadtxt(f,ndmin=2)
    if len(r) > 0:
        return transpose(r)
    else:
        return r

def eb_tr(p, ddir='data', cnv=1):
    h0,d = rfac.read_enf('%s/%s.en'%(ddir,p))
    h1,r = rfac.read_trf('%s/%s.tr'%(ddir,p))
    n = d[-1]['ilev'][-1]+1
    m = max(d[-1]['pbasis'])+1
    egy = zeros(n)
    for i in range(len(d)):
        egy[d[i]['ilev']] = d[i]['energy']
    tr0 = zeros((n,n))
    for i in range(len(r)):
        i0 = r[i]['lower_index']
        i1 = r[i]['upper_index']
        tr0[i0,i1] += r[i]['rate']
    if cnv == 0:
        return egy,tr0,d,r
    tr1 = zeros((m,m))
    ntr1 = zeros(m)
    b = loadtxt('%s/%s.bas'%(ddir,p), unpack=1, ndmin=2)
    c = []
    pb = []
    mb = []
    for i in range(len(d)):
        i0 = d[i]['ilev'][0]
        i1 = d[i]['ilev'][-1]
        ni = i1-i0+1
        w = where((b[0] >= i0)&(b[0] <= i1))
        ix = int32(b[0][w]-min(b[0][w]))*1000000 + int32(b[1][w]-min(b[1][w]))*100 + max(int32(b[2][w]))+int32(b[2][w])
        sx = argsort(ix)
        ci = transpose((b[3][w][sx]).reshape((ni,ni)))
        c.append(ci)
        pb.append(int32(b[1][w][sx]))
        mb.append(int32(b[2][w][sx]))
    for i in range(len(d)):
        ci = c[i]
        pbi = pb[i]
        mbi = mb[i]
        i0 = d[i]['ilev'][0]
        i1 = d[i]['ilev'][-1]
        ni = i1-i0+1
        for j in range(i,len(d)):
            cj = c[j]
            pbj = pb[j]
            mbj = mb[j]
            j0 = d[j]['ilev'][0]
            j1 = d[j]['ilev'][-1]
            nj = j1-j0+1
            ix = repeat(arange(i0,i1+1),nj).reshape((ni,nj))
            jx = repeat(arange(j0,j1+1).reshape((1,nj)),ni,axis=0)
            tij = matmul(ci*ci,matmul(tr0[ix,jx],transpose(cj*cj)))
            for jj in range(nj):
                jpb = pbj[jj]
                ntr1[jpb] = max(ntr1[jpb],mbj[jj])
                for ii in range(ni):
                    ipb = pbi[ii]
                    if ipb < jpb:
                        tr1[ipb,jpb] += tij[ii,jj]
    for j in range(m):
        if ntr1[j] >= 0:
            tr1[:,j] /= 1+ntr1[j]

    return egy,tr0,tr1,d,r

def eb_cas(z, k, p, emin, emax, ddir='data'):
    a = fac.ATOMICSYMBOL[z]
    f1 = '%s%02d%sa'%(a,k,p)
    es, tr, d, r = eb_tr(f1, ddir=ddir, cnv=0)
    m = len(tr[0])
    w = where(tr > 0)
    i0 = w[0]
    i1 = w[1]
    egy = es[i1]-es[i0]
    tr = -tr
    for i in range(m):
        tr[i,i]=-sum(tr[:i,i])
    if k == 1:
        tpr = crm.TwoPhoton(z, 0)
        tr[4,4] += tpr
        tr[5,5] += tpr
    elif k == 2:
        tpr = crm.TwoPhoton(z, 1)
        tr[8,8] += tpr
    tri = linalg.pinv2(tr[1:,1:], cond=1e-12)
    w = where((egy >= emin)&(egy <= emax))
    egy = egy[w]
    i0 = i0[w]
    i1 = i1[w]
    ai = matmul(diag(-tr[i0,i1]),tri[i1-1])
    return (egy, i0, i1, tri, tr, ai)

def cs_cx(z, k, n, nk='', sdir='spec'):
    a = fac.ATOMICSYMBOL[z]    
    f = sdir+'/%s%02d%sa.r7'%(a,k,nk)
    r1 = loadtxt(f, unpack=1, ndmin=2)
    i1 = int32(r1[2])
    n1 = 1+max(i1)
    if n1 < n:
        n1 = n
    c1 = zeros(n1)
    c1[i1] = r1[3]
    return c1

def eb_cx(z, k, ip0, np, nk='', ddir='data', sdir='spec'):
    a = fac.ATOMICSYMBOL[z]
    p = '%s%02d'%(a,k)
    f = ddir+'/%sa.en'%p
    r = rfac.FLEV(f)
    w = where(abs(r.j-r.j[0])%2 == 1)
    w = w[0]
    c1 = cs_cx(z, k, w, nk=nk, sdir=sdir)
    #w = where(r.v%100 > 2)
    #c1[w] = 0.0
    for ip in range(ip0, ip0+np):
        f0 = ddir+'/%sF%02da.bas'%(p,ip)
        print(f0)
        r0 = loadtxt(f0, unpack=1, ndmin=2)
        h1,r1 = rfac.read_enf('%s/%sF%02da.en'%(ddir,p,ip))
    
        i = int32(r0[0])
        j = int32(r0[1])
        m = int32(r0[2])
        b = r0[3]
        n = 1+max(i)
        c = zeros(n)
        c2 = zeros(len(c1))
        for ib in range(len(r1)):
            i0 = r1[ib]['ilev'][0]
            i1 = r1[ib]['ilev'][-1]
            ni = i1-i0+1
            w = where((i >= i0)&(i <= i1))
            ix = (i[w]-min(i[w]))*1000000 + (j[w]-min(j[w]))*100 + (max(m[w])+m[w])
            sx = argsort(ix)
            ci = (b[w][sx]).reshape((ni,ni))
            ct = transpose(ci)
            pb = j[w][sx]
            mb = m[w][sx]
            cm = c1[pb[:ni]]/(r.j[pb[:ni]]+1.0)
            c[i0:i1+1] = matmul(ci*ci, cm)
            ck = matmul(ct*ct, c[i0:i1+1])
            for t in range(ni):
                c2[pb[t]] += ck[t]
        if ip == ip0:
            ac0 = zeros((np,n))
            ac2 = zeros((np,len(c1)))
        ac0[ip-ip0] = c
        ac2[ip-ip0] = c2
    return ac0,c1,ac2

def rad_cas(z, k, emin, emax, nmin=0, nmax=100,
            ist='', ddir='data', sdir='spec', pf=''):
    """calcualte the radiative cascade spectra
    z: atomic number
    k: number of electron
    emin: min line energy
    emax: max line energy
    nmin: min n of capture
    nmax; max n of capture
    ist: initial state
    ddir: FAC data directory
    sdir: crm data directory
    """
    a = fac.ATOMICSYMBOL[z]
    ps0 = '%s/%s%02d'%(sdir, a, k)
    ds0 = '%s/%s%02d'%(ddir, a, k)
    rd = rfac.FLEV(ds0+'a.en')
    if pf != '':
        e0,t0,t1,dd,rr = eb_tr('%s%02d%sa'%(a,k,pf), ddir=ddir)
        im = len(t1[0])
        im1 = im-1
        trm = -t1[:im,:im]
        ir0 = repeat(arange(im),im).reshape((im,im))
        ir1 = repeat(arange(im).reshape((1,im)),im,axis=0)
        r = t0
    else:
        w = where(abs(rd.j-rd.j[0])%2 == 1)
        w = w[0]
        im = w[0]
        im1 = im-1
        r = rcs(ps0+'a.r1')
        ir0 = int32(r[2])
        ir1 = int32(r[1])
        trm = zeros((im,im))
        trm[ir0,ir1] = -r[3]
    
    for i in range(im):
        trm[i,i] = -sum(trm[:i,i])
    w = []
    if k == 1:
        tpr = crm.TwoPhoton(z, 0)
        w = where(rd.n == '2s+1(1)1')
    elif k == 2:
        tpr = crm.TwoPhoton(z, 1)
        w = where(rd.n == '1s+1(1)1.2s+1(1)0')
    elif k == 4:
        tpr = crm.TwoPhoton(z, 2)
        w = where(rd.n == '2s+1(1)1.2p-1(1)0')
    if len(w) == 1:        
        w = w[0]
        if len(w) == 1:
            w = w[0]
            trm[w,w] += tpr
    egy = rd.e[ir1] - rd.e[ir0]
    w = where(logical_and(egy >= emin, egy <= emax))
    ir0 = ir0[w]
    ir1 = ir1[w]
    egy = egy[w]
    nt = len(ir0)
    tri = linalg.pinv2(trm[1:,1:],cond=1e-12)
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

def eb_trm(z, k, nf=20, e0=2.0, e1=8.0, ddir='data'):
    a = fac.ATOMICSYMBOL[z]    
    es = array([10**(e0+i*(e1-e0)/(nf-1)) for i in range(nf)])

    tpr = 0.0
    rd = rfac.FLEV('%s/%s%02da.en'%(ddir,a,k))
    w = []
    if k == 1:
        tpr = crm.TwoPhoton(z, 0)
        w = where(rd.n == '2s+1(1)1')
    elif k == 2:
        tpr = crm.TwoPhoton(z, 1)
        w = where(rd.n == '1s+1(1)1.2s+1(1)0')
    elif k == 4:
        tpr = crm.TwoPhoton(z, 2)
        w = where(rd.n == '2s+1(1)1.2p-1(1)0')
    itp = -1
    if len(w) == 1:        
        w = w[0]
        if len(w) == 1:
            itp = w[0]
    for i in range(nf):
        p = '%s%02dF%02da'%(a,k,i)
        print(p)
        eg,t0,t1,d,r = eb_tr(p, ddir=ddir)
        n = len(t1[0])
        for j in range(n):
            t1[j,j] = -sum(t1[:j,j])
        t1[itp,itp] -= tpr
        if i == 0:
            trm = zeros((nf,n,n))
        trm[i] = -t1
    return es,trm

def interp_cs(es, cs, rx):
    ef = log(12.4e3*1e-8/(2*pi*137.0*(rx*0.53e-8)**2))
    loges = log(es)
    dloges = loges[1]-loges[0]        
    if (ef <= loges[0]):
        a = cs[0]
    elif (ef >= loges[-1]):
        a = cs[-1]
    else:
        j = int((ef-loges[0])/dloges)
        w = j+1
        f = (ef-loges[j])/dloges
        print([j,w,f])
        a = cs[j]*(1-f) + cs[w]*f
    return a

def lzrx(lzf):
    with open(lzf) as f:
        r = f.readlines(2000)
        d = loadtxt(lzf, unpack=1)
        rx = array([float(x) for x in r[11][11:-1].split()])
        rx = matmul(transpose(d[1:-1]),rx)/sum(d[1:-1],0)
        return d[0],rx
    
def int_rate(es, trm, y0i, ecx=0,
             mt=10.0,ntm=50000,method='Radau',
             rx=20.0, eu=10.0):
    if ecx == 1:
        n = len(y0i[0])
    else:
        n = len(y0i)
    i = arange(n)
    r0 = min(trm[0,i,i][1:])
    r1 = max(trm[-1,i,i][1:])
    t1 = mt/r0
    dt = 1/(2*mt*r1)
    nt = t1/dt
    if nt > ntm:
        nt = ntm
    dtx = (log(t1)-log(dt))/nt
    ta = zeros(nt)
    ta[1:] = dt*exp(arange(nt-1)*dtx)
    t1 = ta[-1]
    emax = 12.4e3*1e-8/(2*pi*137.0*(rx*0.53e-8)**2)
    xt = 1e10*ebit.e2v(eu,1.0)/(rx*0.53e-8)
    loges = log(es)
    dloges = loges[1]-loges[0]
    def fd(t,y):
        ef = log(emax/(1+(xt*t)**2))        
        if (ef <= loges[0]):
            a = trm[0]
        elif (ef >= loges[-1]):
            a = trm[-1]
        else:
            j = int((ef-loges[0])/dloges)
            w = j+1
            f = (ef-loges[j])/dloges
            a = trm[j]*(1-f) + trm[w]*f
        return -matmul(a,y)
    if ecx > 0:
        ef = log(emax)
        if (ef <= loges[0]):
            y0 = y0i[0]
        elif (ef >= loges[-1]):
            y0 = y0i[-1]
        else:
            j = int((ef-loges[0])/dloges)
            w = j+1
            f = (ef-loges[j])/dloges
            y0 = y0i[j]*(1-f) + y0i[w]*f
    else:
        y0 = y0i
    sf = integrate.solve_ivp(fd, [0.0,t1], y0, t_eval=ta, method=method)
    return sf

def int_lines(r,sf,es,trm,rx=20.0,eu=10.0):
    n = len(r.ir0)
    s = zeros(n)
    emax = 12.4e3*1e-8/(2*pi*137.0*(rx*0.53e-8)**2)
    xt = 1e10*ebit.e2v(eu,1.0)/(rx*0.53e-8)
    loges = log(es)
    dloges = loges[1]-loges[0]
    for i in range(1,len(sf.t)):
        t = sf.t[i]
        ef = log(emax/(1+(xt*t)**2))        
        if (ef <= loges[0]):
            a = trm[0]
        elif (ef >= loges[-1]):
            a = trm[-1]
        else:
            j = int((ef-loges[0])/dloges)
            w = j+1
            f = (ef-loges[j])/dloges
            a = trm[j]*(1-f) + trm[w]*f
        dt = sf.t[i]-sf.t[i-1]
        y = sf.y[r.ir1,i]
        w = where(y < 0)
        y[w] = 0.0
        s = s + y*dt*(-a[r.ir0,r.ir1])
    return s

def ssp_basis(z, k, emin, emax, n0, n1, kmax, neu, eus0, eus1, fsav,
              ddir='data', sdir='spec', r=None, es=[], trm=[], sw=0,
              tgt='H', ecx=0,
              rxf=1.0,
              lzd='/Users/yul20/src/Kronos_v3.1/CXDatabase/Projectile_Ions'):
    if r == None:
        r = rad_cas(z, k, emin, emax, pf='F10', ddir=ddir, sdir=sdir)
    ne = len(r.egy)
    ys = zeros((neu,n1+1,kmax+1,ne))
    nlev = len(r.trm[0])
    if len(es)==0 or len(trm)==0:
        es, trm = eb_trm(z, k, ddir=ddir)
    deu = (log(eus1)-log(eus0))/(neu-1)
    eus = exp(arange(log(eus0), log(eus1)+0.1*deu, deu))
    a = fac.ATOMICSYMBOL[z]
    lzf = '%s/%s/Charge/%2d/Targets/%s/%s%2d+%s_sec_faclz_nres.cs'%(lzd,a,z,tgt,a.lower(),z,tgt.lower())
    print(lzf)
    rxe,rxv = lzrx(lzf)
    rxi = interpolate.interp1d(log(rxe), rxv*rxf,
                               bounds_error=False, fill_value='extrapolate')
    reu = rxi(log(eus))
    for n in range(n0,n1+1):
        for km in range(min(n,kmax+1)):
            if ecx == 0:
                if sw == 0:
                    cs = cs_cx(z, k, nlev,
                               nk='n%02dk%02d'%(n,km), sdir=sdir)
                else:
                    cs = cs_cx(z, k, nlev,
                               nk='n%d%02dk%02d'%(sw,n,km), sdir=sdir)
            else:
                if sw == 0:
                    cs0,cs1,cs = eb_cx(z, k, 0, len(es),
                                       nk='n%02dk%02d'%(n,km), sdir=sdir)
                else:
                    cs0,cs1,cs = eb_cx(z, k, 0, len(es),
                                       nk='n%d%02dk%02d'%(sw,n,km), sdir=sdir)

            for ie in range(neu):
                print([n,km,ie,eus[ie],reu[ie]])
                sys.stdout.flush()
                sf = int_rate(es, trm, cs, ecx=ecx, rx=reu[ie], eu=eus[ie])
                ys[ie,n,km] = int_lines(r, sf, es, trm, rx=reu[ie], eu=eus[ie])
    if fsav != '':
        with open(fsav, 'wb') as f:
            pickle.dump((es,trm,eus,r,ys), f)

def load_basis(f):
    with open(f, 'rb') as fs:
        return pickle.load(fs)
    
def mavg_basis(f0, f1, nt, t0, t1):
    with open(f0,'rb') as f:
        es,trm,eus,r,ys = pickle.load(f)
    dt = (log(t1)-log(t0))/(nt-1)
    ts = exp(arange(log(t0), log(t1)+0.1*dt, dt))
    ss = ys.shape
    ys = ys.reshape((ss[0],ss[1]*ss[2]*ss[3]))
    n1 = 8
    eus1 = zeros(ss[0]+n1)
    ys1 = zeros((ss[0]+n1,len(ys[0])))
    ys1[:ss[0],:] = ys
    eus1[:ss[0]] = eus
    for i in range(ss[0],len(eus1)):
        eus1[i] = eus1[i-1]*(eus[1]/eus[0])
    for i in range(len(ys[0])):
        fi = interpolate.interp1d(log(eus),ys[:,i],
                                  fill_value='extrapolate', bounds_error=False)
        ys1[ss[0]:,i] = fi(log(eus1[ss[0]:]))
    wt = zeros((nt,ss[0]+n1))
    for i in range(nt):
        t = ts[i]
        x = eus1/t
        dx = log(x[1])-log(x[0])
        wt[i] = 1.12838*sqrt(x)*exp(-x)*x*dx
    yt = matmul(wt, ys1).reshape((nt,ss[1],ss[2],ss[3]))
    with open(f1, 'wb') as f:
        pickle.dump((es,trm,ts,r,yt), f)
    return ts, yt

def pick_lines(idx, ai, egy, de, eps):
    w = where(idx >= 0)
    ir0 = w[0]
    ir1 = w[1]
    nw = len(ir0)
    iz = repeat(False, nw)
    for i in range(0,nw):
        ix = idx[ir0[i],ir1[i]]
        e = egy[ix]
        k = where(abs(egy-e) <= de)
        k = k[0]
        if ((ai[:,ix,:] > eps*nanmax(ai[:,k,:],1)).any()):
            iz[ix] = True
    return (where(iz))[0]    
        
def ion_data(z, k, ns, ws0, emin, emax,
             ddir='data', sdir='spec', df='',
             kmin=0, kmax=-1, pwi=6.0, pth=0.0,
             lzd='/Users/yul20/src/Kronos_v3.1/CXDatabase/Projectile_Ions',
             tgt=''):
    """cascade data
    z: atomic number
    k: number of electrons
    ws0: 2J+1/2S+1 of j/s split
    emin: min line energy
    emax: max line energy
    ddir: FAC data directory
    sdir: crm data direcotry
    kmin: min l
    kmax: max l
    """
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
        if df == '':
            ir0 = int32(r[2])
            ir1 = int32(r[1])
            im = 1 + max(ir1)
            ttr = zeros((im,im))
            ttr[ir0,ir1] = r[3]
            ga = sum(ttr,0)*3.29e-16
            idx = zeros((im,im), dtype=int32)
            egy = rd.e[ir1] - rd.e[ir0]
            w = where(logical_and(egy > emin, egy < emax))
            ir0 = ir0[w]
            ir1 = ir1[w]
            egy = egy[w]
            idx[:,:] = -1
            idx[ir0,ir1] = arange(len(ir0), dtype=int32)
        else:
            with open(df, 'rb') as fs:
                es,trm,eus,rs,ys = pickle.load(fs)
                im = len(trm[0,0])
                ga = 3.29e-16*diag(rs.trm)
                egy = rs.egy
                w = where((egy>emin)&(egy<emax))
                w = w[0]
                egy = egy[w]
                ir0 = rs.ir0[w]
                ir1 = rs.ir1[w]
                ys = ys[:,:,:,w]
                idx = zeros((im,im), dtype=int32)
                idx[:,:] = -1
                idx[ir0,ir1] = arange(len(ir0), dtype=int32)
                ev = list(eus)
    else:
        ds0 = ''
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
        ga = zeros(im)
    nt = len(ir0)
    nn = len(ns)
    nm = int(mean(ns))
    if kmax < 0:
        kmax = nm-1
    nmax = max(ns)
    if kmax > nmax-1:
        kmax = nmax-1
    if nn == 1 and ns[0] == 0:
        kmin = 0
        kmax = 0
    nm = kmax-kmin+1
    xk = arange(nm)+kmin
    ai = zeros((nn, nt, nm))
    iv = 0
    ev = []
    av = []
    if (df != ''):
        ev = list(eus)
        iv = where(eus >= 4.5)
        iv = iv[0][0]
        for i in range(len(ev)):
            av.append(zeros((nn,nt,nm)))
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
                if n > 0 and kk >= n:
                    continue
                ki = kk-kmin
                if k > 0:
                    if n == 0:
                        ps = ps0
                    else:
                        ps = '%s%s%02dk%02d'%(ps0, nc, (ws*100+n), kk)
                    if df == '':
                        ofn = ps + 'a.ln'
                        d = rcs(ofn)
                        if len(d) == 0:
                            wzai[ki] = 1
                            continue
                        it0 = atleast_1d(int32(d[1]))
                        it1 = atleast_1d(int32(d[2]))
                        si = atleast_1d(d[6]).copy()
                    else:
                        it0 = ir0
                        it1 = ir1
                        si = atleast_1d(ys[iv,n,kk]).copy()
                    if n == 0:
                        ofn = ps + 'a.r3'
                    else:
                        ofn = ps + 'a.r7'
                    c = rcs(ofn)
                    ic0 = atleast_1d(int32(c[1]))
                    ic1 = atleast_1d(int32(c[2]))
                    w = where(ic0 == min(ic0))
                    w = w[0]
                    ad[i,ki,ic1[w]] += c[3][w]
                else:
                    it0 = ir0
                    it1 = ir1
                    si = atleast_1d(d[6]).copy()
                    si[:ki] = 0.0
                    si[ki+1:] = 0.0
                ix = idx[it0, it1]
                w = where(ix >= 0)
                ix = ix[w]
                ai[i,ix,ki] += si[w]
                for ii in range(len(ev)):
                    av[ii][i,ix,ki] += ys[ii,n,kk][w]
        zai = logical_and(zai, wzai)

    if pth <= 0:
        if k < 3:
            pth = 2e-2
        else:
            pth = 6e-2
    nl0 = len(egy)
    if pwi > 0:
        w = pick_lines(idx, ai, egy, pwi, pth)
        egy = egy[w]
        ir0 = ir0[w]
        ir1 = ir1[w]
        ai = ai[:,w,:]
        for ii in range(len(ev)):
            av[ii] = av[ii][:,w,:]
    
    print([df,k,len(egy),nl0,len(ev),iv,pwi,pth])
    idx[:,:] = -1
    idx[ir0,ir1] = arange(len(ir0), dtype=int32)
    an = zeros(nn+1)
    ae = zeros(nn+1)
    wk0 = fac.LandauZenerLD(z, nm, 5)
    wk0 = wk0/sum(wk0)
    w = where(zai > 0)
    w = w[0]
    wk0[w] = 0.0
    wk = zeros(nm)
    we = zeros(nm)
    swk = zeros(nm)
    swe = zeros(nm)
    ide = zeros(nt, dtype=int8)
    de = zeros(nt)
    if lzd != '' and tgt != '':
        z1 = z-k+1
        a1 = fac.ATOMICSYMBOL[z1]
        lzf = '%s/%s/Charge/%2d/Targets/%s/%s%2d+%s_sec_faclz_nres.cs'%(lzd,a1,z1,tgt,a1.lower(),z1,tgt.lower())
        rz = loadtxt(lzf, unpack=1)
        nr,nx = rz.shape
        lznd = zeros((nn+1,nx))
        lznd[0] = log10(rz[0])
        for ii in range(nn):
            j = ns[ii]+1
            lznd[ii+1] = rz[-j]
        wcs = 1/sum(lznd[1:,],0)
        lznd[1:,] = matmul(lznd[1:,], diag(wcs))
    else:
        lzf = ''
        lznd = None
    
    return IonData(ddir, sdir, ps0, ds0, rd, r, ir0, ir1, idx, egy, ai, ev, av, ad, zai, z, k, ns, nm, kmin, kmax, an, ae, an.copy(), wk0, wk, we, swk, swe, xk, ide, de, ws0, emin, emax, df, lzd, tgt, lzf, lznd, ga)

def escape_peak(x, beta, gamma):
    xx = (x+beta)/(1.41421356*gamma)
    yc = zeros(len(xx))
    w = where(xx >= 0)
    yc[w] = log(special.erfcx(xx[w])) - xx[w]*xx[w]
    w = where(xx < 0)
    yc[w] = log(special.erfc(xx[w]))
    y = yc + x*beta
    return y

def convolve(e, s, sig, emin=-1, emax=-1, nde=3):
    de = sig/nde
    if emin <= 0:
        emin = max(0, min(e)-5*sig)
    if emax <= 0:
        emax = max(e)+5*sig

    em = arange(emin, emax, de)
    elo = em-0.5*de
    ehi = em+0.5*de
    nt = len(e)
    nd = len(em)
    wf = 0.3989423*(ehi-elo)/sig    
    y = zeros(nd)
    for i in range(nt):
        x = (em-e[i])/sig
        ax = abs(x)
        w = where(x < 10.0)
        w = w[0]
        if len(w) > 0:
            y[w] += wf[w]*s[i]*exp(-0.5*x[w]*x[w])
    return SpecData('', -1, [emin,emax], elo, ehi, em, y, None, None, None)

def calib_c2e(x, es):
    nes = len(es)
    if nes < 3:
        return x
    
    xe = x-es[1]
    if nes >= 3:
        x = es[0] + xe/es[2]
        if nes > 3:
            for iter in range(5):
                xp = x-es[0]
                xep = xp.copy()
                xeq = xe.copy()
                for i in range(3,nes):
                    xep *= xp
                    xeq -= es[i]*xep
                x = es[0] + xeq/es[2]
    return x

def calib_e2c(egy, es):
    nes = len(es)
    if nes < 3:
        return egy
    xe = egy-es[0]
    egy = es[1] + es[2]*xe
    if nes > 3:
        xep = xe.copy()
        for i in range(3, nes):
            xep *= xe
            egy += es[i]*xep
    return egy

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

gfxy = []
vfxy = []
dga = 0.5/5000
aga = arange(0.0,0.5+0.1*dga,dga)
def prep_gprof():
    x = arange(-15.0,1e-5,0.02)
    y = exp(-0.5*x*x)
    y = cumsum(y)
    y = log(y*0.5/y[-1])
    fxy = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value = (y[0],y[-1]))
    gfxy.append(fxy)
    
def prep_vprof():
    for ig in range(len(aga)):
        x = arange(-15.0,1e-5,0.02)
        a = aga[ig]
        y = voigt(a, x)
        y = cumsum(y)
        y = log(y*0.5/y[-1])
        fxy = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value = (y[0],y[-1]))
        vfxy.append(fxy)
        
def response(d, s, sig, es=[], eip=[]):
    msp = int(s.stype/100)
    vmd = 0
    if len(eip) > 0 and eip[-1] < 0:
        sig0 = sig.copy()
        eip0 = eip.copy()
        vmd = int(-eip[-1]+0.1)
        eip = eip[:-1]
        if vmd < 30:
            sig = sig[:-(vmd%10)]
        else:
            sig = sig[:-1]
    else:
        sig0 = sig
        eip0 = eip
    elo = s.elo
    ehi = s.ehi
    nr = len(elo)
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
    nst = len(sig)
    nei = (len(eip)+1)
    ns = int(0.01+nst/nei)
    ga = d.ga[d.ir1]
    afxy = []
    afxy0 = []
    for ie in range(nei):
        isig = sig[ie*ns:ie*ns+ns]
        if (msp==2):
            gamma = isig[2]
            x = arange(-50.0/isig[1], 50.0*gamma, 0.025)
            y = escape_peak(x, isig[1], gamma)
            yi = zeros(len(y))
            w = where(y > -300)
            yi[w] = exp(y[w])
            yi = cumsum(yi)
            yi = log(yi/yi[-1])
            fxy = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0],y[-1]))
            x = arange(-15.0,1e-5,0.01)
            y = exp(-0.5*x*x)
            y = cumsum(y)
            y = log(y*0.5/y[-1])
            fxy0 = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value = (y[0],y[-1]))
            afxy.append(fxy)
            afxy0.append(fxy0)
        elif msp == 0:
            if len(vfxy) == 0:
                prep_vprof()
            afxy = vfxy
        elif msp == 1:
            if len(gfxy) == 0:
                prep_gprof()
            afxy = gfxy

    def cprof(xx, ie, a):
        if msp == 1:
            fxy = lambda x: exp(afxy[0](x))
        elif msp == 0:
            if a <= aga[0]:
                fxy = lambda x: exp(afxy[0](x))
            elif a >= aga[-1]:
                fxy = lambda x: exp(afxy[-1](x))
            else:
                ig0 = int((a-aga[0])/dga)
                ig1 = ig0+1
                fg = (a-aga[ig0])/dga
                fxy0 = afxy[ig0]
                fxy1 = afxy[ig1]
                fxy = lambda x: exp(fxy0(x))*(1-fg)+exp(fxy1(x))*fg
        else:
            fxy = lambda x: exp(afxy[ie](x))
            
        if msp == 2:
            fxy0 = lambda x: exp(afxy0[ie](x))
            yy1 = fxy(xx)
            w = xx <= 0
            w0 = where(w)[0]
            yy = zeros(len(xx))
            if len(w0) > 0:
                yy[w0] = fxy0(xx[w0])
                w1 = where(logical_not(w))[0]
                if len(w1) > 0:
                    yy[w1] = 1-fxy0(-xx[w1])
            yy = (yy + sig[nei+1]*yy1)/(1+sig[nei+1])
            return yy
        w = xx <= 0
        w0 = where(w)[0]
        yy = zeros(len(xx))
        if len(w0) > 0:
            yy[w0] = fxy(xx[w0])
        w1 = where(logical_not(w))[0]
        if len(w1) > 0:
            yy[w1] = 1-fxy(-xx[w1])
        return yy
    
    nes = len(es)
    if nes > 2:
        egy = calib_e2c(egy, es)
    for i in range(nt):        
        e = egy[i]
        if nei > 1:
            for ie in range(nei):
                if ie == nei-1:
                    break
                if eip[ie]>e:
                    break
            esig = sig[ie*ns]
            asig = sig[ie*ns+1]
            if ns > 2:
                ssig = sig[ie*ns+2]
                if ns > 3:
                    bsig = sig[ie*ns+3]
                    csig = sig[ie*ns+4]
        else:
            ie = 0
            esig = sig[0]
            asig = sig[1]
            if ns > 2:
                ssig = sig[2]
                if ns > 3:
                    bsig = sig[3]
                    csig = sig[4]
        if msp == 1:
            dhi = (ehi-e)/esig
            dlo = (elo-e)/esig
            w = where(logical_or(abs(dhi)<15,abs(dlo)<15))
            if ns == 1:
                ar[w,i] = cprof(dhi[w],ie,0.0)-cprof(dlo[w],ie,0.0)
            else:
                ar[w,i] = (1-ssig)*(cprof(dhi[w],ie,0.0)-cprof(dlo[w],ie,0.0))
                dhi = (ehi-e)/(esig*asig)
                dlo = (elo-e)/(esig*asig)
                w = where((abs(dhi)<15)&(abs(dlo)<15))
                ar[w,i] += ssig*(cprof(dhi[w],ie,0.0)-cprof(dlo[w],ie,0.0))
        elif msp == 0:
            a = (ga[i]+asig)/(1.41421*esig)
            if ns == 2:
                ar[:,i] = cprof((ehi-e)/esig,ie,a)-cprof((elo-e)/esig,ie,a)
            else:
                ar[:,i] = (1-bsig)*(cprof((ehi-e)/esig,ie,a)-cprof((elo-e)/esig,ie,a))
                a /= ssig
                esig *= ssig
                ar[:,i] += bsig*(cprof((ehi-e)/esig,ie,a)-cprof((elo-e)/esig,ie,a))
                xm = (0.5*(ehi+elo)-e)/esig
                xs = 1+csig*xm*exp(-0.001*xm*xm)
                wm = where(xs < 0.05)
                xs[wm] = 0.05
                ar[:,i] *= xs
                
    aj = zeros((nn,nr,nm))
    if (len(d.ev) > 0 and vmd > 0):
        evs = sig0[-(vmd%10):]
        if vmd < 20:
            ev = 10**evs[0]
            if ev <= d.ev[0]:
                d.ai[:,:,:] = d.av[0][:,:,:]
            elif ev >= d.ev[-1]:
                d.ai[:,:,:] = d.av[-1][:,:,:]
            else:
                ie0 = int((log(ev)-log(d.ev[0]))/(log(d.ev[1])-log(d.ev[0])))
                ie1 = ie0+1
                fe = (log(ev)-log(d.ev[ie0]))/(log(d.ev[ie1])-log(d.ev[ie0]))
                d.ai[:,:,:] = d.av[ie0]*(1-fe) + d.av[ie1]*fe
            if vmd > 10:
                ev0 = evs[0]
                ev1 = ev0
                if vmd > 11:
                    ev0 = evs[1]
                    ev1 = ev0
                    if vmd > 12:
                        ev1 = evs[2]
                if d.k == 1:
                    ev = ev0
                else:
                    ev = ev1
                if ev <= d.lznd[0][0]:
                    d.an[:-1] = d.lznd[1:,0]
                elif ev >= d.lznd[0][-1]:
                    d.an[:-1] = d.lznd[1:,-1]
                else:
                    ie0 = int((ev-d.lznd[0][0])/(d.lznd[0][1]-d.lznd[0][0]))
                    ie1 = ie0+1
                    fe = (ev-d.lznd[0][ie0])/(d.lznd[0][ie1]-d.lznd[0][ie0])
                    d.an[:-1] = d.lznd[1:,ie0]*(1-fe) + d.lznd[1:,ie1]*fe
                    d.an[:-1] /= sum(d.an[:-1])
        elif vmd > 20:
            e0 = 10**evs[0]
            a = 10**evs[1]
            xe = 10**arange(-2.0,0.0,0.01)
            ye = 1-xe**(1/a)
            ye = ye/ye[0]
            d.ai[:,:,:] = 0.0
            for ie in range(1,len(xe)):
                ev = e0*xe[ie]
                if ev <= d.ev[0]:
                    ai = d.av[0][:,:,:]
                elif ev >= d.ev[-1]:
                    ai = d.av[-1][:,:,:]
                else:
                    ie0 = int((log(ev)-log(d.ev[0]))/(log(d.ev[1])-log(d.ev[0])))
                    ie1 = ie0+1
                    fe = (log(ev)-log(d.ev[ie0]))/(log(d.ev[ie1])-log(d.ev[ie0]))
                    ai = d.av[ie0][:,:,:]*(1-fe) + d.av[ie1][:,:,:]*fe
                d.ai[:,:,:] += (ye[ie-1]-ye[ie])*ai
    for i in range(nn):
        aj[i] = matmul(ar, d.ai[i])
    return Response(sig0, es, eip0, bn, ar, aj)

def calc_spec(d, r, ar=None):
    if type(r) == int or type(r) == int64:
        if (r < 0):
            y = 0.0
            for i in range(len(d.ds)):
                yi,ri = calc_spec(d, i)
                y += yi
            return y
        y,ar = calc_spec(d.ds[r], d.rs[r], ar)
        y *= d.sp.eff
        if len(d.rs[r].eip) > 0 and d.rs[r].eip[-1] <= -30:
            xeff = -d.rs[r].eip[-1]
            w = where(d.sp.xm < xeff)
            y[w] *= d.sp.eff[w]**(d.rs[r].sig[-1])
        return (y,ar)
    n = len(d.ns)
    if ar == None:
        ar = 0.0
        for i in range(n):
            ar += d.anr[n]*d.anr[i]*r.aj[i]
    y = matmul(ar, d.wk)
    return (y,ar)

def scale_spec(xm, yd, ye):
    w = where(ye > 0)
    eff = yd[w]/(ye[w]*ye[w])
    xx = xm[w]
    i = where(ye[w][1:-1] < 7.0)
    eff[1+i[0]] = 0.0
    i = where(eff > 0)
    xx = xx[i]
    eff = eff[i]
    fi = interpolate.interp1d(xx, eff, bounds_error=False, fill_value=(eff[0],eff[-1]))
    eff = fi(xm)
    yc = yd*eff
    return (yc, eff)

def mcmc_cmp(z, m):
    np = len(z.mpa)
    ni = len(z.ds)
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
    es = z.rs[0].es.copy()
    eip = z.rs[0].eip.copy()
    nes = len(es)-1
    if nes < 2:
        nes = 0
    ia = z.ia
    iw = z.iw
    iid = z.iid
    if m1 <= 0:
        m1 = z.imp
    m0 = m1-m
    if m0 < 0.1*m1:
        m0 = int32(0.1*m1)
    r = z.ene[m0:m1]
    ra = mean(r)
    rd = std(r)
    wr = m0+arange(m, dtype=int32)
    wr0 = wr
    if eps > -1e30:
        wr = where(r > ra+eps*rd)
        wr = wr[0]+m0
    elif rmin > -1e30:
        wr = where(r > rmin)
        wr = wr[0]+m0
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
    z.sp.xm[:] = calib_c2e(z.sp.em, z.rs[0].es)
    if nes > 0:
        es[1:] = z.mpa[nsig:nsig+nes]
    for ii in range(ni):
        z.rs[ii] = response(z.ds[ii], z.sp, z.mpa[:nsig], es, eip)
        (y,r) = calc_spec(z.ds[ii], z.rs[ii])
        z.ym[:] += z.sp.eff*y        
    if len(z.ib) > 0:
        if z.bf != None:
            yb = z.bf(z.sp, z.mpa[z.ib])
        z.ym[:] += yb*z.sp.eff
    if len(z.rs[0].eip) > 0 and z.rs[0].eip[-1] <= -30:
        xeff = -z.rs[0].eip[-1]
        w = where(z.sp.xm < xeff)
        z.ym[w] *= z.sp.eff[w]**(z.rs[0].sig[-1])
        
def read_spec(df, stype0, er=[]):
    stype = stype0%100
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
        elif stype == 5:
            d = rcs(df[0])
            t = rcs(df[1])
            if (len(er) == 0):
                er = [6.5e3,9.4e3]
            w = logical_and(d[0] > er[0], d[0] < er[1])
            ww = where(w)[0]
            d[1][ww[0]]=0.0
            d[1][ww[-1]]=0.0
            for i in range(2,len(er),2):
                w1 = logical_and(d[0]>er[i],d[0]<er[i+1])
                ww1 = where(w1)[0]
                d[1][ww1[0]] = 0.0
                d[1][ww1[-1]] = 0.0
                w = logical_or(w, w1)
            w = where(w)                
            elo = d[0][w]-1.0
            ehi = d[0][w]+1.0
            yc = d[1][w]
            #w = where(t[0] < 3e3)
            #t[1][w] = 0.85
            fti = interpolate.interp1d(t[0],t[1],kind='linear',bounds_error=False, fill_value=(0.0,1.0))
            eff = fti(0.5*(elo+ehi))
            w = where(yc < 0)
            yc[w] = 0
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
                er = [1.9e3, 4e3]
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
        (yc,eff) = scale_spec(0.5*(elo+ehi), yd, ye)
    return SpecData(df, stype0, er, elo, ehi, 0.5*(elo+ehi), yc, yd, eff, ye, 0.5*(elo+ehi))

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

def ar_bkgd(s, p):
    if len(p) == 3:
        x0 = p[2]
    else:
        x0 = 3.1e3
    x = (s.em-x0)/p[1]
    y = zeros(len(s.em))
    de = s.ehi-s.elo
    w = where(x > 50)[0]
    if len(w) > 0:
        xw = exp(-x[w])
        y[w] = p[0]*de[w]*xw/(1+xw)
    w = where(x <= 50)[0]
    if len(w) > 0:
        y[w] = p[0]*de[w]/(1+exp(x[w]))
    return y

def mcmc_spec(ds, sp, sig, eth, imp, fixld=[], fixnd=[], racc=0.4,
              es=[], wes=[], wb=[],
              wsig=[], sav=[], mps=[], nburn=0, nopt=0, sopt=0, fopt='',
              yb=1e-1, ecf='', ierr=[], emd=0, wreg=10.0,
              sde0=100.0, sde1=0.25, sde2=1.0,
              bkgd=([],None), eip=[], fixwk=[]):
    t0 = time.time()
    yd = sp.yc + yb
    wyg = where(yd >= 10)[0]
    wys = where(yd < 10)[0]
    eff = sp.eff
    elo = sp.elo
    ehi = sp.ehi
    nd = len(yd)
    lnydi = special.gammaln(yd+1.0)
    lnysi = log(sqrt(2*pi*yd))
    lnydi -= lnysi
    slnydi = sum(lnydi)
    na = 0
    nw = 0
    nde = 0
    nei = len(eip)+1
    ia = [0]
    iw = [0]
    ide0 = []
    ide1 = []
    ti = 0.0
    ni = len(ds)
    d0 = ds[0]
    i0 = 0
    load_ecf(ds, ecf)
    k0 = 1000
    k1 = 0
    emin = calib_c2e(elo[0], es)
    emax = calib_c2e(ehi[-1], es)
    for ii in range(len(ds)):
        d = ds[ii]
        if d.k > 0:
            if d.k > k1:
                k1 = d.k
            if d.k < k0:
                k0 = d.k
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
        w = where(logical_and(d.egy > emin, d.egy < emax))
        msi = max(si[w])
        ti += sum(si[w])
        ww = where(logical_and(si[w] > msi*eth, d.ide[w] == 0))
        if (len(ww[0]) > 0):
            d.ide[w[0][ww]] = 1
        if d.k == 0:
            d.ide[:] = 1
    iea = zeros(ni)
    merr = 0.0
    if len(ierr) > 0:
        ie = atleast_1d(ierr)
        merr = max(ie)
        for i in range(ni):
            j = ds[i].k-k0
            if j >= 0:
                if j < len(ie):
                    iea[i] = ie[j]
                else:
                    iea[i] = ie[-1]
    if len(bkgd) == 3:
        merr = max([merr,bkgd[2]])
    w = where(d0.ide == 1)
    nde0 = nde
    nde = nde + len(w[0])
    for i1 in range(i0, len(ds)):
        ide0.append(nde0)
        ide1.append(nde)
        if i1 > i0:
            ds[i1].ide[:] = d0.ide
    nsig = len(sig)
    es = array(es)
    nes = len(es)-1
    if nes < 2:
        nes = 0
    nrp = nsig+nes
    bfun = bkgd[1]
    nbp = len(bkgd[0])
    nsb = nrp+nbp
    ibp = range(nrp,nsb)
    ide0 = [nsb+x for x in ide0]
    ide1 = [nsb+x for x in ide1]
    nde1 = nde + nsb
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
    yd1 = yd+1.0
    ye = sqrt(yd1)
    syd = sum(yd)
    yte = sqrt(syd)
    mp[:nsig] = sig
    if nes > 0:
        mp[nsig:nrp] = es[1:]
    mp[iid] = 1.0    
    for i in range(ni):
        mp[ia[i]:ia[i+1]] = 1.0/(ia[i+1]-ia[i])
        mp[iw[i]:iw[i+1]] = 1.0/(iw[i+1]-iw[i])
        wz = where(ds[i].zai > 0)
        mp[iw[i]+wz[0]] = 0.0
    for fwk in fixwk:
        i = fwk[0]
        j = fwk[1]
        a = fwk[2]
        b = sum(mp[iw[i]:iw[i+1]])
        mp[iw[i]:iw[i+1]] *= (b-a)/(b-mp[iw[i]+j])
        mp[iw[i]+j] = a
    mp0 = zeros(np)
    mp1 = zeros(np)
    smp = zeros(np)
    smp[:nsig] = 0.25*abs(mp[:nsig])
    if nes > 0:
        smp[nsig:nrp] = 0.002*mp[nsig:nrp]
        for i in range(3,nes+1):
            smp[nsig+i] = 0.05*mp[nsig+i]
    smp[nsb:nde1] = 0.25
    smp[iid] = 0.25
    smp[ia[0]:ia[-1]] = 0.1*mp[ia[0]:ia[-1]]
    smp[iw[0]:iw[-1]] = 0.1*mp[iw[0]:iw[-1]]
    mp0[:nsig] = sig*0.01
    mp1[:nsig] = sig*100.0
    if nes > 0:
        mp0[nsig:nrp] = mp[nsig:nrp]*0.996
        mp1[nsig:nrp] = mp[nsig:nrp]*1.004
        for i in range(3, nes+1):
            mp0[nsig+i] = 1e-16
            mp1[nsig+i] = mp[nsig+2]*1e-3
        for i in range(len(wes)):
            mp0[nsig+i] = wes[i][0]
            mp1[nsig+i] = wes[i][1]
            
    for i in range(nbp):
        bp = bkgd[0][i]
        mp[ibp[i]] = bp[0]
        mp0[ibp[i]] = bp[1]
        mp1[ibp[i]] = bp[2]
        smp[ibp[i]] = bp[3]
    w = where(smp <= 0)
    smp[w] = 1e-5
    for i in range(len(wsig)):
        mp0[i] = wsig[i][0]
        mp1[i] = wsig[i][1]
    for ii in range(ni):
        if ds[ii].k == 0:
            mp0[ide0[ii]:ide1[ii]] = -sde0
            mp1[ide0[ii]:ide1[ii]] = sde0
        elif ds[ii].k == 1:
            mp0[ide0[ii]:ide1[ii]] = -sde1
            mp1[ide0[ii]:ide1[ii]] = sde1
        else:
            mp0[ide0[ii]:ide1[ii]] = -sde2
            mp1[ide0[ii]:ide1[ii]] = sde2
            
    mp0[iid] = 0.0
    mp1[iid] = 1e31
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
    ap = zeros((ni,nd,nw))
    rs = [None for i in range(len(ds))]
    rs0 = [None for i in range(len(ds))]
    y = zeros(nd)
    iye = zeros(nd)
    iy = zeros((ni,nd))
    xin = arange(-30.0, 0.05, 0.05)
    xin[-1] = 0.0
    yin = integrate.cumtrapz(stats.norm().pdf(xin), xin, initial=0.0)
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
    
    def getym(ip):
        if (ip < nde1):
            for i in range(ni):
                if ip < nrp or (ip >= ide0[i] and ip < ide1[i]):
                    ww = wde[i]
                    if len(ww[0]) > 0:
                        de0[i][:] = ds[i].de[ww]
                        ds[i].de[ww] = mp[ide0[i]:ide1[i]]
                    rs0[i] = rs[i]
                    if nrp > nsig:
                        es[1:] = mp[nsig:nrp]
                    rs[i] = response(ds[i], sp, mp[0:nsig], es, eip)

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
                    #mp[ia[i]:ia[i+1]] = 0.0
                    mp[ia[i]:ia[i+1]] = wki
        ap[:,:,:] = 0.0
        for i in range(ni):
            if len(eip) > 0 and eip[-1] < -10 and eip[-1] > -20:
                mp[ia[i]:ia[i+1]] = ds[i].an[:-1]
            x = mp[ia[i]:ia[i+1]]
            for k in range(len(ds[i].ns)):
                ap[i,:,iw0[i]:iw0[i+1]] += rs[i].aj[k]*x[k]*mp[iid[i]]
            wz = where(ds[i].zai > 0)
            mp[iw[i]+wz[0]] = 0.0
        wk = mp[iw[0]:iw[-1]]
        iye[:] = 0.0
        for i in range(ni):
            iy[i,:] = matmul(ap[i], wk)*eff
            iye[:] += (iea[i]*iy[i])**2
        y[:] = sum(iy, axis=0)
        ybk = 0.0
        if nbp > 0:
            if bfun == None:
                ybk = mp[ibp[0]]*eff
            else:
                ybk = bfun(sp, mp[ibp])*eff
            y[:] += ybk
            if len(bkgd) == 3:
                iye[:] += (ybk*bkgd[2])**2
        iye[:] = sqrt(iye+(merr*yb)**2)
        if len(eip) > 0 and eip[-1] <= -30:
            xeff = -eip[-1]            
            w = where(sp.xm < xeff)
            y[w] *= eff[w]**(mp[nsig-1])

    def ilnlikely(ip):
        getym(ip)
        yt = y + yb
        if len(ierr) == 0:
            r = yd*log(yt)-yt-lnydi
        elif emd == 0:
            yt0 = yt - iye
            ytm = 1e-3*yt
            w = where(yt0 <= ytm)
            yt0[w] = ytm[w]
            yt1 = yt + iye
            w = where(logical_and(yd >= yt0, yd <= yt1))
            yt[w] = yd[w]
            w = where(yd > yt1)
            yt[w] = yt1[w]
            w = where(yd < yt0)
            yt[w] = yt0[w]
            r = yd*log(yt)-yt-lnydi
        else:
            r = zeros(nd)
            yt0 = yt - iye
            yt1 = yt + iye
            dyt = yt1-yt0
            if len(wys) > 0:
                r[wys] = special.gammainc(yd1[wys], yt1[wys]) - special.gammainc(yd1[wys], yt0[wys])
                i = where(r[wys] > 0)
                i = wys[i[0]]
                if len(i) > 0:
                    r[i] = log(r[i]/dyt[i]) + lnysi[i]
                i = where(r[wys]<=0)
                i = wys[i[0]]
                if len(i) > 0:
                    r[i] = yd[i]*log(yt[i])-yt[i]-lnydi[i]
            if len(wyg) > 0:
                dy1 = (yt1[wyg]-yd1[wyg])/ye[wyg]
                dy0 = (yt0[wyg]-yd1[wyg])/ye[wyg]
                q1 = logqx(dy1)
                q0 = logqx(dy0)
                w = where(logical_and(dy1 > 0, dy0 > 0))
                w = w[0]
                if len(w) > 0:
                    wi = wyg[w]
                    dq = q1[w]-q0[w]
                    qr = zeros(len(w))
                    i = where(dq > -300)
                    qr[i] = exp(dq[i])
                    r[wi] = q0[w] + log(1 - qr)
                w = where(logical_and(dy1 < 0, dy0 < 0))
                w = w[0]
                if len(w) > 0:
                    wi = wyg[w]
                    dq = q0[w]-q1[w]
                    qr = zeros(len(w))
                    i = where(dq > -300)
                    qr[i] = exp(dq[i])
                    r[wi] = q1[w] + log(1-qr)
                w = where(logical_and(dy1 > 0, dy0 < 0))
                w = w[0]
                if len(w) > 0:
                    wi = wyg[w]
                    i = where(q1[w] > -300)
                    i = i[0]                    
                    wii = wi[i]
                    r[wii] += exp(q1[w[i]])
                    i = where(q0[w] > -300)
                    i = i[0]
                    wii = wi[i]
                    r[wii] += exp(q0[w[i]])
                    r[wi] = log(1-r[wi])
                r[wyg] = r[wyg]-log(dyt[wyg]) + lnysi[wyg]
        return r
    
    def lnlikely(ip):
        r = ilnlikely(ip)
        r = sum(r)
        if wreg <= 0:
            return r
        
        for i in range(ni):
            if len(fixld) > 0:
                ii = fixld[i]
                if ii >= 0 and ii < ni:
                    continue            
            wk = mp[iw[i]:iw[i+1]]
            nw = len(wk)
            for j in range(1,nw):
                if wk[j] < wk[j-1]-0.01:
                    break
            for i in range(j+1,nw):
                if wk[i] > wk[i-1]+0.01:
                    dr = (wk[i]-wk[i-1])*wreg
                    r -= dr*dr
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
                if mode == 1:
                    if (ds[ii].zai[ip-idx[ii]] or ds[ii].zai[jp-idx[ii]]):
                        continue
                    skipij = 0
                    for fwk in fixwk:
                        ix = fwk[0]
                        jx = fwk[1]
                        if ix == ii and (jx+iw[ix] == ip or jx+iw[ix] == jp):
                            skipij = 1
                            break
                    if skipij:
                        continue
                imin = mp0[ip]
                imax = mp1[ip]
                mpi = mp[ip]
                jmin = mp0[jp]
                jmax = mp1[jp]
                mpj = mp[jp]
                sigma = sar[ip-idx[0],jp-idx[0]]
                #sigma = max(max(mpi,mpj),sigma)
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
                    #hmp[i1,ip] = mp[ip]
                    #hmp[i1,jp] = mp[jp]
                    har[i, ip-idx[0],jp-idx[0]] = dp
                trej += rp
                nrej += 1
                rar[i,ip-idx[0],jp-idx[0]] = rp
                
        return (r0,trej,nrej)
    
    r0 = lnlikely(0)
    dsc = syd/sum(y)
    mp[iid] *= dsc
    mp0[iid] = 0
    mp1[iid] = mp[iid]*1e10
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
            sys.stdout.flush()
        trej = 0.0
        nrej = 0
        for ip in range(nde2):
            if mp1[ip] <= mp0[ip]:
                continue
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
                            if ip < nrp or (ip >= ide0[ii] and ip < ide1[ii]):
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
            if len(eip) > 0 and eip[-1] < -10 and eip[-1] > -20:
                continue
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
                    ra = mean(rrej[im:i+1,ip])
                    fa = fmp[ip]*((1-ra)/racc)**2
                    fa = min(fa, 1e2)
                    fa = max(fa, 1e-2)
                    fa = 0.25*fmp[ip]+0.75*fa
                else:
                    ra = 0.0
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
        if nes == 0:
            pps = [mp[0], smp[0], fmp[0]]
        else:
            pps = [mp[0], mp[nsig], mp[nsig+1]]
        pp = [i, trej, ttr, rrm, arm, wrm, 
              r0, r0-ene[i1], pps[0], pps[1], pps[2],
              rrej[i,0], mean(rrej[max(0,i-25):i+1,0])]
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
        pp.append(time.time()-t0)
        print('imc: %6d %7.1E %7.1E %7.1E %7.1E %7.1E %12.5E %8.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %7.1E %10.4E'%tuple(pp))
        sys.stdout.flush()
        savenow = False
        if i == imp-1:
            savenow = True
        elif nsav > 0 and (i+1)%nsav == 0:
            savenow = True
        elif tsav != None and os.path.isfile(tsav):
            savenow = True
            os.system('rm '+tsav)            
        if savenow:
            print('pickling: %s %10.3E'%(fsav, time.time()-t0))
            sys.stdout.flush()
            if nsav > 0:
                fs = open(fsav, 'wb')
            for ii in range(ni):
                if (len(fixnd) > 0):
                    ij = fixnd[ii]
                    if ij >= 0 and ij < ni:
                        hmp[:,ia[ii]:ia[ii+1]] = hmp[:,ia[ij]:ia[ij+1]]
                if (len(fixld) > 0):
                    ij = fixld[ii]
                    if ij >= 0 and ij < ni:
                        hmp[:,iw[ii]:iw[ii+1]] = hmp[:,iw[ij]:iw[ij+1]]
            zi = FitMCMC(i+1, sp, y.copy(), mpa, mpe, r, ds, rs, ap, hmp, ene, nde, ide0, ide1, iid, ia, iw, ibp, bfun, frej, rrej, (iea,iye), fixnd, fixld)
            mcmc_avg(zi, int(i/2))
            if nsav > 0:
                pickle.dump(zi, fs)
                fs.close()

    print('done: %10.3E'%(time.time()-t0))
    return zi

def rebin_spec(s, nr):
    elo = s.elo
    ehi = s.ehi
    yd = s.yd
    yc = s.yc
    ye = s.ye
    eff = s.eff
    nd = len(elo)
    n1 = nd/nr
    elo1 = zeros(n1)
    ehi1 = zeros(n1)
    yc1 = zeros(n1)
    yd1 = zeros(n1)
    ye1 = zeros(n1)
    eff1 = zeros(n1)
    for i in range(n1):
        ir = i*nr
        elo1[i] = elo[ir]
        ehi1[i] = ehi[ir+nr-1]
        yd1[i] = sum(yd[ir:ir+nr])
        ye1[i] = sqrt(sum(ye[ir:ir+nr]**2))
        eff1[i] = mean(eff[ir:ir+nr])
        yc1[i] = sum(yc[ir:ir+nr])
    
    return SpecData(s.fns, s.stype, s.er, elo1, ehi1, 0.5*(elo1+ehi1), yc1, yd1,eff1, ye1)

def mod_spec(z, df, stype, er=[]):
    s = read_spec(df, stype, er)
    zm = FitMCMC(z.imp, s, s.yc.copy(), z.mpa, z.mpe, z.ra, z.ds.copy(), z.rs.copy(), z.ap.copy(), z.hmp, z.ene, z.nde, z.ide0, z.ide1, z.iid, z.ia, z.iw, z.ib, z.bf, z.frej, z.rrej, z.ierr, z.fixnd, z.fixld)
    emin = calib_c2e(s.elo[0], z.rs[0].es)
    emax = calib_c2e(s.ehi[-1], z.rs[0].es)
    for i in range(len(zm.ds)):
        ws = z.ds[i].ws
        if i < len(ws):
            iws = ws[i]
        else:
            iws = 0
        ddir = z.ds[i].ddir.split(',')
        if len(ddir) == 2:
            dfn = ddir[1]
        else:
            dfn = ''
        ddir = ddir[0]
        zm.ds[i] = ion_data(z.ds[i].z, z.ds[i].k, z.ds[i].ns, ws, emin, emax, ddir=ddir, sdir=z.ds[i].sdir, kmin=z.ds[i].kmin, kmax=z.ds[i].kmax, df=dfn, tgt=z.ds[i].tgt)
        wde = where(z.ds[i].ide)[0]
        for j in range(len(wde)):
            ir0 = z.ds[i].ir0[wde[j]]
            ir1 = z.ds[i].ir1[wde[j]]
            w = where(logical_and(zm.ds[i].ir0 == ir0, zm.ds[i].ir1 == ir1))[0]
            zm.ds[i].de[w] = z.ds[i].de[wde[j]]
    mcmc_avg(zm, int(zm.imp/2), zm.imp)
    return zm

def apply_es(s, es):
    s.elo[:] = calib_c2e(s.elo, es)
    s.ehi[:] = calib_c2e(s.ehi, es)
    s.em[:] = 0.5*(s.elo+s.ehi)
    s.xm[:] = s.em
    
def fit_spec(df, z, ks, ns, ws, sig, eth, stype, es=[], aes=0, wes=[],
             er=[], nmc=5000, fixld=[], fixnd=[],
             racc=0.35, kmin=0, kmax=-1,
             wb=[], wsig=[], sav=[], mps=[], nburn=0.25,
             nopt=0, sopt=0, fopt='', yb=1e-1, ecf='',
             ddir='data', sdir='spec', wreg=100.0,
             ierr=[], emd=0, sde0=100.0, sde1=0.25, sde2=1.0,
             bkgd=([],None), eip=[],
             fixwk=[], tgt=''):
    """driver for mcmc fit of cx spectra
    df: spectral data file
    z: atomic number
    ks: array of number of electrons
    ns: array of number of n
    ws: array of number of 2J+1/2S+1
    sig: parameters of response function
    eth: intensity threshold for liens need energy correction
    stype: spectral data type
    es: energy calibration scale, x=es[1]+es[2]*(e-es[0])+es[3]*(e-es[0])**2+...
    er: energy range of the fit
    nmc: number of mcmc iterations
    fixld: flags to fix l-dist
    fixnd: flags to fix n-dist
    racc: desired acceptance probability
    kmin: min l
    kmax: max l
    wb: min/max l-dist
    wsig: min/max of response params
    sav: (fn,niter) save the FitMCMC result in fn every niter iteration
    mps: continue the mcmc with pervious parameters
    nburn: burn-in iterations
    nopt,sopt,fopt: perform a non-linear-square fit periodically.
    yb: add a small background to the spectra just to avoid 0 count
    ecf: energy correction file
    ddir: FAC data direcotry
    sdir: crm data directory
    wreg: regularization parameter for high l capture
    ierr: theoretical model error
    sde: min/max energy corrections
    bkgd: include a background term in the fit model    
    """
    
    s = read_spec(df, stype, er)
    if aes > 0:
        apply_es(s, es)
        es = []
    sig = atleast_1d(sig)
    kmin = atleast_1d(kmin)
    kmax = atleast_1d(kmax)
    ds = []
    if len(es) > 2:
        es = array(es)
    emin = calib_c2e(s.elo[0], es)
    emax = calib_c2e(s.ehi[-1], es)
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
        df = ''
        ddir1 = ddir
        sdir1 = sdir
        
        if ki <= 2 and len(eip) > 0 and eip[-1] < 0 and eip[-1] > -30:
            df = '%s%02d%sKL%d.pkl'%(fac.ATOMICSYMBOL[z],ki,tgt,iws)
            #ddir1 = 'data1'
            #sdir1 = 'spec1'                
            
        di = ion_data(z, ki, nsi, iws, emin, emax, kmin=k0, kmax=k1, ddir=ddir1, sdir=sdir1, df=df, tgt=tgt)
        ds.append(di)
    z = mcmc_spec(ds, s, sig, eth, nmc, fixld, fixnd, racc, es, wes, wb, wsig, sav, mps, nburn, nopt, sopt, fopt, yb, ecf, ierr, emd, wreg, sde0, sde1, sde2, bkgd, eip, fixwk)
    return z

def plot_ldist(z, op=0, sav='', k=0, ap=plt, marker='o', mkf='none', lst='-', dx=0.0,color='k'):
    ds = z.ds
    if not op:
        clf()
    ymin = 1e31
    ymax = -1e31
    for i in range(len(ds)):
        if ds[i].k == 0:
            continue
        if k > 0 and ds[i].k != k:
            continue
        ymin0 = min(ds[i].wk-ds[i].we)
        ymax0 = max(ds[i].wk+ds[i].we)
        ymin = min(ymin0, ymin)
        ymax = max(ymax0, ymax)
        if k > 0 and ds[i].k == k:
            break

    dy = 0.025*(ymax-ymin)
    ymin -= dy
    ymax += dy
    ap.set_ylim(ymin, ymax)
    ni = len(ds)
    for i in range(ni):
        d = ds[i]
        if d.k == 0:
            continue
        if k > 0 and ds[i].k != k:
            continue
        ap.errorbar(d.xk-(0.2/ni)*(i-ni/2)+dx, d.wk,
                    yerr=d.we*1.5, marker=marker, capsize=1.5,
                    linestyle='-', fillstyle=mkf,
                    color=color, markersize=6.5, linewidth=0.5)
        if k > 0 and ds[i].k == k:
            break
    if ap == plt:
        xlabel('L')
        ylabel('Fraction')
    if sav != '':
        savefig(sav)
    
def plot_idist(z, op=0, sav=''):
    if not op:
        clf()
    ni = len(z.ds)
    x0 = arange(ni)
    y0 = zeros(ni)
    e0 = zeros(ni)
    for k in range(ni):
        d = z.ds[k]
        nn = len(d.ns)
        for i in range(nn):
            ta = sum(sum(d.ad[i], axis=1)*d.wk)*d.anr[-1]
            y0[k] += d.anr[i]*ta
            e0[k] += (d.ae[i]*ta)**2
    e0 = sqrt(e0)
    ys = sum(y0)
    y0 /= ys
    e0 /= ys
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
    
def plot_ndist(z, op=0, sav='', k=0):
    ds = z.ds
    if not op:
        clf()
    ymin = 1e31
    ymax = -1e31
    ni = len(ds)
    for i in range(len(ds)):
        d = ds[i]
        if k > 0 and ds[i].k != k:
            continue
        y0 = d.an[:-1]
        e0 = d.ae[:-1]
        ymin0 = min(y0-e0)
        ymax0 = max(y0+e0)
        ymin = min(ymin0, ymin)
        ymax = max(ymax0, ymax)
        if k > 0 and ds[i].k == k:
            break
    dy = 0.025*(ymax-ymin)
    ymin -= dy
    ymax += dy
    ylim(ymin, ymax)
    for i in range(len(ds)):
        if k > 0 and ds[i].k != k:
            continue
        d = ds[i]
        x0 = array(d.ns)-(0.2/ni)*(i-ni/2)
        y0 = d.an[:-1]
        e0 = d.ae[:-1]
        errorbar(x0, y0, yerr=e0, marker='o', capsize=3)
        if k > 0 and ds[i].k == k:
            break
    xlabel('N')
    ylabel('Fraction')
    if sav != '':
        savefig(sav)

def sum_wnk(z, k=0, ws=0):
    if k == 0:
        k = z.ds[0].k
    for d in z.ds:
        if d.k == k:
            break
    if k < 0:
        d = z.ds[-1-k]
    nn = len(d.ns)
    nk = len(d.xk)
    wn = zeros(nn)
    wk = zeros(nk)
    en = zeros(nn)
    ek = zeros(nk)
    for id in range(len(z.ds)):
        d = z.ds[id]
        if k >= 0:
            if d.k != k:
                continue
            if len(d.ns) != nn:
                continue
            if len(d.xk) != nk:
                continue
        else:
            if -1-k != id:
                continue
        if ws >= 100:
            if ws%100 > 0:
                if d.ws[0] != ws:
                    continue
            else:
                if d.ws[0]/100 != ws/100:
                    continue
        elif ws > 0:
            if d.ws[0]%100 != ws:
                continue
        for i in range(nn):
            ta = sum(sum(d.ad[i],axis=1)*d.wk)
            ra = ta*d.anr[-1]
            wn[i] += d.anr[i]*ra
            en[i] += (d.ae[i]*ra)**2 + (d.anr[i]*ta*d.ae[-1])**2
        for i in range(nk):
            ra = sum(d.ad[:,i,:],axis=1)
            sa = sum(ra*d.anr[:-1])
            ta = sa*d.anr[-1]
            wk[i] += d.wk[i]*ta
            ek[i] += (d.we[i]*ta)**2 + (d.wk[i]*sa*d.ae[-1])**2+ sum(((d.wk[i]*d.anr[-1]*ra)**2)*d.ae[:-1]**2)
    swn = sum(wn)
    swk = sum(wk)
    wn /= swn
    wk /= swk
    en = sqrt(en)/swn
    ek = sqrt(ek)/swk
    return (wn, en, wk, ek)
    
def avg_wnk(z, k=0, ws=0, m=0.25):
    if m < 1:
        m0 = int32(z.imp*m)
    else:
        m0 = int32(m)
    if k == 0:
        k = z.ds[0].k
    for d in z.ds:
        if d.k == k:
            break
    if k < 0:
        d = z.ds[-1-k]
    nn = len(d.ns)
    nk = len(d.xk)
    wn = zeros(nn)
    wk = zeros(nk)
    en = zeros(nn)
    ek = zeros(nk)
    for j in range(z.imp-m0, z.imp):
        mcmc_cmp(z, j)
        (iwn, ien, iwk, iek) = sum_wnk(z, k, ws)
        wn += iwn
        en += iwn*iwn
        wk += iwk
        ek += iwk*iwk
    wn /= m0
    wk /= m0
    en /= m0
    ek /= m0
    en = sqrt(en - wn*wn)
    ek = sqrt(ek - wk*wk)
    return (wn, en, wk, ek)
    
def plot_ink(z, k=0, s=0, op=0, col=0, xoffset=0, pn=1, sav='', ws=0,
             m=0, tsav='', ff=0):
    cols = ['k', 'b','r','g','c','m','y']
    if not op:
        clf()
    ii = 0
    if k == 0:
        k = z.ds[0].k
        ii = 0
    for d in z.ds:
        if d.k == k:
            break
        ii += 1
    if k < 0:
        d = z.ds[-1-k]
        ii = -1-k
    if m <= 0:
        (wn, en, wk, ek) = sum_wnk(z, k, ws)
    else:
        (wn, en, wk, ek) = avg_wnk(z, k, ws, m)
    xn = array(d.ns)+0.1+xoffset
    xk = d.xk-0.1+xoffset
    nn = len(d.ns)
    nk = len(d.xk)
    if nn > 1 and pn > 0:
        errorbar(xn, wn, yerr=en, capsize=3, marker='s', fillstyle='none', color=cols[col%len(cols)])
    if pn != 2:
        errorbar(xk, wk, yerr=ek, capsize=3, marker='o', color=cols[col%len(cols)])
    if ff > 0:
        p = z.hmp[int(z.imp/2):z.imp,z.iw[ii]:z.iw[ii]+nk]
        ap = sum(p[:,ff:],1)
        aa = mean(ap)
        da = std(ap)
        errorbar([ff+ii*0.15], [aa], yerr=da, capsize=3, marker='^', color=cols[col%len(cols)])
    xlabel('N/L')
    ylabel('Fraction')
    if sav != '':
        savefig(sav)
    if tsav != '':
        tf='%s%02d'%(tsav,k)
        if ws > 0:
            tf = '%ss%d'%(tf,ws)
        savetxt(tf+'.nd', transpose((array(d.ns)+1e-3, wn, en)), fmt='%2d %11.4E %11.4E')
        savetxt(tf+'.kd', transpose((array(d.xk)+1e-3, wk, ek)), fmt='%2d %11.4E %11.4E')

def plot_snk(z, sav='', col=0, op=0, xoffset=0, m=0, ws=0, tsav='', ff=0):
    cols = ['k', 'b','r','g','c','m','y']
    if not op:
        clf()
    plot_ink(z, k=1, col=col, pn=1, op=op, m=m, tsav=tsav, ff=ff)
    if ws == 2:
        plot_ink(z, k=2, ws=1, col=col+1, pn=1, op=1, m=m, tsav=tsav, ff=ff)
        plot_ink(z, k=2, ws=3, col=col+2, pn=1, op=1, m=m, tsav=tsav, ff=ff)
        labs = ['N-dist H-like', 'L-dist H-like',
                'N-dist He-Like Singlet', 'L-dist He-like Singlet',
                'N-dist He-Like Triplet', 'L-dist He-like Triplet']            
    else:
        plot_ink(z, k=2, ws=ws, col=col+1, pn=1, op=1, m=m, tsav=tsav, ff=ff)
        labs = ['N-dist H-like', 'L-dist H-like']
        if ff > 0:
            labs.append(r'H-like $f_{L>%d}$'%(ff-1))
        labs.append('N-dist He-like')
        labs.append('L-dist He-like')
        if ff > 0:
            labs.append(r'He-like $f_{L>%d}$'%(ff-1))
        
    legend(labs)
    #plot_ink(z, k=1, col=col, pn=2, op=1, m=m)    
    #plot_ink(z, k=2, ws=1, col=col+1, pn=2, op=1, m=m)    
    #plot_ink(z, k=2, ws=3, col=col+2, pn=2, op=1, m=m)
    if sav != '':
        savefig(sav)
    
def plot_spec(z, res=0, op=0, ylog=0, sav='', ymax=0, effc=0, es=1, ic=[], xr=[], yr=[], err=0, tsav=''):
    fm = z.sp
    if es > 0:
        xe = z.sp.xm
    else:
        xe = z.sp.em
    if not op:
        clf()
    ic = atleast_1d(array(ic))
    nc = len(ic)
    if nc > 0:
        yx = zeros((nc,len(xe)))
    if effc == 0:
        yd = fm.yc
        ym = z.ym
        ye = sqrt(yd)
        if z.bf != None:
            yb = z.bf(fm, z.mpa[z.ib])*fm.eff
        for i in range(nc):
            (y,r) = calc_spec(z, ic[i])
            yx[i] = y
    else:
        yd = fm.yd
        ym = z.ym/fm.eff
        ye = fm.ye.copy()
        if z.bf != None:
            yb = z.bf(fm, z.mpa[z.ib])
        for i in range(nc):
            (y,r) = calc_spec(z.ds[i], z.rs[i])
            yx[i] = y
    ye1 = zeros(len(yd))
    if type(z.ierr) == type(0.0):
        ye1 = z.ierr*ym    
    elif len(z.ierr[1]) == len(yd):
        ye1 = z.ierr[1]
    w = where(logical_and(ye < ye1, ye1 > 1))
    ye[w] = ye1[w]
    if res == 0:
        if ymax > 0:
            ylim(-ymax*0.01, ymax)
        if ylog > 0:
            ymin = max(yd)*ylog
            semilogy(xe, ymin+yd)
            semilogy(xe, ymin+ym)
            if z.bf != None:
                semilogy(xe, ymin+yb)
            for i in range(nc):
                semilogy(xe, ymin+yx[i])
        else:
            if err:
                plot(xe,yd)
                errorbar(xe, yd, yerr=sqrt(yd), marker='.', capsize=3,fmt=' ')
            else:
                plot(xe, yd)
            plot(xe, ym)
            if z.bf != None:
                plot(xe, yb)
            for i in range(nc):
                plot(xe, yx[i])
        labs = ['data','model']
        for i in ic:
            labs.append('ion %d'%i)
        legend(labs)
        ylabel('Intensity')
    else:
        r = yd-ym
        w = where(ye > 0)
        r[w] /= ye[w]
        w = where(ye <= 0)
        r[w] = 0
        plot(xe, r)
        ylabel('Residual')
    xlabel('Energy (eV)')
    if len(xr) == 2:
        xlim(xr[0], xr[1])
        if res == 0:
            w = where(logical_and(xe>=xr[0],xe<=xr[1]))
            ymax=max(yd[w].max(),ym[w].max())
            ylim(-0.05*ymax, 1.1*ymax)
    if len(yr) == 2:
        ylim(yr[0], yr[1])
    if sav != '':
        savefig(sav)
    if tsav != '':
        savetxt(tsav, transpose((xe, yd, ym)), fmt='%7.2f %11.4E %11.4E')
        
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

def sum_tcx(z, k=0, ws=0):
    c = 0.0
    e = 0.0
    if k == 0:
        k = z.ds[0].k
    for d in z.ds:
        if d.k != k:
            continue
        if ws >= 100:
            if ws%100 > 0:
                if d.ws[0] != ws:
                    continue
            else:
                if d.ws[0]/100 != ws/100:
                    continue
        elif ws > 0:
            if d.ws[0]%100 != ws:
                continue
        ci, ei = sum_cx(d)
        c += ci
        e += ei
    return (c,e)

def sel_icx(z, ic, k = 0, ws = 0):
    if k == 0:
        k = z.ds[0].k
    c,e = sum_tcx(z, k, ws)
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
    
def sum_icx(z, k = 0, ws = 0):
    if k == 0:
        k = z.ds[0].k
    c,e = sum_tcx(z, k, ws)
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

def sum_tnk(z, k=0, ws = 0):
    if k == 0:
        k = z.ds[0].k
    c,e = sum_tcx(z, k, ws)
    for d in z.ds:
        if d.k == k:
            break
    v = d.rd.v[:len(c)]
    vn = int32(v/100)
    vk = int32(v%100)
    nmax = nanmax(vn)
    r = zeros((nmax,nmax))
    s = zeros((nmax,nmax))
    for n in range(nmax):
        for j in range(nmax):
            w = where(logical_and(vn==n+1, vk==j))
            r[n,j] += sum(c[w])
            s[n,j] += sum(e[w])
    return (r,s)

def avg_tnk(z, k=0, ws=0, m=0.25):
    if k == 0:
        k = z.ds[0].k
    for d in z.ds:
        if d.k == k:
            break
    im = len(d.ad[0,0])
    v = d.rd.v[:im]
    vn = int32(v/100)
    vk = int32(v%100)
    nmax = nanmax(vn)
    r = zeros((nmax,nmax))
    s = zeros((nmax,nmax))
    if m < 1:
        m0 = int32(z.imp*m)
    else:
        m0 = int32(m)    
    for i in range(z.imp-m0, z.imp):
        mcmc_cmp(z, i)
        c,e = sum_tnk(z, k, ws)
        r += c
        s += c*c
    r /= m0
    s /= m0
    s = s - r*r
    mcmc_avg(z, z.imp/2)
    return (r,s)

def plot_tnk(z, op=0, nmin=0, nmax=-1, kmax=6, pn0=4, pn1=12,
             k=0, md=0, sav='', ws=0, col=0, lab=1, tsav=''):
    if not op:
        clf()
    if k == 0:
        k = z.ds[0].k
    cols = ['k', 'b','r','g','c','m','y']
    if md <= 0:
        r,s = sum_tnk(z, k, ws)
    else:
        r,s = avg_tnk(z, k, ws, md)
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
    if pn1 > pn0:
        errorbar(xn[idx], (tr/trs)[idx], yerr=(ts/trs)[idx], capsize=3, marker='s', fillstyle='none', color=cols[col])
    labs = ['N-dist']
    if (nmax >= nmin):
        for i in range(nmin, nmax+1):
            km = min(kmax+1, i)
            errorbar(xk[:km], y[i-1][:km], yerr=ye[i-1][:km], capsize=3, marker='o', color=cols[col+(i-nmin)%(len(cols)-1)])
            labs.append('L-dist N=%d'%i)
    else:
        yt = zeros(kmax+1)
        yte = zeros(kmax+1)
        for i in range(n):
            km = min(kmax+1,i)
            yt[:km] += y[i][:km]*tr[i]
            yte[:km] += (ye[i][:km]*tr[i])**2
        yt /= sum(yt)
        yte = sqrt(yte)/sum(tr)
        errorbar(xk[:km], yt, yerr=yte, capsize=3, marker='o', color=cols[col])
        labs.append('L-dist')        
        
    if lab > 0:
        legend(labs)
    xlabel('N/L')
    ylabel('Fraction')
        
    if sav != '':
        savefig(sav)
    if tsav != '':
        tf = '%s%02d'%(tsav,k)
        if ws > 0:
            tf = '%ss%d'%(tf,ws)
        savetxt(tf+'.nd', transpose((xn[idx], (tr/trs)[idx], (ts/trs)[idx])), fmt='%2d %11.4E %11.4E')
        savetxt(tf+'.kd', transpose((xk[:km], yt, yte)), fmt='%2d %11.4E %11.4E')
    
def plot_rnk(z, sav='', op=0, ws=0):
    i = argmax(z.ds[0].an[:-1])
    nm = z.ds[0].ns[i]
    n0 = z.ds[0].ns[0]
    n1 = z.ds[0].ns[-1]
    if n1 > n0:
        if ws == 2:
            lab=['N-dist H-like','L-dist H-like', 'N-dist He-like',
                 'L-dist He-like Singlet','L-dist He-like Triplet']
        elif ws == 0:
            lab = ['N-dist H-like', 'L-dist H-like', 'N-dist He-like', 'L-dist He-like']
        elif ws == 1:
            lab = ['N-dist H-like', 'L-dist H-like', 'N-dist He-like Singlet', 'L-dist He-like Singlet']
        elif ws == 3:
            lab = ['N-dist H-like', 'L-dist H-like', 'N-dist He-like Triplet', 'L-dist He-like Triplet']
        i0 = 0
    else:
        lab = ['H-like', 'He-like Singlet', 'He-like Triplet']
        i0 = -1
    plot_tnk(z, k=1, nmin=nm, nmax=nm, kmax=z.ds[0].kmax,
             pn0=n0, pn1=n1, lab=0, col=i0, op=op)
    n0 = z.ds[1].ns[0]
    n1 = z.ds[1].ns[-1]
    if ws == 2:
        plot_tnk(z, k=2, nmin=nm, nmax=nm, kmax=z.ds[1].kmax,
                 pn0=n0, pn1=n1, lab=0, col=i0+1, op=1, ws=1)    
        plot_tnk(z, k=2, nmin=nm, nmax=nm, kmax=z.ds[1].kmax,
                 pn0=n0, pn1=0, lab=0, col=i0+2, op=1, ws=3)
    else:
        plot_tnk(z, k=2, nmin=nm, nmax=nm, kmax=z.ds[1].kmax,
                 pn0=n0, pn1=n1, lab=0, col=i0+1, op=1, ws=0)
    legend(lab)
    if sav != '':
        savefig(sav)
        
def plot_icx(z, op=0, sav='', k=0, ws=0, xr=[100, 700], yr=[1e-3, 0.1], col=0, ic0=0):
    if not op:
        clf()
    if k == 0:
        k = z.ds[0].k
    for d in z.ds:
        if d.k == k:
            break
    nmin = min(d.ns)
    nmax = max(d.ns)
    for d1 in z.ds:
        if d1.k == k:
            nmin = min(min(d1.ns),nmin)
            nmax = max(max(d1.ns),nmax)
    cols = ['k', 'b','r','g','c','m','y']
    if col > 0:
        cols = cols[col:]+cols[:col]
    c0,e0,tc = sel_icx(z, 0, k, ws)
    ix = ic0+array(range(len(c0)))
    if d.k == 10:
        c1,e1,tc = sel_icx(z, 1, k, ws)
        c2,e2,tc = sel_icx(z, 2, k, ws)
    xlim(xr[0], xr[1])
    ylim(yr[0], yr[1])
    semilogy(ix, c0+1e-8, color=cols[0])
    v = int32(d.rd.v/100)
    if d.k == 10:
        semilogy(c1+1e-8, marker='o', color=cols[1])
        semilogy(c2+1e-8, marker='o', color=cols[2])
        labs = ['2p+', '2p-', '2s+']
        legend(labs)
        w1 = argmax(c1)
        w2 = argmax(c2)
        if v[w1] >= 10:
            n1 = d.rd.n[w1][-9:-5]
        else:
            n1 = d.rd.n[w1][-8:-5]
        if v[w2] >= 10:
            n2 = d.rd.n[w2][-9:-5]
        else:
            n2 = d.rd.n[w2][-8:-5]
        n1 = '%s J=%d'%(n1.decode(), d.rd.j[w1]/2)
        n2 = '%s J=%d'%(n2.decode(), d.rd.j[w2]/2)
        text(w1, c1[w1]*1.15, n1, color=cols[1])
        text(w2, c2[w2]*1.15, n2, color=cols[2])
    for n in range(nmin, nmax+1):
        w = where(v == n)
        w = w[0]
        i = argmax(c0[w])
        if c0[w[i]] < yr[0]*1.1:
            continue
        if n >= 10:
            ip = -9
        else:
            ip = -8
        if d.k == 10:
            if d.rd.j[w[i]]%2 == 0:
                ns = '%s J=%d'%(d.rd.n[w[i]][ip:-5].decode(), d.rd.j[w[i]]/2)
            else:
                ns = '%s J=%d/2'%(d.rd.n[w[i]][ip:-5].decode(), d.rd.j[w[i]])
        else:
            ns = d.rd.n[w[i]][ip:-5].decode()
        text(ic0+w[i], 1.15*c0[w[i]], ns, horizontalalignment='center', color=cols[0], size=7.5)
    xlabel('Level Index')
    if ws == 1:
        ylabel('Singlets Relative Cross Section')
    elif ws == 3:
        ylabel('Triplets Relative Cross Section')
    else:
        ylabel('Relative Cross Section')
    if sav != '':
        savefig(sav)

def plot_kcx(z, sav='', xr=[0, 250]):
    plot_icx(z, k=1, xr=xr, yr=[1e-4, 2])
    plot_icx(z, k=2, xr=xr, yr=[1e-4, 2], op=1, col=1, ic0=100)
    legend(['H-like', 'He-like'])
    if sav != '':
        savefig(sav)

def load_pkl(fn):
    f = open(fn,'rb')
    z = pickle.load(f)
    mcmc_avg(z, min(1000,int(z.imp/2)))
    f.close()
    return z

def step_sig(z, s, ds=0.5):
    x = arange(-ds,ds,ds*0.1)
    y = zeros(len(x))
    hs = z.hmp[:z.imp,s].copy()
    yd = z.sp.yc+0.01
    for i in range(len(x)):
        z.hmp[:z.imp,s] = hs + x[i]
        mcmc_avg(z, int(z.imp/2))
        yt = z.ym+0.01
        y[i] = sum(yd*log(yt)-yt)
        print([z.rs[0].sig[s],y[i]])

    return x,y

def plot_ksp(z):
    plot_spec(z, xr=[6.5e3,9.4e3], sav='fe+h2_spec_n1.pdf')
    plot_spec(z, xr=[2.025e3, 2.325e3], err=1, sav='fe+h2_spec_n2.pdf')
    plot_spec(z, xr=[6.61e3,6.72e3], err=1, sav='fe+h2_spec_hea.pdf')
    plot_spec(z, xr=[6.93e3,6.99e3], err=1, sav='fe+h2_spec_lya.pdf')
    plot_spec(z, xr=[7.8e3, 9.3e3], err=1, sav='fe+h2_spec_hn.pdf')
    plot_snk(z, sav='fe+h2_snk.pdf')
    
def plot_kbs(z, i, sav='', xr=[[2,2.35],[6.5,9.6]], lw=1):    
    x = z.sp.xm
    if len(xr) == 0:
        xr = [[x[0],x[-1]]]    
    d = z.ds[i]
    r = z.rs[i]
    k = argmax(d.an[:-1])
    nn = d.ns[k]
    ks = ['s', 'p', 'd', 'f', 'g', 'h', 'i']
    y = r.aj[k]
    if i == 1 and len(z.rs) > 2:
        y = y + z.rs[2].aj[k]
    y = y/sum(y)
    y *= sum(z.sp.yc)
    if i == 1:
        y *= 2.5
    dy = max(z.sp.yc)*0.075
    eff = z.sp.eff.copy()
    labs = []
    if len(r.eip) > 0:
        xe = r.eip[-1]
        if xe <= -30:
            w = where(x < -xe)
            eff[w] *= eff[w]**(r.sig[-1])
    for j in range(len(d.wk)):
        y[:,j] *= eff
    nr = len(xr)

    x = x/1e3
    proj = fac.ATOMICSYMBOL[z.ds[0].z]
    tgt = z.ds[0].tgt
    f,a = plt.subplots(1,nr)
    f.set_size_inches(8,6)
    f.subplots_adjust(wspace=0)
    for ir in range(nr):
        if i == 0:
            dx = 0.02*(xr[ir][1]-xr[ir][0])
        else:
            dx = 0.04*(xr[ir][1]-xr[ir][0])
        for j in range(len(d.wk)):
            if ir == 0:
                a[ir].plot(x, y[:,j]+j*dy, linewidth=lw)
            else:
                a[ir].plot((x+dx*j), y[:,j]+j*dy, linewidth=lw)
            labs.append(r'$%d%s\times%4.2f$'%(nn,ks[j], d.wk[j]))
            
        ys = (1+len(d.wk))*dy
        yc = z.sp.yc
        a[ir].plot(x, ys+yc, color='k', linewidth=lw)
        labs.append('total exp')
        if i == 0:
            y0,r0 = calc_spec(z, 0)
            a[ir].plot(x, ys+y0, color='r', linewidth=lw)
            if ir == 1:
                a[ir].text((xr[ir][0]+0.3*(xr[ir][1]-xr[ir][0])),
                         max(ys+yc)*0.9, proj+'+'+tgt+'  H-Like')
        else:
            y1,r1 = calc_spec(z, 1)
            if i == 1 and len(z.rs) > 2:
                y2,r2 = calc_spec(z, 2)
                y1 += y2                
            a[ir].plot(x, ys+y1, color='r', linewidth=lw)
            if ir == 1:
                a[ir].text((xr[ir][0]+0.3*(xr[ir][1]-xr[ir][0])),
                         max(ys+yc)*0.9, proj+'+'+tgt+'  He-Like') 
        labs.append('total mod')
        a[ir].set_xlim(xr[ir][0], xr[ir][1])
        a[ir].set_xlabel('Energy (keV)')
        if ir == 0:
            a[ir].set_ylabel('Counts')
            a[ir].legend(labs)         
        if ir == 1:
            a[ir].set_yticklabels([])   
    if sav != '':
        savefig(sav)

def plot_hp(z, i, bins=25, xsc=0, xlab='', op=0):
    if op == 0:
        clf()
    if (i >= 0):
        y = z.hmp[int(z.imp/2):z.imp,i]
    else:
        y = z.ene[int(z.imp/2):z.imp]
    h = histogram(y, bins=bins)
    x = h[1][:-1]
    if xsc == 0:
        plot(x, h[0]/max(h[0]), drawstyle='steps')
    elif xsc == 1:
        semilogx(10**x, h[0]/max(h[0]), drawstyle='steps')
    xlabel(xlab)

def load_fecx(md):
    ts = ['h2', 'h', 'n2', 'he']
    zs = []
    for t in ts:
        if md == 0:
            zs.append(load_pkl('kfe_%s.pkl'%t))
        elif md == 1:
            zs.append(load_pkl('nfe_%s.pkl'%t))
        else:
            zs.append(load_pkl('sfe_%s.pkl'%t))
    return zs

def plot_r01ce(zs, sav=''):
    clf()
    n = len(zs)
    tb = zeros(n)
    t0 = zeros(n)
    t1 = zeros(n)
    r = zeros(n)
    re = zeros(n)
    for i in range(n):
        ice = len(zs[i].rs[0].sig)-1
        tb[i] = 10**zs[i].mpa[ice]
        t0[i] = 10**(zs[i].mpa[ice]-zs[i].mpe[ice])
        t1[i] = 10**(zs[i].mpa[ice]+zs[i].mpe[ice])
        r[i] = zs[i].ds[0].wk[0]/zs[i].ds[0].wk[1]
        re[i] = sqrt((zs[i].ds[0].we[0]/zs[i].ds[0].wk[0])**2 +
                     (zs[i].ds[0].we[1]/zs[i].ds[0].wk[1])**2) * r[i]
    errorbar(tb, r, yerr=re, xerr=[tb-t0, t1-tb], marker='o', capsize=3, fmt=' ')
    xscale('log')
    yscale('linear')
    xlim(0.1, 10)
    ylim(3, 13)
    xlabel('Collision Energy (eV)')
    ylabel('s/p ratio')
    if sav != '':
        savefig(sav)
    return tb,t0,t1,r,re
        
def plot_ce(zs, alab=[], sav=''):
    labs = []
    for z in zs:
        ice = len(z.rs[0].sig)-1
        proj = fac.ATOMICSYMBOL[z.ds[0].z]
        tgt = z.ds[0].tgt
        op = len(labs) > 0
        plot_hp(z, ice, xsc=1, op=op, xlab='Collision Energy (eV)')
        labs.append(proj+'+'+tgt)
    for i in range(len(alab)):
        labs[i] = labs[i] + ' ' + alab[i]
    legend(labs)
    xlim(0.05,10)
    ylabel('Probability (arb. unit)')
    if sav != '':
        savefig(sav)

def lznm1e(e, t, zs=range(5, 21), md=0, src='fac'):
    r = zeros(len(zs))
    for i in range(len(zs)):
        lzf = lzfn(zs[i], t, src)
        es,nm = lznm(lzf, md=md)
        f = interpolate.interp1d(log(es), nm)
        r[i] = f(log(e))
    return zs, r

def lznm(lzf, md=0):
    d = loadtxt(lzf)
    nx,nn = d.shape
    nm = zeros(nx)
    es = zeros(nx)
    x = arange(nn-2,0,-1)
    for i in range(nx):
        y = d[i][1:-1]
        m = argmax(y)
        m0 = m-2
        if m0 <= 0:
            m0 = 0
        es[i] = d[i][0]
        if md == 0:
            nm[i] = sum(y[m0:]*x[m0:])/sum(y[m0:])
        else:
            nm[i] = x[m]
    return es,nm

def lznd(lzf, e):
    d = loadtxt(lzf, unpack=1)
    nn,nx = d.shape
    y = zeros(nn-2)
    xe = log(e)
    x = log(d[0])
    
    for i in range(nn-2):
        f = interpolate.interp1d(x, d[-i-2],
                                 bounds_error=False,
                                 fill_value='extrapolate')
        y[i] = f(xe)
    return arange(nn-2)+1, y

def plot_lzcmp(z, tgt, op=0,
               lzd='/Users/yul20/src/Kronos_v3.1/CXDatabase/Projectile_Ions'):
    if not op:
        clf()
    a = fac.ATOMICSYMBOL[z]
    lzf0 = '%s/%s/Charge/%d/Targets/%s/%s%d+%s_sec_faclz_nres.cs'%(lzd,a,z,tgt,a.lower(),z,tgt.lower())
    lzf1 = '%s/%s/Charge/%d/Targets/%s/%s%d+%s_sec_mclz_nres.cs'%(lzd,a,z,tgt,a.lower(),z,tgt.lower())
    r0 = loadtxt(lzf0, unpack=1)
    r1 = loadtxt(lzf1, unpack=1)
    loglog(r0[0], r0[-1])
    loglog(r1[0], r1[-1])

def fit_ndist(z, i):
    xn = array(list(z.ds[i].ns))
    yn = z.ds[i].an[:-1]
    ye = z.ds[i].ae[:-1]
    m = argmax(yn)
    m0 = m-1
    if m0 < 0:
        m0 = 0
    ye = 2*ye/yn[m]
    yn /= yn[m]
    xn = xn[m0:]
    yn = yn[m0:]
    ye = ye[m0:]
    ice = len(z.rs[0].sig)-1
    def fitfun(x, p):
        xm,ym = lznd(z.ds[i].lzf, 10**p)
        k = int32(x-xm[0])
        w = where((k>=0)&(k<len(xm)))
        y = zeros(len(x))
        y[w] = ym[k[w]]
        return y/max(y)
    e0 = z.mpa[ice]
    de0 = z.mpe[ice]
    d = {'xd': xn, 'yd':yn, 'ye':ye,
         'mp':array([e0]),
         'mp0':array([e0-100*de0]),
         'mp1':array([e0+100*de0]),
         'smp':array([0.1*de0]),
         'ftp':0}
    mcmc.mcmc(fitfun, d, 2000, npr=500)
    """
    popt,pcov = optimize.curve_fit(fitfun, xn, yn, sigma=ye,
                                   bounds=(e0-100*de0,e0+100*de0))
    """
    
    return (e0,de0,d['mpa'][0],d['mpe'][0],d)
    
def plot_cecmp(zs, op=0, sav=''):
    if not op:
        clf()
    plt.rcParams.update({'font.size':13})
    plt.rcParams.update({'text.usetex': False})
    plt.subplots_adjust(bottom=0.15,left=0.15,right=0.95,top=0.95)
    syms=['o', 'd', 's', '^']
    cols=['r','g','b','m']
    r1 = loadtxt('cendist.txt')
    #r1 = transpose(array([fit_ndist(z, 0) for z in zs]))
    #r2 = transpose(array([fit_ndist(z, 1) for z in zs]))
    x = 10**r1[0]
    dx0 = (x-10**(r1[0]-r1[1]))
    dx1 = (10**(r1[0]+r1[1])-x)
    y = 10**r1[2]
    dy0 = 2*((y-10**(r1[2]-r1[3])))
    dy1 = 2*((10**(r1[2]+r1[3])-y))
    #z = 10**r2[2]
    #dz0 = z-10**(r2[2]-r2[3])
    #dz1 = 10**(r2[2]+r2[3])-z
    labs = []
    for i in range(len(x)):
        a = fac.ATOMICSYMBOL[zs[i].ds[0].z]
        tgt = zs[i].ds[0].tgt
        errorbar([x[i]], [y[i]], xerr=([dx0[i]],[dx1[i]]), yerr=([dy0[i]],[dy1[i]]), marker=syms[i], capsize=3, fmt=' ', color=cols[i], markersize=8, fillstyle='none')
        labs.append('%s+%s'%(a, tgt))
        text(x[i]*1.15, y[i]/1.25, tgt)
    #errorbar(x, z, xerr=(dx0,dx1), yerr=(dz0,dz1), marker='^', capsize=3, fmt=' ')
    #legend(labs)
    
    xscale('log')
    yscale('log')
    #legend(['H-like', 'He-like'])
    xlabel('Collision Energy from Cascade Model (eV/amu)')
    ylabel('Collision Energy from n Distribution (eV/amu)')
    emin = 0.1
    emax = 10.0
    xlim(emin,emax)
    ylim(emin,emax)
    plot([emin,emax],[emin,emax], color='k')
    if sav:
        savefig(sav)
        
def lzfn(z,tgt,md,
         lzd='/Users/yul20/src/Kronos_v3.1/CXDatabase/Projectile_Ions'):
    a = fac.ATOMICSYMBOL[z]
    lzf = '%s/%s/Charge/%d/Targets/%s/%s%d+%s_sec_%slz_nres.cs'%(lzd,a,z,tgt,a.lower(),z,tgt.lower(),md)
    return lzf

def plot_nc(zs, k=1, op=0, sav='',
            lzd='/Users/yul20/src/Kronos_v3.1/CXDatabase/Projectile_Ions'):
    if not op:
        clf()
    k -= 1
    z = zs[k].ds[k].z
    a = fac.ATOMICSYMBOL[z]
    cols = ['k','b','r','g','o']
    labs = []
    for i in range(len(zs)):
        zi = zs[i]
        proj = fac.ATOMICSYMBOL[zi.ds[k].z]
        tgt = zi.ds[k].tgt
        #lzf = '%s/%s/Charge/%2d/Targets/%s/%s%2d+%s_sec_faclz_nres.cs'%(lzd,a,z,tgt,a.lower(),z,tgt.lower())
        lzf = zi.ds[k].lzf
        print(lzf)
        es,nm = lznm(lzf)
        w = where((es>=0.05)&(es<=50))
        semilogx(es[w], nm[w], color=cols[i])
        labs.append(proj+'+'+tgt)
    legend(labs)
    for i in range(len(zs)):        
        zi = zs[i]
        ice = len(zi.rs[0].sig)-1
        proj = fac.ATOMICSYMBOL[zi.ds[k].z]
        tgt = zi.ds[k].tgt
        eb = 10**(zi.mpa[ice])
        e0 = 10**(zi.mpa[ice]-zi.mpe[ice])
        e1 = 10**(zi.mpa[ice]+zi.mpe[ice])
        xn = array(zi.ds[k].ns)
        im0 = min(1000,int((0.5*zi.imp)))
        ap = zi.hmp[im0:zi.imp,zi.ia[k]:zi.ia[k+1]]
        xm = zeros(len(ap[:,0]))
        for j in range(len(ap[:,0])):            
            m = argmax(ap[j])
            m0 = m-2
            if m0 <= 0:
                m0 = 0
            xm[j] = sum(xn[m0:]*ap[j,m0:])/sum(ap[j,m0:])
        xe = 2*std(xm)
        xm = mean(xm)
        errorbar([eb], [xm], xerr=([e0],[e1]), yerr=xe, linewidth=0.75,
                 capsize=3, marker='o', fmt=' ', color=cols[i])
    xlabel('Collision Energy (eV)')
    if k == 0:
        ylabel('H-like Mean Capture n')
    else:
        ylabel('He-like Mean Capture n')
    if sav != '':
        savefig(sav)
        
def plot_feld(zs, k=1, sav='', ap=plt, ileg=1):
    labs = []
    ymin = 1e30
    ymax = -1e30
    syms=['o','d','s','^']
    lst=['-','dashed','dotted','-.']
    dx = array([-0.3,-0.1,0.1,0.3])*0.65
    cols = ['r','g','b','m']
    i = 0
    for z in zs:
        proj = fac.ATOMICSYMBOL[z.ds[0].z]
        tgt = z.ds[0].tgt
        plot_ldist(z, k=k, op=1, ap=ap, marker=syms[i],
                   lst=lst[i], dx=dx[i], color=cols[i])
        labs.append(proj+'+'+tgt)
        yr = ylim()
        ymin = min(ymin,yr[0])
        ymax = max(ymax,yr[1])
        i=i+1
    if ileg==1:
        ap.legend(labs, fontsize='small')
    ap.set_ylim(ymin, ymax)
    if sav != '':
        savefig(sav)

def plot_feld_all(zn, zs):
    plt.rcParams.update({'font.size':15})
    plt.rcParams.update({'text.usetex': True})
    f,a = plt.subplots(3,1)
    a[0].text(-0.3, 0.85, '(a)')
    a[1].text(-0.3, 0.85, '(b)')
    a[2].text(-0.3, 0.85, '(c)')
    f.set_size_inches(5,9)
    f.subplots_adjust(hspace=0,wspace=0,bottom=0.15,left=0.17,top=0.95,right=0.95)
    plot_feld(zn, k=1, ap=a[0],ileg=1)
    plot_feld(zn, k=2, ap=a[1],ileg=0)
    plot_feld(zs, k=1, ap=a[2],ileg=0)
    a[0].tick_params(direction='in')
    a[1].tick_params(direction='in')
    a[0].set_xticks(range(6))
    a[1].set_xticks(range(6))
    a[2].set_xticks(range(6))
    a[0].set_xticklabels([])
    a[1].set_xticklabels([])
    a[0].set_xlim(-0.5,5.5)
    a[1].set_xlim(-0.5,5.5)
    a[2].set_xlim(-0.5,5.5)
    a[0].set_ylim(-0.05,0.95)
    a[1].set_ylim(-0.05,0.95)
    a[2].set_ylim(-0.05,0.95)
    a[2].set_xlabel(r'Orbital Angular Momentum $\mathit{l}$')
    a[1].set_ylabel(r'Capture Fraction')
    savefig('figures/angular_momentum.eps')
    
def plot_fend(zs, k=1, sav=''):
    labs = []
    for z in zs:
        proj = fac.ATOMICSYMBOL[z.ds[0].z]
        tgt = z.ds[0].tgt
        if k == 1:
            tgt += '  H-Like'
        elif k == 2:
            tgt += '  He-Like'
        op = len(labs) > 0
        plot_ndist(z, k=k, op=op)
        labs.append(proj+'+'+tgt)
    legend(labs)
    if sav != '':
        savefig(sav)
    
def deme(z, r0 = 5e2,
         e0=10.0, xmin=1e-3, alpha=0.0,
         tgt='H', tmass=1.0,
         lzd='/Users/yul20/src/Kronos_v3.1/CXDatabase/Projectile_Ions'):    
    a = fac.ATOMICSYMBOL[z]
    lzf='%s/%s/Charge/%2d/Targets/%s/%s%2d+%s_sec_faclz_nres.cs'%(lzd,a,z,tgt,a.lower(),z,tgt.lower())
    r = loadtxt(lzf, unpack=1)
    xe = r[0]
    xs = r[-1]*sqrt(xe)
    m1 = fac.ATOMICMASS[z]
    g = (xe/e0)**alpha*(r0*sqrt(e0)/xs)*(m1*tmass)/(m1+tmass)**2
    gi = interpolate.interp1d(log(xe/e0), log(g),
                              bounds_error=False, fill_value='extrapolate')
    x = 10**arange(log10(xmin), 0.0, 0.005)
    lx = log(x)
    gg = exp(gi(lx))
    dlx = lx[1]-lx[0]
    ef = cumsum(dlx/gg)
    ef = ef[-1]-ef
    y = 1.0/(x*gg)*exp(-ef)
    ys = cumsum(y*x)
    ys = (1-ys/ys[-1])
    
    return r,x*e0,gg,ef,y,ys

def plot_hr(z, k, tgt, n, md=0, sw=0, kmax=5, op=0, sav='', ylog=0, yr=None):
    if not op:
        clf()
        
    plt.rcParams.update({'font.size':15})
    plt.subplots_adjust(bottom=0.15, top=0.95, right=0.95)
    a = fac.ATOMICSYMBOL[z]
    fn = '%s%02d%sKL%d.pkl'%(a,k,tgt,sw)
    es,trm,eus,r,ys = load_basis(fn)
    if k == 1:
        if md == 0:
            w0 = where((r.ir0 == 0)&(r.ir1 < 4))
            w1 = where((r.ir0 == 0)&(r.ir1 >= 4))
            ylab = r'H-like $H=F_{>2}/F_2$'
        elif md == 1:            
            w0 = where((r.ir0 == 0)&(r.ir1<3))
            w1 = where((r.ir0 == 0)&(r.ir1 == 3))
            ylab = r'Ly$_{\alpha 1}$/Ly$_{\alpha 2}$'
        elif md == 2:
            w0 = where((r.ir0 == 0)&(r.ir1 == 3))
            w1 = where((r.ir0 == 0)&(r.ir1 == 2))
            ylab = r'2s-1s/Ly$_{\alpha 1}$'
        elif md == 3:
            w0 = where((r.ir0 == 0)&(r.ir1 == 3))
            w1 = where((r.ir0 == 0)&(r.ir1 == 1))
            ylab = r'2p-1s/Ly$_{\alpha 1}$'
        else:
            w1 = where((r.ir0 == 0)&(r.ir1<3))
            w0 = where((r.ir0 == 0)&(r.ir1 == 3))
            ylab = r'Ly$_{\alpha 2}$/Ly$_{\alpha 1}$'
    else:
        if md == 0:
            w0 = where((r.ir0 == 0)&(r.ir1 < 7))
            w1 = where((r.ir0 == 0)&(r.ir1 >= 7))
            ylab = r'He-like $H=F_{>2}/F_2$'
        elif md == 1:
            w0 = where((r.ir0 == 0) & (r.ir1 == 6))
            w1 = where((r.ir0 == 0) & (r.ir1 < 6))
            ylab = r'He-like $G=(x+y+z)/w$'
        elif md == 2:
            w0 = where((r.ir0 == 0) & (r.ir1 > 1) & (r.ir1 < 6))
            w1 = where((r.ir0 == 0) & (r.ir1 == 1))
            ylab = r'He-like $R=z/(x+y)$'
        elif md == 3:
            w1 = where((r.ir0 == 0) & (r.ir1>6))
            w0 = where((r.ir0 == 0)&(r.ir1==6))
            ylab = r'He-like $H^{\prime}=F_{>2}/w$'
        elif md == 4:
            w0 = where((r.ir0 == 0) & (r.ir1 >= 6))
            w1 = where((r.ir0 == 0) & (r.ir1 < 6))
            ylab = r'$G^{\prime}=(x+y+z)/(w+F_{>2}$)'
        elif md == 5:
            w0 = where((r.ir0 == 0))
            w1 = where((r.ir0 == 0) & (r.ir1 == 1))
            ylab = r'z/tot'
        elif md == 6:
            w0 = where((r.ir0 == 0))
            w1 = where((r.ir0 == 0) & (r.ir1 >= 7))
            ylab = r'$F_{>2}/tot$'
            
    w0 = w0[0]
    w1 = w1[0]
    y0 = sum(ys[:,:,:,w0], 3)
    y1 = sum(ys[:,:,:,w1], 3)
    hr = y1.copy()
    hr[:,:,:] = 0.0
    w = where(y0 > 0)
    hr[w] = y1[w]/y0[w]
    labs = []
    ss = ['s','p','d','f','g','h']
    syms=['^','<','s','o','d','v']
    cols = ['k','r','g','b','y','m']
    for i in range(kmax+1):
        if i%2 == 0:            
            semilogx(eus, hr[:,n,i],color=cols[i], marker=syms[i])
        else:
            semilogx(eus, hr[:,n,i], color=cols[i], marker=syms[i], fillstyle='none')
        labs.append('%d%s'%(n,ss[i]))
    if ylog:
        yscale('log')
    if not yr is None:
        ylim(yr)
    legend(labs, fontsize='x-small')
    xlabel('Relative Collision Energy (eV/amu)')
    ylabel(ylab)
        
    if sav != '':
        savefig(sav)

def sum_hr(z, i, nr, ir, sn):
    w1 = where(z.ds[i].ir0 == 0)[0]
    wn = z.ds[i].an[:-1].copy()
    if sn == 1:
        m = argmax(wn)
        wn[:] = 0.0
        wn[m] = 1.0
    elif sn > z.ds[i].ns[0]:
        m = sn-z.ds[i].ns[0]
        wn[:] = 0.0
        wn[m] = 1.0
        
    r = 0.0
    for j in range(len(nr)):
        n = nr[j]
        k = ir[j]
        if n > 2:
            nx = int32(z.ds[i].rd.v/100)
            w = where((z.ds[i].ir0 == 0)&(nx[z.ds[i].ir1]==n))
            r += matmul(transpose(sum(z.ds[i].ai[:,w[0],:],1)),wn)
        else:
            ir1 = [int(d) for d in str(k)]
            w = z.ds[i].ir1 == ir1[0]
            for i1 in range(1,len(ir1)):
                w |= z.ds[i].ir1 == ir1[i1]
            w = where(w & (z.ds[i].ir0 == 0))
            r += matmul(transpose(sum(z.ds[i].ai[:,w[0],:],1)),wn)
    rt = matmul(transpose(sum(z.ds[i].ai[:,w1,:],1)),wn)
    r *= z.ds[i].an[-1]
    rt *= z.ds[i].an[-1]
    
    return r, rt

def plot_hrpoly(zs, ix, iy, ap=plt, op=0, sav='',
                md=0, sn=1, tsr=-1., pref='NewCX',
                fc='c', nodata=0, ileg=0, ixd=None):
    if not op:
        clf()
    xlab = ix[0]
    ylab = iy[0]
    xk = ix[1]
    yk = iy[1]
    xr = ix[2]
    yr = iy[2]
    
    labs = []
    cols = ['r','g','b','m']
    ic = ['h','he']
    syms=['o', 'd', 's', '^']
    for i in range(len(zs)):
        a = fac.ATOMICSYMBOL[zs[i].ds[0].z]
        tgt = zs[i].ds[0].tgt
        if md == 0:
            fn1 = '%s%s%s_%slike.txt'%(a,tgt,pref,ic[xk-1])
            fn2 = '%s%s%s_%slike.txt'%(a,tgt,pref,ic[yk-1])
        else:
            fn1 = '%srec_%slike.txt'%(tgt.lower(),ic[xk-1])
            fn2 = '%srec_%slike.txt'%(tgt.lower(),ic[yk-1])
            
        r1 = loadtxt(fn1, unpack=1)
        r2 = loadtxt(fn2, unpack=1)
        ax = sum(r1[5][xr])
        ax2 = sum(r1[6][xr]**2)
        ay = sum(r2[5][yr])
        ay2 = sum(r2[6][yr]**2)
        tx = sum(r1[5])
        tx2 = sum(r1[6]**2)
        ty = sum(r2[5])
        ty2 = sum(r2[6]**2)
        bx = tx-ax
        bx2 = tx2-ax2
        by = ty-ay
        by2 = ty2-ay2
        rx = ax/tx
        ry = ay/ty
        dx = rx**2*(sqrt(bx2/bx**2+ax2/ax**2)*bx/ax)
        dy = ry**2*(sqrt(by2/by**2+ay2/ay**2)*by/ay)
        if nodata > 0:
            continue
        ap.errorbar([rx], [ry], xerr=[dx], yerr=[dy],
                    marker=syms[i], markersize=7, linewidth=1.0,
                    capsize=3, fmt=' ', color=cols[i], fillstyle='none')
        labs.append('%s+%s'%(a,tgt))

    if ileg == 1:
        ap.legend(labs)
    if ap == plt:
        ap.xlabel(xlab)
        ap.ylabel(ylab)

    astr = ['s', 'p', 'd', 'f', 'g', 'h']
    asym = ['o','d','<','^','v','s']
    labs = []
    if ixd is None:
        ixd = range(len(astr))
    for i in range(len(zs)):
        nd = len(zs[i].ds)
        if xk == 1 or nd == 2:
            rx,rxt = sum_hr(zs[i], xk-1,
                            int32(r1[1][xr]), int32(r1[2][xr]), sn)
        else:
            rx1,rxt1 = sum_hr(zs[i], 1,
                              int32(r1[1][xr]), int32(r1[2][xr]), sn)
            rx3,rxt3 = sum_hr(zs[i], 2,
                              int32(r1[1][xr]), int32(r1[2][xr]), sn)
            if tsr < 0:
                rx = rx1 + rx3
                rxt = rxt1 + rxt3
            else:
                rx = rx1/zs[i].ds[1].an[-1] + tsr/3*rx3/zs[i].ds[2].an[-1]
                rxt = rxt1/zs[i].ds[1].an[-1] + tsr/3*rxt3/zs[i].ds[2].an[-1]
        if yk == 1 or nd == 2:
            ry,ryt = sum_hr(zs[i], yk-1,
                            int32(r2[1][yr]), int32(r2[2][yr]), sn)
        else:
            ry1,ryt1 = sum_hr(zs[i], 1,
                              int32(r2[1][yr]), int32(r2[2][yr]), sn)
            ry3,ryt3 = sum_hr(zs[i], 2,
                              int32(r2[1][yr]), int32(r2[2][yr]), sn)
            if tsr < 0:
                ry = ry1 + ry3
                ryt = ryt1 + ryt3
            else:
                ry = ry1/zs[i].ds[1].an[-1] + tsr/3*ry3/zs[i].ds[2].an[-1]
                ryt = ryt1/zs[i].ds[1].an[-1] + tsr/3*ryt3/zs[i].ds[2].an[-1]
            
        rx /= rxt
        ry /= ryt
        for i0 in range(len(rx)):
            for i1 in range(len(rx)):
                if i1 == i0:
                    continue
                for i2 in range(len(rx)):
                    if i2 != i0 and i2 != i1:
                        ap.fill([rx[i0],rx[i1],rx[i2],rx[i0]],
                                [ry[i0],ry[i1],ry[i2],ry[i0]], color='0.7')
            ap.plot(rx[i0], ry[i0], color='k', marker=asym[i0],
                    fillstyle='none', markersize=7, linestyle='none')
            if i == 0:
                labs.append(astr[i0])
        if i == 0:
            ap.plot(rx[ixd], ry[ixd], color='k', linewidth=0.75)
    if ileg == 2:
        ap.legend(labs,labelspacing=0.3)
            
    if sav != '':
        savefig(sav)

def plot_hrpoly2(zn, zs):
    plt.rcParams.update({'font.size':15})
    f,a = plt.subplots(2,1)
    a[0].set_xlim(-0.05,0.9)
    a[1].set_xlim(-0.05,0.9)
    a[0].set_ylim(-0.05,0.375)
    a[1].set_ylim(-0.05,0.375)
    a[0].text(0.8, 0.05, '(a)')
    a[1].text(0.8, 0.05, '(b)')
    a[0].tick_params(direction='in')
    a[0].set_xticks(arange(0.0,0.91,0.1))
    a[0].set_xticklabels([])
    a[1].set_xticks(arange(0.0,0.91,0.1))
    f.set_size_inches(5.5,8)
    f.subplots_adjust(hspace=0,wspace=0,bottom=0.1,left=0.175,right=0.95,top=0.95)
    plot_hrpoly(zn, 
                ('r1', 1, range(9,16)),
                ('r2', 2, range(11,18)),
                ap=a[0], op=1, ileg=2,
                ixd=[0,1,2,3,4,5,0])
    plot_hrpoly(zs, 
                ('r1', 1, range(9,16)),
                ('r2', 2, range(11,18)),
                ap=a[1], op=1, ileg=1,
                ixd=[0,1,3,4,0])

    a[0].set_ylabel(r'He-like n>9')
    a[1].set_ylabel(r'He-like n>9')
    a[1].set_xlabel(r'H-like n>9')
    savefig('figures/old_new_poly_highn.eps')
    
def plot_hrpoly1(zn, zs):
    plt.rcParams.update({'font.size':15})
    f,a = plt.subplots(2,1)
    a[0].set_xlim(0,0.7)
    a[1].set_xlim(0,0.7)
    a[0].set_ylim(0.0,0.35)
    a[1].set_ylim(0.05,0.325)
    a[0].text(0.6, 0.05, '(a)')
    a[1].text(0.6, 0.095, '(b)')
    a[0].tick_params(direction='in')
    a[0].set_xticks(arange(0.0,0.71,0.1))
    a[0].set_xticklabels([])
    a[1].set_xticks(arange(0.0,0.71,0.1))
    f.set_size_inches(5.5,8)
    f.subplots_adjust(hspace=0,wspace=0,bottom=0.1,left=0.175,right=0.95,top=0.95)
    plot_hrpoly(zn, 
                (r'Ly$_{\alpha 1}$', 1, [1]),
                (r'Ly$_{\alpha 2}$', 1, [0]),
                ap=a[0], op=1, ileg=2,
                ixd=[0,1,2,3,4,5,0])
    plot_hrpoly(zs, 
                (r'Ly$_{\alpha 1}$', 1, [1]),
                (r'Ly$_{\alpha 2}$', 1, [0]),
                ap=a[1], op=1, ileg=1,
                ixd=[0,1,5,4,3,2,0])

    a[0].set_ylabel(r'Ly$_{\alpha 2}$')
    a[1].set_ylabel(r'Ly$_{\alpha 2}$')
    a[1].set_xlabel(r'Ly$_{\alpha 1}$')
    savefig('figures/old_new_poly_lya12.eps')
    
def plot_tshrpoly(zs, ix, iy, tsr=3.0):
    plot_hrpoly(zs, ix, iy, tsr=1e10, noleg=1)
    plot_hrpoly(zs, ix, iy, tsr=0.0, noleg=1, nodata=1, op=1, fc='r')
    plot_hrpoly(zs, ix, iy, tsr=-1., noleg=1, nodata=1, op=1, fc='y')

def plot_ovsp(zs, xr=[(1.98,2.32),(6.54,7.04),(7.8,9.3)], sav='', wm=0, ierr=0):
    plt.rcParams.update({'font.size':15})
    n = len(zs)
    m = len(xr)
    f,a = plt.subplots(n,m)
    f.set_size_inches(12,3.5)
    f.subplots_adjust(hspace=0,wspace=0)
    if n == 1:
        f.subplots_adjust(bottom=0.18,left=0.1,right=0.95)
    for i in range(n):        
        x = zs[i].sp.xm/1e3
        y = zs[i].sp.yc
        for j in range(m):
            if (n == 1):
                aij = a[j]
            else:
                aij = a[i,j]
            w = where((x>=xr[j][0])&(x<=xr[j][1]))
            sy = 1/max(y[w])
            aij.plot(x[w], y[w]*sy, color='b', marker='o', markersize=3, fillstyle='none', linestyle=' ', markeredgewidth=0.3)
            if wm > 0:
                aij.plot(x[w], zs[i].ym[w]*sy, color='r', linestyle='-', linewidth=0.8)
            aij.set_xlim(xr[j][0],xr[j][1])
            aij.set_ylim(-0.1,1.1)
            if j == 0:
                aij.set_xticks([2.0,2.1,2.2,2.3])
            elif j == 1:
                aij.set_xticks([6.6,6.7,6.8,6.9,7.0])
            elif j == 2:
                aij.set_xticks([8.0,8.5,9.0])
            aij.set_yticks([0.0,0.5,1.0])
            if i < n-1:
                aij.set_xticklabels([])
            if j > 0:
                aij.set_yticklabels([])
            if j == 1:
                aij.text(6.78, 0.7, 'Fe+'+zs[i].ds[0].tgt)
            if j == 0:
                aij.text(2.15, 0.95, 'max cnt: %d'%int(1/sy))
            elif j == 1:
                aij.text(6.8, 0.95, 'max cnt: %d'%int(1/sy))
            else:
                aij.text(8.25, 0.95, 'max cnt: %d'%int(1/sy))
            if ierr > 0:
                xm = x[w][::ierr]
                ym = zs[i].ym[w][::ierr]*sy
                ye = sqrt(y[w])[::ierr]*sy
                
                aij.errorbar(xm, ym,
                             yerr=(ye,ye), capsize=2, linewidth=0.85,
                             linestyle='none', color='r')
                             
    f.text(0.04,0.5,'Intensity (arb. units)',va='center',rotation='vertical')
    f.text(0.5,0.03,'Energy (keV)', ha='center')
    if n > 1:
        a00 = a[0,0]
        a01 = a[0,1]
        a02 = a[0,2]
    else:
        a00 = a[0]
        a01 = a[1]
        a02 = a[2]
        
    a00.text(2.03,0.975,'He-like')
    a00.text(2.23,0.7,'H-like')
    a01.text(6.65,0.8, r'He$_{\alpha}$')
    a01.text(6.93,0.4,r'Ly$_{\alpha}$')
    a02.text(7.83,0.575,r'He$_{\beta}$')
    a02.text(8.14,0.4, r'Ly$_{\beta}$')
    a02.text(8.29,0.35, r'He$_{\gamma}$')
    a02.text(8.45,0.23, r'He$_{\delta}$')
    a02.text(8.66,0.325, r'Ly$_{\gamma}$')
    a02.text(8.86,0.2, r'Ly$_{\delta}$')
    a02.text(8.975,0.95,r'n=14')
    if n > 1:
        a[1,2].text(8.95,0.975,r'n=14')
        a[2,2].text(8.95,0.975,r'n=14')
        a[3,2].text(8.95,0.975,r'n=12')
    if sav != '':
        f.savefig(sav)

def plot_h2sp(zs):
    plot_ovsp(zs[:1], wm=1, sav='figures/spec_h2.eps', ierr=0)
    
def plot_wdce(zs, sav='', op=0):
    clf()
    r = mcpk.plot_widths()
    clf()
    ice = len(z.rs[0].sig)-1
    t = transpose([(z.mpa[ice],z.mpe[ice]) for z in zs])
    tb = 10**t[0]
    t0 = 10**(t[0])-10**(t[0]-t[1])
    t1 = 10**(t[0]+t[1])-10**(t[0])
    errorbar(r[0], tb, xerr=r[1], yerr=(t0,t1), marker='o', capsize=3, fmt=' ')
    xlabel('Decrease in FWHM (eV)')
    ylabel('Collision Energy (eV/amu)')
    
    if sav != '':
        savefig(sav)
        
def plot_tsr(zs, sav='', op=0):
    if not op:
        clf()
    isd = zs[0].iid[1]
    itd = zs[0].iid[2]
    r = transpose([(z.mpa[isd],z.mpe[isd],z.mpa[itd],z.mpe[itd]) for z in zs])
    y = r[2]/r[0]
    ye = sqrt((r[1]/r[0])**2+(r[3]/r[2])**2)*y
    errorbar(range(4), 3*y, yerr=3*ye, marker='o', capsize=3, fmt=' ')
    ax = gca()
    t = ax.set_xticks([0,1,2,3])
    t = ax.set_xticklabels(['H2','H','N2','He'])
    xlabel('Neutral Targets')
    ylabel('He-like Triplet/Singlet Capture Ratio')
    if sav != '':
        savefig(sav)
        
        
    
