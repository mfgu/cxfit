from pylab import *
from scipy import integrate, interpolate, special, stats, linalg
import pickle
import time, os
from collections import namedtuple

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

def mcmc(fun, d, nm, sav=[], racc=0.4, npr=100):        
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
    ifx = d.get('ifx')
    cc = d.get('cc')
    ftp = d['ftp']
    if ftp == 0:
        def flike(d, ip):
            xd = d['xd']
            yd = d['yd']
            ye = d['ye']
            ym = fun(xd, d['mp'])
            r = -0.5*sum(((yd-ym)/ye)**2)
            return r
        lnlikely = flike
    else:
        lnlikely = fun
    
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
            if mp1[ip] <= mp0[ip]:
                continue
            if ifx != None:
                if ifx[ip] > 0:
                    continue
                if cc != None:
                    w = (where(ifx == ip))[0]
                    for i in w:
                        if i == ip:
                            continue
                        uc = cc[i]
                        xp = mp[ip]
                        if uc[2]:
                            xp = xp**uc[2]
                        if uc[0]:
                            xp = uc[0]*xp
                        if uc[1]:
                            xp += uc[1]
                        if uc[3]:
                            xp = xp**uc[3]
                        mp[i] = xp
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
            pp.append(time.time()-t0)
            print('imc: %6d %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E'%tuple(pp))

        savenow = False
        if i == nm-1:
            savenow = True
        elif nsav > 0 and (i+1)%nsav == 0:
            savenow = True
        elif tsav != None and os.path.isfile(tsav):
            savenow = True
            os.system('rm '+tsav)            
        if savenow:
            i0 = min(nburn,int((i+1)/2))
            d['hmp'] = hmp
            d['fmp'] = fmp        
            d['smp'] = smp
            d['ene'] = ene
            d['frej'] = frej
            d['rrej'] = rrej
            d['mpa'] = mean(hmp[i0:i+1],0)
            d['mpe'] = std(hmp[i0:i+1],0)
            if (fsav != None):
                print('pickling: %s %10.3E'%(fsav, time.time()-t0))
                fs = open(fsav, 'wb')
                pickle.dump(d, fs)
                fs.close()

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
