import cxfit
import pickle
from pylab import *

fn = 'Ar_CX_6keV_Raw.dat'
ns1 = [8,9,10]
ns2 = [7,8,9]
js = [0, 1, 3]
f = open('zar_mpik.pkl','r')
z0 = pickle.load(f)
f.close()
yc = poisson(z0.ym)
yd = yc/z0.sp.eff
ye = sqrt(yc)/z0.sp.eff
sig = z0.rs[0].sig
sig0 = sig*0.99999
sig1 = sig*1.00001
z=cxfit.fit_spec((z0.sp.elo, z0.sp.ehi, yd, ye), 18, [1, 2, 2], [ns1,ns2,ns2], js,
#                 [ 59.77338742,0.31329662,0.07111712,0.35941966,0.0],
                 z0.rs[0].sig,
                 1.0, 0,sav=['zar_mpik_rec.pkl',500],
                 ecf='data/ArK.ecf',
                 bkgd=(((1e3,1.0,1e10,500),
                         (250.0,50.0,500.0,25.0),
                         (3e3,2.5e3,3.9e3,1e2)),cxfit.ar_bkgd,0.05),
                 wsig=[(sig0[i],sig1[i]) for i in range(len(sig0))],
                 fixnd=[-1,0,0],wreg=5.0,
                 kmax=7)
