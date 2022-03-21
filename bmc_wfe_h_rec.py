import cxfit
from pylab import *

ft = 'FeCXEff.txt'
fn = 'FeHNewCX.txt'
ns1 = range(6,17)
ns2 = range(6,17)
ns3 = range(6,15)
js = [0, 0, 0]
z0 = cxfit.load_pkl('wfe_h.pkl')
yc = poisson(z0.ym)
yd = yc/z0.sp.eff
ye = sqrt(yc)/z0.sp.eff
z=cxfit.fit_spec((z0.sp.elo,z0.sp.ehi,yd,ye), 26, [1, 2, 3],
                 [ns1,ns2,ns3], js,
                 [2.0,3.0,3.0,3.0,3.0,3.0,3.0,0.1,1.0],
                 0.5, 0, er=[1e3, 2.3e3, 6.5e3, 9.4e3],
                 wsig=[(1.8,2.2),(2.,5.0),(2.,5.0),
                       (2.,5.0),(2.,5.0),(2.,5.0),(2.,5.0),(0,0.2),(0.0,10.0)],
                 eip=[2.5e3,6.65e3, 6.67e3, 6.85e3, 7.5e3, 8.5e3,-3e3],
                 #ecf='data/SK.ecf',
                 fixnd = [-1, -1, -1],
                 fixld = [-1, -1, -1],
                 sav=['wfe_h_rec.pkl',500],
                 kmax=6, wreg=0.0,
                 nmc=5000)

